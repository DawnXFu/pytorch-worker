import gc  # 添加这一行，用于垃圾回收
import logging
import os
import shutil
import time  # 添加这一行，用于格式化当前时间
from collections import defaultdict
from timeit import default_timer as timer  # 确保这一行存在

import torch
from sympy import im

# 添加混合精度训练所需的导入
from torch.amp import GradScaler, autocast  # 从torch.amp导入而不是torch.cuda.amp
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from tools.eval_tool import gen_time_str, output_value, valid
from tools.init_tool import checkpoint, init_test_dataset

logger = logging.getLogger(__name__)


def train(parameters, config, gpu_list, do_test=False):
    # —— 1. 准备阶段 ——
    model = parameters["model"]
    optimizer = parameters["optimizer"]
    scheduler = parameters["scheduler"]
    scaler = parameters["scaler"]
    writer = parameters["writer"]
    start_epoch = parameters["start_epoch"]
    global_step = parameters["global_step"]

    metrics = parameters["metrics"]
    output_function = parameters["output_function"]

    # 添加总训练开始时间
    train_start_time = timer()
    logger.info(f"开始训练: {start_epoch} -> {config.getint('train', 'epoch')} epochs")

    # —— 2. 主循环 ——
    for epoch in range(start_epoch, config.getint("train", "epoch")):
        # 输出 epoch 开始时间
        epoch_start_time = timer()
        logger.info(f"Epoch {epoch}/{config.getint('train', 'epoch')} 开始于 {time.strftime('%Y-%m-%d %H:%M:%S')}")

        model.train()
        loss, acc_result, global_step = run_one_epoch(
            model,
            optimizer,
            scheduler,
            scaler,
            parameters["train_dataset"],
            config,
            gpu_list,
            epoch,
            global_step,
            writer,
            metrics,
        )

        # 计算 epoch 耗时
        epoch_time = timer() - epoch_start_time
        remaining_epochs = config.getint("train", "epoch") - epoch - 1
        estimated_remaining_time = epoch_time * remaining_epochs

        # 输出更详细的时间信息
        logger.info(
            f"Epoch {epoch+1} 完成: 耗时 {gen_time_str(epoch_time)}, "
            f"预计剩余时间: {gen_time_str(estimated_remaining_time)}, "
            f"总耗时: {gen_time_str(timer() - train_start_time)}"
        )

        # 记录 epoch 级指标 & checkpoint
        log_epoch(writer, epoch, loss, acc_result, optimizer, config, metrics)
        output_value(
            epoch,
            "train",
            "%d/%d" % (epoch + 1, config.getint("train", "epoch")),
            "%s/%s"
            % (
                gen_time_str(epoch_time),
                gen_time_str(estimated_remaining_time),
            ),
            "%.3lf" % loss,
            output_function(acc_result, config),
            None,
            config,
        )

        # 周期性验证 (添加时间计时)
        if epoch % config.getint("output", "test_time") == 0 and parameters.get("valid_dataset"):
            logger.info(f"开始验证: Epoch {epoch+1}")
            valid_start_time = timer()

            res = valid(model, parameters["valid_dataset"], epoch, writer, config, gpu_list, output_function, metrics)

            valid_time = timer() - valid_start_time
            logger.info(f"验证完成: 耗时 {gen_time_str(valid_time)}")

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # 需要传入验证指标(通常是验证损失)
                scheduler.step(res["loss"])
                logger.info(f"ReDu学习率更新: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

        if isinstance(
            scheduler,
            (
                torch.optim.lr_scheduler.StepLR,
                torch.optim.lr_scheduler.MultiStepLR,
                torch.optim.lr_scheduler.ExponentialLR,
                torch.optim.lr_scheduler.CosineAnnealingLR,
                torch.optim.lr_scheduler.ConstantLR,
                torch.optim.lr_scheduler.LinearLR,
                # 其他基于epoch的调度器
            ),
        ):
            scheduler.step()

        # 周期性保存模型 (添加时间计时)
        if epoch % config.getint("output", "output_time") == 0:
            logger.info(f"保存模型检查点: Epoch {epoch+1}")
            save_start_time = timer()

            output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
            if os.path.exists(output_path):
                logger.warning("Output path exists, check whether need to change a name of model")
            os.makedirs(output_path, exist_ok=True)
            checkpoint(
                os.path.join(output_path, "%d.pkl" % epoch),
                model,
                optimizer,
                epoch,
                global_step,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
            )

            save_time = timer() - save_start_time
            logger.info(f"模型保存完成: 耗时 {gen_time_str(save_time)}")

    # —— 3. 收尾 ——
    total_train_time = timer() - train_start_time
    logger.info(f"训练完成! 总耗时: {gen_time_str(total_train_time)}")
    writer.close()
    return model, optimizer, global_step


def run_one_epoch(
    model,
    optimizer,
    scheduler,
    scaler,
    dataset,
    config,
    gpu_list,
    epoch,
    global_step,
    writer,
    metrics,
):
    """
    单次 epoch 训练：含混合精度、梯度累积、梯度裁剪、OneCycleLR 更新，
    并定期将 step 级指标写入 TensorBoard。
    返回: avg_loss, acc_result, global_step
    """
    model.train()
    total_loss = 0.0
    acc_result = None

    # 配置
    use_amp = config.getboolean("train", "use_amp", fallback=True) and bool(gpu_list)
    accum_steps = config.getint("train", "grad_accumulation_steps", fallback=1)
    tb_interval = config.getint("output", "tb_write_interval", fallback=20)
    max_grad_norm = config.getfloat("train", "max_grad_norm", fallback=None)

    # 添加时间计时
    epoch_start = timer()
    data_time = 0.0
    forward_time = 0.0
    backward_time = 0.0
    opt_time = 0.0
    batch_start = timer()

    # 添加进度显示频率
    log_interval = config.getint("output", "log_interval", fallback=10)
    total_batches = len(dataset)

    # 缓存 step 级指标
    tb_cache = defaultdict(list)

    for step, batch in enumerate(dataset):
        # 记录数据加载时间
        data_time += timer() - batch_start

        # 准备数据
        if gpu_list:
            data = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        else:
            data = {k: Variable(v) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # 梯度累积：第一步清零
        if step % accum_steps == 0:
            optimizer.zero_grad()

        # 前向 + 反向
        forward_start = timer()
        if use_amp:
            with autocast("cuda"):
                out = model(data, config, gpu_list, acc_result, "train")
                loss, acc_result = out["loss"], out["acc_result"]
                loss = loss / accum_steps
            forward_time += timer() - forward_start

            backward_start = timer()
            scaler.scale(loss).backward()
            backward_time += timer() - backward_start
        else:
            out = model(data, config, gpu_list, acc_result, "train")
            loss, acc_result = out["loss"], out["acc_result"]
            loss = loss / accum_steps
            forward_time += timer() - forward_start

            backward_start = timer()
            loss.backward()
            backward_time += timer() - backward_start

        # 累加统计
        loss_val = loss.item() * accum_steps
        total_loss += loss_val

        # 梯度累积完成或最后一批时，更新参数
        opt_start = timer()
        if (step + 1) % accum_steps == 0 or step == len(dataset) - 1:
            # 梯度裁剪
            if max_grad_norm:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # 优化器步 & AMP 更新
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # 以下调度器在每步更新
            if isinstance(
                scheduler,
                (
                    torch.optim.lr_scheduler.OneCycleLR,
                    torch.optim.lr_scheduler.CyclicLR,
                    # 如果配置为按步数更新的CosineAnnealingWarmRestarts
                ),
            ):
                scheduler.step()
        opt_time += timer() - opt_start

        global_step += 1

        # 缓存到 tb_cache
        lr_now = optimizer.param_groups[0]["lr"]
        tb_cache["train_loss"].append((step, loss_val))
        tb_cache["learning_rate"].append((step, lr_now))

        # 定期 flush
        flush_tb(tb_cache, writer, tb_interval, acc_result, metrics, step)

        # 定期打印进度和时间信息
        if (step + 1) % log_interval == 0 or step == total_batches - 1:
            iter_time = timer() - batch_start
            progress = (step + 1) / total_batches * 100
            speed = (step + 1) / (timer() - epoch_start) * 60  # 每分钟处理的批次数

            logger.info(
                f"Epoch: {epoch}, Step: {step+1}/{total_batches} ({progress:.1f}%), "
                f"Loss: {loss_val:.4f}, LR: {lr_now:.6f}, "
                f"Speed: {speed:.1f} batches/min, "
                f"Times: [数据: {data_time/(step+1):.3f}s | "
                f"前向: {forward_time/(step+1):.3f}s | "
                f"反向: {backward_time/(step+1):.3f}s | "
                f"优化: {opt_time/(step+1):.3f}s | "
                f"总计: {iter_time:.3f}s]"
            )

        # 为下一批次准备
        batch_start = timer()

    # epoch 级平均 loss
    avg_loss = total_loss / (step + 1)
    writer.add_scalar("epoch/train_loss", avg_loss, epoch)

    return avg_loss, acc_result, global_step


def flush_tb(
    tb_cache,
    writer,
    interval,
    acc_result,
    metrics,
    step_index,
):
    """
    若缓存中任一 tag 的数据量达到 interval，则批量写入并清空缓存，
    同时写入当前 step 的降水评估指标。
    """
    # 检查是否需要写入
    if not any(len(v) >= interval for v in tb_cache.values()):
        return

    # 写入 train_loss & learning_rate
    for tag, records in tb_cache.items():
        for s, v in records:
            writer.add_scalar(f"step/{tag}", v, s)
        tb_cache[tag].clear()

    # 写入降水评估的 step 级指标
    if acc_result and not acc_result.get("NULL", True):
        for metric, tb_tag in metrics.items():
            if metric in acc_result and acc_result[metric]:
                # 列表最后一个元素即当前 step 的值
                writer.add_scalar(f"step/{tb_tag}", acc_result[metric][-1], step_index)

    writer.flush()


def log_epoch(writer, epoch, avg_loss, acc_result, optimizer, config, metrics):
    """
    在一个 epoch 结束时，向 TensorBoard 写入：
      - 平均训练损失
      - 各参数组学习率
      - 梯度范数（若开启裁剪）
      - 各降水评估指标的平均值
    """
    # 1. 平均训练损失
    writer.add_scalar("epoch/train_loss_avg", avg_loss, epoch)

    # 2. 学习率（每个参数组）
    for i, pg in enumerate(optimizer.param_groups):
        writer.add_scalar(f"epoch/learning_rate/group_{i}", pg["lr"], epoch)

    # 3. 梯度范数
    if config.has_option("train", "max_grad_norm"):
        total_norm = 0.0
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        writer.add_scalar("epoch/grad_norm", total_norm, epoch)

    # 4. 降水评估指标平均值
    if acc_result and not acc_result.get("NULL", True):
        for metric, tag in metrics.items():
            vals = acc_result.get(metric, [])
            if vals:
                avg = sum(vals) / len(vals)
                writer.add_scalar(f"epoch/{tag}", avg, epoch)
