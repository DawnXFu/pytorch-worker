import logging
import os
from collections import defaultdict
from timeit import default_timer as timer

import torch
from cv2 import log
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60
    return "%2d:%02d" % (minute, second)


def output_value(epoch, mode, step, time, loss, info, end, config):
    try:
        delimiter = config.get("output", "delimiter")
    except Exception as e:
        delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 7:
        s += " "
    s = s + str(mode) + " "
    while len(s) < 14:
        s += " "
    s = s + str(step) + " "
    while len(s) < 25:
        s += " "
    s += str(time)
    while len(s) < 40:
        s += " "
    s += str(loss)
    while len(s) < 48:
        s += " "
    s += str(info)
    s = s.replace(" ", delimiter)
    if not (end is None):
        print(s, end=end)
    else:
        print(s)


def valid(model, dataset, epoch, writer, config, gpu_list, output_function, metrics):
    """
    执行验证，使用内存优化策略
    """
    model.eval()

    # 使用较小批次并手动管理缓存
    total_loss = 0
    acc_result = None
    total_samples = 0

    # 添加进度日志
    total_batches = len(dataset)
    log_interval = max(1, total_batches // 10)  # 每10%报告一次

    logger.info(f"开始验证: 共{total_batches}批次，每批{dataset.batch_size}样本")

    tb_cache = defaultdict(list)  # 用于存储 TensorBoard 缓存

    with torch.no_grad():  # 确保不储存梯度
        for step, batch in enumerate(dataset):
            # 主动管理内存
            torch.cuda.empty_cache()

            if gpu_list:
                data = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                data = batch

            # 使用混合精度加速验证且节省内存
            if config.getboolean("train", "use_amp", fallback=True) and gpu_list:
                with torch.amp.autocast("cuda"):
                    results = model(data, config, gpu_list, acc_result, "valid")
            else:
                results = model(data, config, gpu_list, acc_result, "valid")

            # 获取批次大小
            batch_size = len(data.get("label", next(iter(data.values()))))

            # 累加损失
            loss_val = results["loss"].item() if isinstance(results["loss"], torch.Tensor) else results["loss"]
            total_loss += loss_val * batch_size
            total_samples += batch_size

            # 保存评估指标
            acc_result = results["acc_result"]

            tb_cache["valid_loss"].append((step, loss_val))

            flush_cache(
                tb_cache,
                writer,
                acc_result,
                metrics,
                step_index=step,
            )

            # 主动释放内存
            del results
            del data
            torch.cuda.empty_cache()  # 每批次后清理缓存

            # 定期打印进度
            if (step + 1) % log_interval == 0 or step == total_batches - 1:
                progress = (step + 1) / total_batches * 100
                logger.info(f"验证进度: {step+1}/{total_batches} 批次 ({progress:.1f}%)")

    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")

    log_epoch(
        writer,
        epoch,
        avg_loss,
        acc_result,
        config,
        metrics,
    )

    # 打印验证结果
    output_value(
        epoch,
        "valid",
        "%d/%d" % (epoch + 1, config.getint("train", "epoch")),
        "",
        "%.3lf" % avg_loss,
        output_function(acc_result, config),
        None,
        config,
    )

    return {"loss": avg_loss, "acc_result": acc_result}


def flush_cache(
    tb_cache,
    writer,
    acc_result,
    metrics,
    step_index,
):

    for tag, records in tb_cache.items():
        for s, v in records:
            writer.add_scalar(f"eval/{tag}", v, s)
        tb_cache[tag].clear()

    # 写入降水评估的 step 级指标
    if acc_result and not acc_result.get("NULL", True):
        for metric, tb_tag in metrics.items():
            if metric in acc_result and acc_result[metric]:
                # 列表最后一个元素即当前 step 的值
                writer.add_scalar(f"eval/{tb_tag}", acc_result[metric][-1], step_index)

    writer.flush()


def log_epoch(writer, epoch, avg_loss, acc_result, config, metrics):
    """
    在一个 epoch 结束时，向 TensorBoard 写入：
      - 平均训练损失
      - 各参数组学习率
      - 梯度范数（若开启裁剪）
      - 各降水评估指标的平均值
    """
    # 1. 平均训练损失
    writer.add_scalar("eval/train_loss_avg", avg_loss, epoch)

    # 2. 降水评估指标平均值
    if acc_result and not acc_result.get("NULL", True):
        for metric, tag in metrics.items():
            vals = acc_result.get(metric, [])
            if vals:
                avg = sum(vals) / len(vals)
                writer.add_scalar(f"eval/{tag}", avg, epoch)
