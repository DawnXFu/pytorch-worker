import logging
import os
import shutil
from collections import defaultdict
from timeit import default_timer as timer

import torch

# 添加混合精度训练所需的导入
from torch.amp import GradScaler, autocast  # 从torch.amp导入而不是torch.cuda.amp
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from tools.eval_tool import gen_time_str, output_value, valid
from tools.init_tool import init_formatter, init_test_dataset

logger = logging.getLogger(__name__)


def checkpoint(
    filename, model, optimizer, trained_epoch, config, global_step, scaler=None, scheduler=None, weights_only=False
):
    """保存检查点，支持仅保存权重或完整训练状态

    Args:
        filename: 保存文件路径
        model: 模型
        optimizer: 优化器
        trained_epoch: 当前训练轮次
        config: 配置对象
        global_step: 全局步数
        scaler: 混合精度训练的缩放器
        scheduler: 学习率调度器
        weights_only: 是否只保存权重(默认False，保存完整训练状态)
    """
    model_to_save = model.module if hasattr(model, "module") else model

    # 确定文件扩展名
    if weights_only and not filename.endswith(".pt"):
        filename = filename.replace(".pkl", ".pt")

    # 准备训练状态数据
    train_state = {
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step,
    }

    # 保存 scaler 和 scheduler 状态
    if scaler is not None:
        train_state["scaler"] = scaler.state_dict()
    if scheduler is not None:
        train_state["scheduler"] = scheduler.state_dict()

    try:
        if weights_only:
            # 只保存模型权重，训练状态保存到单独文件
            model_state = model_to_save.state_dict()

            # 生成两个文件路径：模型权重文件和训练状态文件
            weights_file = filename
            state_file = filename.replace(".pt", ".state")

            # 保存模型权重
            torch.save(model_state, weights_file)

            # 保存训练状态
            save_training_state = config.getboolean("train", "save_training_state", fallback=True)
            if save_training_state:
                torch.save(train_state, state_file)
                logger.info(f"✓ 模型权重已保存至 {weights_file}, 训练状态已保存至 {state_file}")
            else:
                logger.info(f"✓ 仅模型权重已保存至 {weights_file}")
        else:
            # 保存完整训练状态（包含模型权重）
            save_params = {"model": model_to_save.state_dict(), **train_state}  # 合并训练状态字典
            torch.save(save_params, filename)
            logger.info(f"✓ 完整训练状态已保存至 {filename}")
    except Exception as e:
        logger.warning(f"× 保存检查点失败: {str(e)}")


def load_checkpoint(checkpoint_path, model=None, device="cpu"):
    """加载检查点文件，支持加载完整检查点或仅模型权重

    Args:
        checkpoint_path: 检查点文件路径
        model: 可选，如果提供则加载权重到模型
        device: 加载权重的设备

    Returns:
        加载的检查点数据字典
    """
    if not os.path.exists(checkpoint_path):
        logger.warning(f"检查点文件不存在: {checkpoint_path}")
        return None

    logger.info(f"正在加载检查点: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 判断是否是权重文件还是完整检查点
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            # 完整检查点
            if model is not None:
                model_to_load = model.module if hasattr(model, "module") else model
                model_to_load.load_state_dict(checkpoint["model"])
                logger.info("✓ 已从完整检查点加载模型权重")
            return checkpoint
        else:
            # 仅权重文件
            if model is not None:
                model_to_load = model.module if hasattr(model, "module") else model
                model_to_load.load_state_dict(checkpoint)
                logger.info("✓ 已加载模型权重")

            # 尝试加载对应的训练状态文件
            state_path = checkpoint_path.replace(".pt", ".state")
            if os.path.exists(state_path):
                try:
                    train_state = torch.load(state_path, map_location=device)
                    logger.info(f"✓ 已加载训练状态: {state_path}")
                    return {"model": checkpoint, **train_state}
                except Exception as e:
                    logger.warning(f"无法加载训练状态文件: {e}")

            return {"model": checkpoint, "trained_epoch": -1, "global_step": 0}

    except Exception as e:
        logger.error(f"加载检查点失败: {str(e)}")
        return None


def get_lr_scheduler(optimizer, config, last_epoch=-1):
    """根据配置动态创建学习率调度器"""
    scheduler_type = config.get("train", "lr_scheduler", fallback="step")

    if scheduler_type == "step":
        step_size = config.getint("train", "step_size")
        gamma = config.getfloat("train", "lr_multiplier")
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)

    elif scheduler_type == "cosine":
        T_max = config.getint("train", "epoch")
        eta_min = config.getfloat("train", "min_lr", fallback=0)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)

    elif scheduler_type == "one_cycle":
        steps_per_epoch = config.getint("train", "steps_per_epoch", fallback=None)
        epochs = config.getint("train", "epoch")
        max_lr = config.getfloat("train", "max_lr")
        if steps_per_epoch is None:
            # 将在训练中重新初始化
            steps_per_epoch = 100
        return lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=steps_per_epoch * epochs, last_epoch=last_epoch
        )

    elif scheduler_type == "reduce_on_plateau":
        patience = config.getint("train", "patience", fallback=3)
        factor = config.getfloat("train", "factor", fallback=0.1)
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience, verbose=True)

    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using StepLR as default")
        return lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, last_epoch=last_epoch)


def train(parameters, config, gpu_list, do_test=False):
    epoch = config.getint("train", "epoch")
    batch_size = config.getint("train", "batch_size")

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")

    # 检查配置中是否启用了混合精度训练，默认启用
    use_amp = True
    if config.has_option("train", "use_amp"):
        use_amp = config.getboolean("train", "use_amp")

    # 添加梯度累积支持
    grad_accumulation_steps = config.getint("train", "grad_accumulation_steps", fallback=1)

    # tensorboard写入频率优化
    tb_log_interval = config.getint("output", "tb_log_interval", fallback=output_time)
    tb_histogram_interval = config.getint("output", "tb_histogram_interval", fallback=12)

    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    if os.path.exists(output_path):
        logger.warning("Output path exists, check whether need to change a name of model")
    os.makedirs(output_path, exist_ok=True)

    # 从检查点恢复训练 - 新增加载逻辑
    checkpoint_file = None
    if config.has_option("train", "resume_checkpoint"):
        checkpoint_file = config.get("train", "resume_checkpoint")
        # 判断是相对路径还是绝对路径
        if not os.path.isabs(checkpoint_file):
            checkpoint_file = os.path.join(output_path, checkpoint_file)

        if os.path.exists(checkpoint_file):
            logger.info(f"将从检查点恢复训练: {checkpoint_file}")
            device = f"cuda:{gpu_list[0]}" if len(gpu_list) > 0 else "cpu"
            checkpoint_data = load_checkpoint(checkpoint_file, parameters.get("model"), device)

            if checkpoint_data is not None:
                # 更新参数字典
                if "model" in checkpoint_data and parameters.get("model") is not None:
                    # model权重已在load_checkpoint中加载
                    logger.info("模型权重已加载")

                if "optimizer" in checkpoint_data and "optimizer_name" in checkpoint_data:
                    if "optimizer" in parameters:
                        try:
                            parameters["optimizer"].load_state_dict(checkpoint_data["optimizer"])
                            logger.info("已恢复优化器状态")
                        except Exception as e:
                            logger.warning(f"无法恢复优化器状态: {e}")

                # 更新训练轮次和全局步数
                if "trained_epoch" in checkpoint_data:
                    parameters["trained_epoch"] = checkpoint_data["trained_epoch"]
                    logger.info(f"恢复训练轮次: {parameters['trained_epoch']}")

                if "global_step" in checkpoint_data:
                    parameters["global_step"] = checkpoint_data["global_step"]
                    logger.info(f"恢复全局步数: {parameters['global_step']}")

                # 保存scaler状态以便后续使用
                if "scaler" in checkpoint_data:
                    parameters["scaler"] = checkpoint_data["scaler"]

                # 保存scheduler状态以便后续使用
                if "scheduler" in checkpoint_data:
                    parameters["scheduler"] = checkpoint_data["scheduler"]
        else:
            logger.warning(f"指定的检查点文件不存在: {checkpoint_file}")

    trained_epoch = parameters.get("trained_epoch", -1) + 1
    model = parameters["model"]
    optimizer = parameters["optimizer"]
    dataset = parameters["train_dataset"]
    global_step = parameters.get("global_step", 0)
    output_function = parameters["output_function"]

    # 初始化 GradScaler 用于混合精度训练
    scaler = None
    if use_amp and len(gpu_list) > 0:
        scaler = GradScaler()  # 简化初始化
        # 如果从检查点恢复，并且保存了scaler状态，则加载它
        if "scaler" in parameters:
            scaler.load_state_dict(parameters["scaler"])
            logger.info("已加载混合精度训练缩放器状态")
        logger.info("Using automatic mixed precision training")
    elif use_amp and len(gpu_list) == 0:
        logger.warning("Mixed precision training requires CUDA. Disabled.")
        use_amp = False

    # 确保优化器的参数组中有initial_lr参数
    if trained_epoch > 0:  # 只有在恢复训练时才需要
        base_lr = config.getfloat("train", "learning_rate")
        for param_group in optimizer.param_groups:
            if "initial_lr" not in param_group:
                param_group["initial_lr"] = param_group.get("lr", base_lr)

    # 初始化学习率调度器
    scheduler = get_lr_scheduler(optimizer, config, last_epoch=trained_epoch - 1 if trained_epoch > 0 else -1)
    if "scheduler" in parameters and parameters["scheduler"] is not None:
        try:
            scheduler.load_state_dict(parameters["scheduler"])
            logger.info("已恢复学习率调度器状态")
        except Exception as e:
            logger.warning(f"无法恢复学习率调度器状态: {e}")

    # 记录调度器类型
    scheduler_type = config.get("train", "lr_scheduler", fallback="step")
    logger.info(f"Using learning rate scheduler: {scheduler_type}")

    if do_test:
        init_formatter(config, ["test"])
        test_dataset = init_test_dataset(config)

    if trained_epoch == 0:
        shutil.rmtree(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")), True)

    os.makedirs(
        os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")), exist_ok=True
    )

    writer = SummaryWriter(
        os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
        config.get("output", "model_name"),
    )

    # 添加模型图可视化
    if config.has_option("output", "log_model_graph") and config.getboolean("output", "log_model_graph"):
        # 尝试记录模型架构，需要提供一个样例输入
        try:
            sample_batch = next(iter(dataset))
            for key in sample_batch.keys():
                if isinstance(sample_batch[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        sample_batch[key] = sample_batch[key].cuda()
            # 记录模型结构
            writer.add_graph(model, sample_batch)
            logger.info("Model graph added to TensorBoard")
        except Exception as e:
            logger.warning(f"Failed to add model graph to TensorBoard: {str(e)}")

    logger.info("Training start....")

    # 打印训练配置信息
    logger.info(f"Batch size: {batch_size}, Grad accumulation steps: {grad_accumulation_steps}")
    logger.info(f"Effective batch size: {batch_size * grad_accumulation_steps}")
    if use_amp:
        logger.info("Mixed precision training: Enabled")
    else:
        logger.info("Mixed precision training: Disabled")

    print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")

    total_len = len(dataset)
    more = ""
    if total_len < 10000:
        more = "\t"

    # 如果使用OneCycleLR并且需要知道总步数，在这里重新初始化
    if scheduler_type == "one_cycle" and hasattr(scheduler, "total_steps"):
        steps_per_epoch = total_len // grad_accumulation_steps
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.getfloat("train", "max_lr"),
            total_steps=steps_per_epoch * (epoch - trained_epoch),
            last_epoch=-1,
        )
        logger.info(f"OneCycleLR scheduler initialized with {steps_per_epoch} steps per epoch")

    # 创建TensorBoard指标缓存，减少写入次数
    tb_metrics = defaultdict(list)

    for epoch_num in range(trained_epoch, epoch):
        model.train()
        start_time = timer()
        current_epoch = epoch_num

        acc_result = None
        total_loss = 0
        running_loss = 0  # 用于梯度累积的损失累计

        output_info = ""
        step = -1

        # 重置梯度累积计数器
        accumulation_counter = 0

        for step, data in enumerate(dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            # 只在累积的第一步或不使用梯度累积时清零梯度
            if accumulation_counter == 0:
                optimizer.zero_grad()

            # 使用混合精度训练
            if use_amp and len(gpu_list) > 0:
                with autocast("cuda"):  # 简化写法
                    results = model(data, config, gpu_list, acc_result, "train")
                    loss, acc_result = results["loss"], results["acc_result"]

                    # 如果使用梯度累积，对损失进行归一化
                    if grad_accumulation_steps > 1:
                        loss = loss / grad_accumulation_steps

                    running_loss += float(loss)  # 累积的损失用于显示
                    total_loss += float(loss) * (1 if grad_accumulation_steps == 1 else grad_accumulation_steps)

                    # 使用 scaler 进行反向传播
                    scaler.scale(loss).backward()

                    # 只在完成累积步数后或最后一批数据时更新参数
                    accumulation_counter += 1
                    if accumulation_counter == grad_accumulation_steps or step == total_len - 1:
                        # 梯度裁剪（可选）
                        if config.has_option("train", "max_grad_norm"):
                            max_grad_norm = config.getfloat("train", "max_grad_norm")
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                        scaler.step(optimizer)
                        scaler.update()
                        accumulation_counter = 0
            else:
                # 原始训练流程
                results = model(data, config, gpu_list, acc_result, "train")
                loss, acc_result = results["loss"], results["acc_result"]

                # 如果使用梯度累积，对损失进行归一化
                if grad_accumulation_steps > 1:
                    loss = loss / grad_accumulation_steps

                running_loss += float(loss)  # 累积的损失用于显示
                total_loss += float(loss) * (1 if grad_accumulation_steps == 1 else grad_accumulation_steps)

                loss.backward()

                # 只在完成累积步数后或最后一批数据时更新参数
                accumulation_counter += 1
                if accumulation_counter == grad_accumulation_steps or step == total_len - 1:
                    # 梯度裁剪（可选）
                    if config.has_option("train", "max_grad_norm"):
                        max_grad_norm = config.getfloat("train", "max_grad_norm")
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()
                    accumulation_counter = 0

                    # 如果使用ReduceLROnPlateau调度器，在每次优化器步骤后更新
                    if scheduler_type == "reduce_on_plateau":
                        scheduler.step(running_loss)
                        running_loss = 0

            # 仅在完成梯度累积步骤或设定的输出间隔时显示信息
            if (accumulation_counter == 0 or step % output_time == 0) and step > 0:
                output_info = output_function(acc_result, config)

                delta_t = timer() - start_time

                output_value(
                    current_epoch,
                    "train",
                    "%d/%d" % (step + 1, total_len),
                    "%s/%s" % (gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                    "%.3lf"
                    % (total_loss / (step + 1) * (1 if grad_accumulation_steps == 1 else grad_accumulation_steps)),
                    output_info,
                    "\r",
                    config,
                )

            # 全局步数增加 - 只在实际更新参数时增加
            if accumulation_counter == 0:
                global_step += 1

                # 如果不是ReduceLROnPlateau，则在每次参数更新后调整学习率
                if scheduler_type != "reduce_on_plateau":
                    scheduler.step()

            # 降低TensorBoard写入频率，减少I/O开销
            if accumulation_counter == 0 and step % tb_log_interval == 0:
                # 批量写入TensorBoard，减少I/O调用
                tb_metrics["training/iteration_loss"].append((global_step, float(loss) * grad_accumulation_steps))

                # 每 N 步记录一次学习率
                for i, param_group in enumerate(optimizer.param_groups):
                    tb_metrics[f"training/learning_rate/group_{i}"].append((global_step, param_group["lr"]))

                # 混合精度训练相关指标
                if use_amp and scaler is not None:
                    tb_metrics["training/amp/scale"].append((global_step, scaler.get_scale()))

                # 每 N 步实际写入 TensorBoard
                if step % (tb_log_interval * 5) == 0:
                    for metric_name, values in tb_metrics.items():
                        for step_val, metric_val in values:
                            writer.add_scalar(metric_name, metric_val, step_val)
                    tb_metrics.clear()  # 清空缓存

                # 定期记录参数和梯度统计(降低频率)
                log_histograms = config.has_option("output", "log_histograms") and config.getboolean(
                    "output", "log_histograms"
                )
                if log_histograms and step % tb_histogram_interval == 0:
                    log_model_stats(writer, model, global_step, True)

                # 添加内存使用监控 (降低频率)
                if len(gpu_list) > 0 and torch.cuda.is_available() and step % (tb_log_interval * 5) == 0:
                    for i, gpu_id in enumerate(gpu_list):
                        writer.add_scalar(
                            f"system/gpu{gpu_id}/memory_allocated",
                            torch.cuda.memory_allocated(i) / (1024 * 1024),
                            global_step,
                        )
                        writer.add_scalar(
                            f"system/gpu{gpu_id}/memory_reserved",
                            torch.cuda.memory_reserved(i) / (1024 * 1024),
                            global_step,
                        )

                        # 添加内存分析功能 (如果支持)
                        if hasattr(torch.cuda, "memory_stats"):
                            stats = torch.cuda.memory_stats(i)
                            writer.add_scalar(
                                f"system/gpu{gpu_id}/active_blocks",
                                stats.get("active_blocks.all.current", 0),
                                global_step,
                            )

                        writer.add_scalar("epoch/train_loss", float(total_loss) / (step + 1), current_epoch)

                        # 添加训练评价指标到TensorBoard
                        if acc_result is not None and acc_result.get("count", 0) > 0:
                            # 计算平均值并写入TensorBoard
                            for metric_name in ["CC", "RMSE", "KGE", "PBIAS", "TS"]:
                                if metric_name in acc_result:
                                    avg_value = acc_result[metric_name] / acc_result["count"]
                                    writer.add_scalar(f"metrics/train/{metric_name}", avg_value, current_epoch)

        # 写入所有剩余的累积指标
        for metric_name, values in tb_metrics.items():
            for step_val, metric_val in values:
                writer.add_scalar(metric_name, metric_val, step_val)
        tb_metrics.clear()

        output_info = output_function(acc_result, config)
        delta_t = timer() - start_time
        output_value(
            current_epoch,
            "train",
            "%d/%d" % (step + 1, total_len),
            "%s/%s" % (gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
            "%.3lf" % (total_loss / (step + 1)),
            output_info,
            None,
            config,
        )

        if step == -1:
            logger.error("There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError

        writer.add_scalar("epoch/train_loss", float(total_loss) / (step + 1), current_epoch)

        # 记录训练时间
        writer.add_scalar("epoch/train_time_seconds", delta_t, current_epoch)

        # 记录样本处理速度
        samples_per_second = total_len / delta_t
        writer.add_scalar("performance/samples_per_second", samples_per_second, current_epoch)

        # 记录当前学习率
        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f"epoch/learning_rate/group_{i}", param_group["lr"], current_epoch)

        # 保存更完整的检查点，包括调度器状态
        # 根据配置决定是否只保存权重
        weights_only_save = config.getboolean("train", "weights_only_save", fallback=False)
        checkpoint(
            os.path.join(output_path, "%d.pkl" % current_epoch),
            model,
            optimizer,
            current_epoch,
            config,
            global_step,
            scaler if use_amp else None,
            scheduler,
            weights_only=weights_only_save,
        )

        writer.add_scalar(
            config.get("output", "model_name") + "_train_epoch", float(total_loss) / (step + 1), current_epoch
        )

        # 如果是ReduceLROnPlateau调度器，在每个epoch结束时以验证损失调整学习率
        if current_epoch % test_time == 0:
            with torch.no_grad():
                # 添加验证/测试计时
                valid_start_time = timer()
                valid_results = valid(
                    model, parameters["valid_dataset"], current_epoch, writer, config, gpu_list, output_function
                )
                valid_time = timer() - valid_start_time

                # 如果使用ReduceLROnPlateau并且有验证损失，用验证损失更新调度器
                if scheduler_type == "reduce_on_plateau" and "loss" in valid_results:
                    scheduler.step(valid_results["loss"])

                # 记录验证时间
                writer.add_scalar("epoch/valid_time_seconds", valid_time, current_epoch)

                if do_test:
                    test_start_time = timer()
                    test_results = valid(
                        model, test_dataset, current_epoch, writer, config, gpu_list, output_function, mode="test"
                    )
                    test_time_taken = timer() - test_start_time
                    writer.add_scalar("epoch/test_time_seconds", test_time_taken, current_epoch)

                # 确保所有事件都被写入
                writer.flush()

    writer.close()
    logger.info(
        f"Training completed. TensorBoard logs saved to {os.path.join(config.get('output', 'tensorboard_path'), config.get('output', 'model_name'))}"
    )

    return model, optimizer, global_step


# 创建学习率记录函数
def log_learning_rate(writer, optimizer, global_step):
    for i, param_group in enumerate(optimizer.param_groups):
        writer.add_scalar(f"training/learning_rate/group_{i}", param_group["lr"], global_step)


# 创建参数和梯度统计函数
def log_model_stats(writer, model, global_step, log_histograms=False):
    # 记录参数统计
    norm_dict = {}  # 收集所有范数以减少TensorBoard调用

    for name, param in model.named_parameters():
        if param.requires_grad:
            # 收集参数范数
            norm_dict[f"parameters/norm/{name}"] = param.norm().item()

            # 收集梯度范数(如果存在)
            if param.grad is not None:
                norm_dict[f"gradients/norm/{name}"] = param.grad.norm().item()

            # 如果配置了直方图记录
            if log_histograms:  # 降低频率以减少存储需求
                writer.add_histogram(f"parameters/histogram/{name}", param.data.cpu(), global_step)
                if param.grad is not None:
                    writer.add_histogram(f"gradients/histogram/{name}", param.grad.cpu(), global_step)

    # 批量写入所有范数值
    for name, value in norm_dict.items():
        writer.add_scalar(name, value, global_step)
