import logging
import os
import shutil
from timeit import default_timer as timer

import torch
from torch.amp import GradScaler
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from model import get_model
from model.optimizer import init_optimizer
from reader.reader import init_dataset, init_test_dataset

from .output_init import init_output_function

logger = logging.getLogger(__name__)


def init_all(config, gpu_list, mode, *args, **params):
    """
    合并 init_training 和 init_all：
      - mode="train" 返回 model, optimizer, scheduler, scaler, writer, start_epoch, global_step, train_dataset, valid_dataset
      - mode="test"  返回 model, test_dataset
    """
    result = {}

    # —— 1. 数据集初始化 ——
    if mode == "train":
        result["train_dataset"], result["valid_dataset"] = init_dataset(config, **params)
    else:
        result["test_dataset"] = init_test_dataset(config, **params)

    # —— 2. 模型 & 设备 ——
    model = get_model(config.get("model", "model_name"))(config, gpu_list, **params)
    device = f"cuda:{gpu_list[0]}" if gpu_list else "cpu"
    model.to(device)
    if len(gpu_list) > 1:
        try:
            model.init_multi_gpu(gpu_list, config, **params)
        except Exception:
            logger.warning("No init_multi_gpu implemented, use single GPU.")

    result["model"] = model
    # 输出函数
    result["output_function"] = init_output_function(config)
    result["metrics"] = {
        "CC": "metrics/correlation_coefficient",
        "RMSE": "metrics/root_mean_square_error",
        "KGE": "metrics/kling_gupta_efficiency",
        "POD_moderate_rain": "metrics/probability_of_detection_moderate",
        "FAR_moderate_rain": "metrics/false_alarm_ratio_moderate",
        "TS_heavy_rain": "metrics/threat_score_heavy",
    }

    if mode == "test":
        # 使用checkpoint参数或检查配置文件
        if config.has_option("test", "resume_checkpoint"):
            ckpt_path = config.get("test", "resume_checkpoint")

        if ckpt_path:
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.join(
                    config.get("output", "model_path"), config.get("output", "model_name"), ckpt_path
                )
            if os.path.exists(ckpt_path):
                logger.info(f"Loading checkpoint for test from {ckpt_path}")
                # 仅加载模型权重
                load_checkpoint(
                    ckpt_path, model=model, map_location=device, optimizer=None, scheduler=None, scaler=None
                )
            else:
                logger.warning(f"Test checkpoint not found: {ckpt_path}")
        return result
    # —— 3. 训练模式下的额外初始化 ——
    elif mode == "train":
        # optimizer
        optimizer = init_optimizer(model, config, **params)
        result["optimizer"] = optimizer

        # scaler（AMP）
        use_amp = config.getboolean("train", "use_amp", fallback=True) and gpu_list
        if use_amp:
            scaler = GradScaler("cpu" if not gpu_list else "cuda")
            result["scaler"] = scaler
        else:
            scaler = None
            result["scaler"] = None

        scheduler = get_lr_scheduler(optimizer, config)

        result["scheduler"] = scheduler
        # 从 checkpoint 恢复
        result["start_epoch"] = 0
        result["global_step"] = 0

        # 使用checkpoint参数或检查配置文件
        if config.has_option("train", "resume_checkpoint"):
            ckpt_path = config.get("train", "resume_checkpoint")
            if os.path.exists(ckpt_path):
                logger.info(f"Loading checkpoint from {ckpt_path}")
                ckpt_data = load_checkpoint(
                    ckpt_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, map_location=device
                )

                result["start_epoch"] = ckpt_data["epoch"]
                result["global_step"] = ckpt_data["global_step"]
                if not ckpt_data["scheduler"]:
                    # 如果没有加载调度器状态，则重新初始化
                    scheduler = get_lr_scheduler(optimizer, config, last_epoch=ckpt_data["epoch"])
                    # OneCycleLR 重新设置 total_steps
                    if isinstance(scheduler, lr_scheduler.OneCycleLR):
                        steps = len(result["train_dataset"]) // config.getint("train", "grad_accumulation_steps")
                        scheduler.total_steps = steps * (config.getint("train", "epoch") - result["start_epoch"])
                        logger.info(f"OneCycleLR total_steps set to {scheduler.total_steps}")
                    result["scheduler"] = scheduler

                logger.info(f"Resumed at epoch={result['start_epoch']}, global_step={result['global_step']}")
            else:
                logger.warning(f"Checkpoint not found: {ckpt_path}, start from scratch.")

        # TensorBoard
        tb_path = os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name"))
        if result["start_epoch"] == 0 and os.path.isdir(tb_path):
            shutil.rmtree(tb_path)
        os.makedirs(tb_path, exist_ok=True)
        writer = SummaryWriter(tb_path, config.get("output", "model_name"))
        logger.info(f"TensorBoard logs → {tb_path}")
        result["writer"] = writer

    return result


def checkpoint(filename, model, optimizer, epoch, global_step, scheduler=None, scaler=None, config=None):
    """将所有信息保存到同一个 .pth 文件里"""
    model_to_save = model.module if hasattr(model, "module") else model
    state = {
        "model": model_to_save.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    # 根据配置决定是否保存额外信息
    if config and config.getboolean("train", "save_optimizer", fallback=True):
        state["optimizer"] = optimizer.state_dict()
    if scheduler and config and config.getboolean("train", "save_scheduler", fallback=False):
        state["scheduler"] = scheduler.state_dict()
    if scaler and config and config.getboolean("train", "save_scaler", fallback=False):
        state["scaler"] = scaler.state_dict()

    torch.save(state, filename)
    logger.info(f"✔ 检查点保存到: {filename}")


def load_checkpoint(path, model=None, optimizer=None, scheduler=None, scaler=None, map_location="cpu", strict=False):
    """加载一个 .pth 文件，按需恢复各个组件"""

    reslut = {
        "epoch": 0,
        "global_step": 0,
        "model": False,
        "scaler": False,
        "scheduler": False,
        "optimizer": False,
    }

    checkpoint = torch.load(path, map_location=map_location)
    if model is not None and "model" in checkpoint:
        mdl = model.module if hasattr(model, "module") else model
        mdl.load_state_dict(checkpoint["model"], strict=strict)
        reslut["model"] = True
        logger.info("✔ 模型权重已加载")

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        reslut["optimizer"] = True
        # 显示优化器类型
        optimizer_type = optimizer.__class__.__name__
        optimizer_params = {}
        # 获取一些常用的优化器参数
        if hasattr(optimizer, "defaults"):
            for key in ["lr", "weight_decay", "betas", "momentum"]:
                if key in optimizer.defaults:
                    optimizer_params[key] = optimizer.defaults[key]

        params_str = ", ".join([f"{k}={v}" for k, v in optimizer_params.items()])
        logger.info(f"✔ 优化器状态已加载: {optimizer_type}({params_str})")

    if scheduler is not None and "scheduler" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
            reslut["scheduler"] = True
            # 显示调度器类型和关键参数
            scheduler_type = scheduler.__class__.__name__
            scheduler_info = ""

            # 获取不同类型调度器的关键参数
            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler_info = f"step_size={scheduler.step_size}, gamma={scheduler.gamma}"
            elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_info = f"mode={scheduler.mode}, factor={scheduler.factor}, patience={scheduler.patience}"
            elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                scheduler_info = f"T_max={scheduler.T_max}, eta_min={scheduler.eta_min}"
            elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler_info = f"max_lr={scheduler.max_lr}, total_steps={scheduler.total_steps}"
            elif isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR):
                scheduler_info = f"gamma={scheduler.gamma}"

            logger.info(f"✔ 调度器状态已加载: {scheduler_type}({scheduler_info})")
        except Exception as e:
            logger.warning(f"无法加载调度器状态: {e}")
            reslut["scheduler"] = False

    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
        reslut["scaler"] = True
        logger.info("✔ 缩放器状态已加载")

    reslut["epoch"] = checkpoint.get("epoch", 0) + 1
    reslut["global_step"] = checkpoint.get("global_step", 0)
    return reslut


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

        # 读取额外的OneCycleLR参数
        pct_start = config.getfloat("train", "pct_start", fallback=0.3)
        div_factor = config.getfloat("train", "div_factor", fallback=25.0)
        final_div_factor = config.getfloat("train", "final_div_factor", fallback=1.0e4)

        # 三阶段配置
        three_phase = config.getboolean("train", "three_phase", fallback=False)

        if steps_per_epoch is None:
            # 将在训练中重新初始化
            steps_per_epoch = 100
            logger.info(f"未配置steps_per_epoch, OneCycleLR将在训练中重新初始化")

        total_steps = steps_per_epoch * epochs

        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch,
        )

    elif scheduler_type == "reduce_on_plateau":
        patience = config.getint("train", "patience", fallback=3)
        factor = config.getfloat("train", "factor", fallback=0.1)
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience, verbose=True)

    elif scheduler_type == "exponential":
        gamma = config.getfloat("train", "gamma", fallback=0.9)
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using StepLR as default")
        return lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, last_epoch=last_epoch)
