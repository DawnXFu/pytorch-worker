import logging
import os
from collections import defaultdict
from timeit import default_timer as timer

import torch
from tools.eval_tool import gen_time_str, output_value
from torch.autograd import Variable

logger = logging.getLogger(__name__)


def test(parameters, config, gpu_list):
    model = parameters["model"]
    dataset = parameters["test_dataset"]
    writer = parameters["writer"]
    metrics = parameters["metrics"]
    output_function = parameters["output_function"]

    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    if os.path.exists(output_path):
        logger.warning("Output path exists, check whether need to change a name of model")
    os.makedirs(output_path, exist_ok=True)

    model.eval()

    total_loss = 0
    acc_result = None
    total_samples = 0
    total_outputs = []
    total_labels = []

    total_batches = len(dataset)
    log_interval = max(1, total_batches // 10)

    logger.info(f"开始测试：共{total_batches}个batch")

    tb_cache = defaultdict(list)

    with torch.no_grad():
        for step, batch in enumerate(dataset):
            torch.cuda.empty_cache()

            if gpu_list:
                data = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                data = batch

            if config.getboolean("train", "use_amp", fallback=True) and gpu_list:
                with torch.amp.autocast("cuda"):
                    results = model(data, config, gpu_list, acc_result, "valid")
            else:
                results = model(data, config, gpu_list, acc_result, "valid")

            batch_size = len(data.get("label", next(iter(data.values()))))
            loss_val = results["loss"].item() if isinstance(results["loss"], torch.Tensor) else results["loss"]
            total_loss += loss_val * batch_size
            total_samples += batch_size

            acc_result = results["acc_result"]

            # 保存 output 和 label
            total_outputs.append(results["output"].cpu())
            total_labels.append(results["label"].cpu())

            tb_cache["valid_loss"].append((step, loss_val))

            flush_cache(
                tb_cache,
                writer,
                acc_result,
                metrics,
                step_index=step,
            )

            torch.cuda.empty_cache()

            if (step + 1) % log_interval == 0 or step == total_batches - 1:
                progress = (step + 1) / total_batches * 100
                logger.info(f"测试进度: {step+1}/{total_batches} 批次 ({progress:.1f}%)")

    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
    # 1. 平均训练损失
    writer.add_scalar("eval/train_loss_avg", avg_loss, 0)

    # 2. 降水评估指标平均值
    if acc_result and not acc_result.get("NULL", True):
        for metric, tag in metrics.items():
            vals = acc_result.get(metric, [])
            if vals:
                avg = sum(vals) / len(vals)
                writer.add_scalar(f"eval/{tag}", avg, 0)

    output_value(
        0,
        "test",
        0,
        "",
        "%.3lf" % avg_loss,
        output_function(acc_result, config),
        None,
        config,
    )

    return {
        "loss": avg_loss,
        "acc_result": acc_result,
        "total_outputs": total_outputs,
        "total_labels": total_labels,
    }


def flush_cache(
    tb_cache,
    writer,
    acc_result,
    metrics,
    step_index,
):

    for tag, records in tb_cache.items():
        for s, v in records:
            writer.add_scalar(f"test/{tag}", v, s)
        tb_cache[tag].clear()

    # 写入降水评估的 step 级指标
    if acc_result and not acc_result.get("NULL", True):
        for metric, tb_tag in metrics.items():
            if metric in acc_result and acc_result[metric]:
                # 列表最后一个元素即当前 step 的值
                writer.add_scalar(f"test/{tb_tag}", acc_result[metric][-1], step_index)

    writer.flush()
