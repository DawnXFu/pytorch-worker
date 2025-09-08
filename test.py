import argparse
import json
import logging
import os
import pickle
from datetime import datetime

import numpy as np
import torch

from config_parser import create_config
from tools.init_tool import init_all
from tools.test_tool import test

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config/UASMLSTM/UASMLSTM.config", help="specific config file")
    parser.add_argument("--gpu", "-g", help="gpu id list", default="0")
    parser.add_argument("--result", help="result file path", default="/mnt/d/Data/result/UGALSTM")
    args = parser.parse_args()

    configFilePath = args.config

    output_path = args.result

    # 创建结果保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_path, f"test_results_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    config = create_config(configFilePath)

    use_gpu = True
    gpu_list = []
    if args.gpu is None:
        use_gpu = False
    else:
        use_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        device_list = args.gpu.split(",")
        for a in range(0, len(device_list)):
            gpu_list.append(int(a))

    os.system("clear")

    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        gpu_list = False

    parameters = init_all(config, gpu_list, "test")

    results = test(parameters, config, gpu_list, output_path)

    # 保存完整结果到 pickle 文件
    pickle_path = os.path.join(save_dir, "full_results.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"完整结果已保存到: {pickle_path}")

    # 提取关键指标并保存到 JSON 文件
    summary = {
        "loss": float(results["loss"]),  # 确保数值可以被 JSON 序列化
        "predictions_shape": results["predictions"].shape if isinstance(results["predictions"], np.ndarray) else None,
        "labels_shape": results["labels"].shape if isinstance(results["labels"], np.ndarray) else None,
        "total_samples": len(results["timestamps"]) if "timestamps" in results else 0,
        "time_range": {
            "start": str(results["timestamps"][0]) if results.get("timestamps") else None,
            "end": str(results["timestamps"][-1]) if results.get("timestamps") else None,
        },
    }

    # 如果有评估指标，也添加到摘要中
    if "acc_result" in results and results["acc_result"]:
        summary["metrics"] = results["acc_result"]

    json_path = os.path.join(save_dir, "results_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    logger.info(f"结果摘要已保存到: {json_path}")
