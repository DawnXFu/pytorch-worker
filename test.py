import argparse
import json
import logging
import os

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
    parser.add_argument("--gpu", "-g", help="gpu id list")
    parser.add_argument("--result", help="result file path", default="result.json")
    args = parser.parse_args()

    configFilePath = args.config

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

    config = create_config(configFilePath)

    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError

    parameters = init_all(config, gpu_list, "test")

    json.dump(
        test(parameters, config, gpu_list),
        open(args.result, "w", encoding="utf8"),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
