import json
import os

import torch
import torchvision.transforms as transforms

from data_formatter.Basic import BasicFormatter


class BasicResnetFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.config = config
        self.mode = mode
        self.normalization = config.getboolean("data", "normalization")

    def process(self, data, config, mode, *args, **params):
        pass
