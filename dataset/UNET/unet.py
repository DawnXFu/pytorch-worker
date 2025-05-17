import os

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class UNetDataset(Dataset):
    def __init__(self, config, mode, *args, **params):
        """
        初始化数据集 - 简化版，只需单帧数据

        参数:
        config: 配置对象，包含数据路径和模型参数
        mode: 数据集模式，如'train', 'valid', 'test'
        """
        self.config = config
        self.mode = mode

        # 从配置中读取参数
        self.data_path = config.get("data", f"{mode}_data_path")

        # 特征变量和目标变量
        self.feature_vars = params.get("feature_vars", ["TAIR", "UWIN", "VWIN", "PRE"])
        self.target_var = params.get("target_var", "corrected_precip")

        # 是否缓存数据以加速训练
        self.cache_data = params.get("cache_data", False)

        # 获取目录中所有NC文件
        self.data_list = sorted([f for f in os.listdir(self.data_path) if f.endswith(".nc")])

        if len(self.data_list) == 0:
            raise ValueError(f"在 {self.data_path} 目录下未找到NC文件")

        # 计算每个文件中有多少个有效样本
        self.sample_map = []  # 索引到(文件索引,时间索引)的映射

        for file_idx, filename in enumerate(self.data_list):
            # 获取文件中的时间点数
            file_path = os.path.join(self.data_path, filename)
            try:
                with xr.open_dataset(file_path) as ds:
                    time_points = len(ds.time)

                    # 每个时间点作为一个独立样本
                    for time_idx in range(time_points):
                        self.sample_map.append((file_idx, time_idx))
            except Exception as e:
                print(f"警告: 无法读取文件 {file_path}: {e}")

        # 数据缓存
        self.data_cache = {} if self.cache_data else None

        print(f"{mode}数据集: 共加载了 {len(self.data_list)} 个文件, 总样本数: {len(self.sample_map)}")

    def __getitem__(self, index):
        file_idx, time_idx = self.sample_map[index]
        filename = self.data_list[file_idx]
        file_path = os.path.join(self.data_path, filename)

        if not (self.cache_data and file_path in self.data_cache):
            with xr.open_dataset(file_path) as ds:
                ds_data = {}
                for var in self.feature_vars + [self.target_var]:
                    ds_data[var] = ds[var].values if var in ds else np.zeros_like(ds[self.feature_vars[0]].values)
            if self.cache_data:
                self.data_cache[file_path] = ds_data
        else:
            ds_data = self.data_cache[file_path]

        # 构建单张图像输入 (多通道)
        features = []
        for var in self.feature_vars:
            features.append(ds_data[var][time_idx])
        features = np.stack(features, axis=0)  # (C, H, W)

        # 提取对应的标签
        label = ds_data[self.target_var][time_idx]  # (H, W)

        # 转为 Tensor
        features = torch.from_numpy(features).float()
        label = torch.from_numpy(label).float()

        return {"data": features, "label": label}

    def __len__(self):
        return len(self.sample_map)
