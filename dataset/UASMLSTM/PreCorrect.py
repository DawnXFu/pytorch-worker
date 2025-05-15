import os

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class PreCorrectDataset(Dataset):
    def __init__(self, config, mode, *args, **params):
        """
        初始化数据集

        参数:
        config: 配置对象，包含数据路径和模型参数
        mode: 数据集模式，如'train', 'valid', 'test'
        """
        self.config = config
        self.mode = mode

        # 从配置中读取参数
        self.data_path = config.get("data", f"{mode}_data_path")
        self.seq_len = config.getint("model", "seq_length")

        # 特征变量和目标变量
        self.feature_vars = params.get("feature_vars", ["TAIR", "UWIN", "VWIN", "PRE"])
        self.target_var = params.get("target_var", "corrected_precip")

        # 是否缓存数据以加速训练
        self.cache_data = params.get("cache_data", False)

        # 滑动窗口参数
        self.stride = params.get("stride", 3)  # 如果配置中没有，使用默认值1
        if hasattr(config, "getint") and config.has_option("model", "stride"):
            self.stride = config.getint("model", "stride")

        # 获取目录中所有NC文件
        self.data_list = sorted([f for f in os.listdir(self.data_path) if f.endswith(".nc")])

        if len(self.data_list) == 0:
            raise ValueError(f"在 {self.data_path} 目录下未找到NC文件")

        # 计算每个文件中有多少个有效样本
        self.samples_per_file = []
        self.sample_map = []  # 索引到(文件索引,文件内起始位置)的映射

        for file_idx, filename in enumerate(self.data_list):
            # 获取文件中的时间点数
            file_path = os.path.join(self.data_path, filename)
            try:
                with xr.open_dataset(file_path) as ds:
                    time_points = len(ds.time)

                    # 计算可能的样本数: (总时间点 - 序列长度) / 步长
                    # 这里假设我们预测的是序列末尾的下一个时间点
                    valid_samples = max(0, (time_points - self.seq_len) // self.stride)
                    self.samples_per_file.append(valid_samples)

                    # 创建索引映射
                    for i in range(valid_samples):
                        self.sample_map.append((file_idx, i * self.stride))
            except Exception as e:
                print(f"警告: 无法读取文件 {file_path}: {e}")

        # 数据缓存
        self.data_cache = {} if self.cache_data else None

        print(f"{mode}数据集: 共加载了 {len(self.data_list)} 个文件, 总样本数: {len(self.sample_map)}")

    def __getitem__(self, index):
        """获取指定索引的样本"""
        file_idx, start_idx = self.sample_map[index]
        filename = self.data_list[file_idx]
        file_path = os.path.join(self.data_path, filename)

        # 用 with 语句打开并自动关闭
        if not (self.cache_data and file_path in self.data_cache):
            with xr.open_dataset(file_path) as ds:
                ds_data = {}
                for var in self.feature_vars + [self.target_var]:
                    ds_data[var] = ds[var].values if var in ds else np.zeros_like(ds[self.feature_vars[0]].values)
            if self.cache_data:
                self.data_cache[file_path] = ds_data
        else:
            ds_data = self.data_cache[file_path]

        # 提取输入序列
        end_idx = start_idx + self.seq_len
        features = []

        for var in self.feature_vars:
            # 选择时间窗口对应的数据
            feature = ds_data[var][start_idx:end_idx]
            features.append(feature)

        # 堆叠特征形成输入张量 (feature_count, seq_len, lat, lon)
        features = np.stack(features, axis=0)

        # 获取标签 - 使用序列末尾之后的时间点
        label = ds_data[self.target_var][end_idx]  # (lat, lon)

        # 交换维度顺序，结果形状为(seq_len, feature_count, lat, lon)
        features = np.transpose(features, (1, 0, 2, 3))

        features = torch.from_numpy(features).float()
        label = torch.from_numpy(label).float()
        # 返回符合原有格式的字典
        return {"data": features, "label": label}

    def __len__(self):
        """返回数据集中的样本总数"""
        return len(self.sample_map)
