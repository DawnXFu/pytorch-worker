import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.UASMLSTM.PreCorrect import PreCorrectDataset

logger = logging.getLogger(__name__)


class ConvLSTMPreCorrectDataset(PreCorrectDataset):
    """
    为ConvLSTM模型设计的降水校正数据集，支持时空分块。
    继承自PreCorrectDataset并增加空间分块功能。

    新增参数:
        patch_size: 空间块大小，可以是(height, width)的元组或单一整数
        patch_stride: 空间块滑动步长，默认等于patch_size
    """

    def __init__(self, config, mode, *args, **params):
        # 首先调用父类初始化
        super().__init__(config, mode, *args, **params)

        # 解析patch_size参数
        patch_size = params.get("patch_size", None)
        if patch_size is None:
            if hasattr(config, "get") and config.has_option("model", "patch_size"):
                patch_size = eval(config.get("model", "patch_size"))
            else:
                patch_size = (64, 64)  # 默认值

        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = patch_size

        # 解析patch_stride参数
        patch_stride = params.get("patch_stride", None)
        if patch_stride is None:
            if hasattr(config, "get") and config.has_option("model", "patch_stride"):
                patch_stride = eval(config.get("model", "patch_stride"))
            else:
                patch_stride = self.patch_size  # 默认等于patch_size

        if isinstance(patch_stride, int):
            self.patch_stride = (patch_stride, patch_stride)
        else:
            self.patch_stride = patch_stride

        # 获取网格形状，如果父类中未定义
        if not hasattr(self, "grid_shape"):
            # 加载第一个文件获取数据形状
            first_file_path = os.path.join(self.data_path, self.data_list[0])
            ds = self._get_file_handle(0)

            # 获取网格形状
            sample_var = self.feature_vars[0]
            if sample_var in ds:
                self.grid_shape = ds[sample_var].isel(time=0).shape
            else:
                # 尝试获取任何变量的形状
                for var in ds.data_vars:
                    if "time" in ds[var].dims:
                        self.grid_shape = ds[var].isel(time=0).shape
                        break

        height, width = self.grid_shape

        # 验证patch_size不超过网格尺寸
        if self.patch_size[0] > height or self.patch_size[1] > width:
            raise ValueError(f"patch_size {self.patch_size} 超过了网格尺寸 {(height, width)}")

        # 构建空间索引
        self._build_spatial_indices(height, width)

        print(
            f"ConvLSTM{mode}数据集: 网格大小 {(height, width)}, "
            f"patch_size {self.patch_size}, patch_stride {self.patch_stride}, "
            f"共 {len(self.sample_map) * len(self.spatial_indices)} 个样本"
        )

    def _build_spatial_indices(self, height, width):
        """构建空间索引列表"""
        self.spatial_indices = []

        for h in range(0, height - self.patch_size[0] + 1, self.patch_stride[0]):
            for w in range(0, width - self.patch_size[1] + 1, self.patch_stride[1]):
                self.spatial_indices.append((h, w))

    def __len__(self):
        """重写长度计算，考虑空间块"""
        return len(self.sample_map) * len(self.spatial_indices)

    def __getitem__(self, index):
        """获取样本，支持空间分块"""
        # 计算时间索引和空间索引
        time_idx = index // len(self.spatial_indices)
        spatial_idx = index % len(self.spatial_indices)

        start_h, start_w = self.spatial_indices[spatial_idx]
        end_h = start_h + self.patch_size[0]
        end_w = start_w + self.patch_size[1]

        # 获取时间序列起始索引
        start_global_idx = self.sample_map[time_idx]

        # 收集特征
        features = []
        read_plan = [(t, *self.global_time_to_file[start_global_idx + t]) for t in range(self.seq_len)]
        read_plan.sort(key=lambda x: x[1])  # 按文件索引排序以优化文件访问

        for var in self.feature_vars:
            var_seq = [None] * self.seq_len
            current_file_idx = None
            current_ds = None

            for t, file_idx, time_idx in read_plan:
                if current_file_idx != file_idx:
                    current_file_idx = file_idx
                    current_ds = self._get_file_handle(file_idx)

                # 加载数据并提取patch
                if var in current_ds:
                    var_data = current_ds[var].isel(time=time_idx).load().values
                    var_patch = var_data[start_h:end_h, start_w:end_w]
                else:
                    var_patch = np.zeros(self.patch_size, dtype=np.float32)

                var_seq[t] = var_patch

            # 按时间维度堆叠
            features.append(np.stack(var_seq, axis=0))

        # 堆叠所有特征变量并调整维度顺序
        # ConvLSTM通常期望的输入格式是 [seq_len, channels, height, width]
        # 而原始格式是 [channels, seq_len, height, width]
        features = np.stack(features, axis=0)  # [channels, seq_len, patch_h, patch_w]
        features = np.transpose(features, (1, 0, 2, 3))  # [seq_len, channels, patch_h, patch_w]

        # 获取标签（最后一个时间步）
        label_global_idx = start_global_idx + self.seq_len - 1
        file_idx, time_idx = self.global_time_to_file[label_global_idx]
        ds = self._get_file_handle(file_idx)

        if self.target_var in ds:
            label_data = ds[self.target_var].isel(time=time_idx).load().values
            label = label_data[start_h:end_h, start_w:end_w]
        else:
            label = np.zeros(self.patch_size, dtype=np.float32)

        # 返回与原代码相同的键名格式
        return {
            "data": torch.from_numpy(features).float(),  # [seq_len, channels, height, width]
            "label": torch.from_numpy(label).float(),  # [height, width]
        }
