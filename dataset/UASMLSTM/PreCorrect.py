import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


class PreCorrectDataset(Dataset):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", f"{mode}_data_path")
        self.seq_len = config.getint("model", "seq_length")
        self.feature_vars = params.get("feature_vars", ["TAIR", "UWIN", "VWIN", "PRE"])
        self.target_var = params.get("target_var", "corrected_precip")
        self.stride = params.get("stride", 3)
        if hasattr(config, "getint") and config.has_option("model", "stride"):
            self.stride = config.getint("model", "stride")
        self.max_open_files = params.get("max_open_files", 5)
        self.file_handles = OrderedDict()

        # 获取所有文件
        self.data_list = sorted([f for f in os.listdir(self.data_path) if f.endswith(".nc")])
        if not self.data_list:
            raise ValueError(f"在 {self.data_path} 目录下未找到NC文件")

        # 构建时间索引
        self._build_global_time_index()

        print(f"{mode}数据集: 共加载了 {len(self.data_list)} 个文件, 总样本数: {len(self.sample_map)}")

    def _build_global_time_index(self):
        """构建全局时间索引"""
        self.file_time_map = []
        all_times = []
        for file_idx, filename in enumerate(self.data_list):
            file_path = os.path.join(self.data_path, filename)
            with xr.open_dataset(file_path, engine="netcdf4", chunks={}, decode_times=True) as ds:
                times = pd.to_datetime(ds.time.values)
                for time_idx, timestamp in enumerate(times):
                    self.file_time_map.append((file_idx, time_idx, timestamp))
                    all_times.append(timestamp)
        self.file_time_map.sort(key=lambda x: x[2])
        all_times.sort()
        self.sample_map = [i for i in range(0, len(all_times) - self.seq_len + 1, self.stride)]
        self.global_time_to_file = {
            i: (file_idx, time_idx) for i, (file_idx, time_idx, _) in enumerate(self.file_time_map)
        }

    def _get_file_handle(self, file_idx):
        """懒加载文件句柄"""
        if file_idx in self.file_handles:
            self.file_handles.move_to_end(file_idx)
            return self.file_handles[file_idx]
        if len(self.file_handles) >= self.max_open_files:
            _, oldest_handle = self.file_handles.popitem(last=False)
            oldest_handle.close()
        file_path = os.path.join(self.data_path, self.data_list[file_idx])
        ds = xr.open_dataset(file_path, engine="netcdf4", chunks="auto")
        self.file_handles[file_idx] = ds
        return ds

    def __getitem__(self, index):
        """获取样本"""
        start_global_idx = self.sample_map[index]
        features = []
        read_plan = [(t, *self.global_time_to_file[start_global_idx + t]) for t in range(self.seq_len)]
        read_plan.sort(key=lambda x: x[1])
        for var in self.feature_vars:
            var_seq = [None] * self.seq_len
            current_file_idx = None
            current_ds = None
            for t, file_idx, time_idx in read_plan:
                if current_file_idx != file_idx:
                    current_file_idx = file_idx
                    current_ds = self._get_file_handle(file_idx)
                var_seq[t] = (
                    current_ds[var].isel(time=time_idx).load().values
                    if var in current_ds
                    else np.zeros(self.grid_shape, dtype=np.float32)
                )
            features.append(np.stack(var_seq, axis=0))
        features = np.transpose(np.stack(features, axis=0), (1, 0, 2, 3))
        label_global_idx = start_global_idx + self.seq_len - 1
        file_idx, time_idx = self.global_time_to_file[label_global_idx]
        ds = self._get_file_handle(file_idx)
        label = (
            ds[self.target_var].isel(time=time_idx).load().values
            if self.target_var in ds
            else np.zeros(self.grid_shape, dtype=np.float32)
        )
        return {"data": torch.from_numpy(features).float(), "label": torch.from_numpy(label).float()}

    def __len__(self):
        return len(self.sample_map)

    def __del__(self):
        for handle in self.file_handles.values():
            handle.close()
