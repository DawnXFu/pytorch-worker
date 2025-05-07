import os

import numpy as np
import xarray
from torch.utils.data import Dataset


class PreCorrectDataset(Dataset):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.data_list = os.listdir(self.data_path)
        self.seq_len = config.getint("model", "seq_length")
        self.item_len = int(24 / self.seq_len)  # 每个文件中时间步的数量

    def __getitem__(self, index):
        data_item = index // self.item_len  # 要使用的文件索引
        item_index = index % self.item_len  # 文件中的时间步索引
        data_path = os.path.join(self.data_path, self.data_list[data_item])

        with xarray.open_dataset(data_path) as ds:
            # 提取所有变量的数据为numpy数组
            tair = ds["TAIR"].values  # 形状为(time, lat, lon)
            uwin = ds["UWIN"].values  # 形状为(time, lat, lon)
            vwin = ds["VWIN"].values  # 形状为(time, lat, lon)
            pre = ds["PRE"].values  # 形状为(time, lat, lon)

            # 将TAIR、UWIN、VWIN、PRE组合为特征数据
            # 在通道维度上堆叠，结果形状为(4, time, lat, lon)
            features = np.stack([tair, uwin, vwin, pre], axis=0)

            # PRE作为标签
            label = ds["STATION"].values  # 形状为(time, lat, lon)
            # 选择前seq_len个时间步的数据
            features = features[:, item_index * self.seq_len : (item_index + 1) * self.seq_len, :, :]
            label = label[(item_index + 1) * self.seq_len - 1, :, :]
            # 交换维度顺序，结果形状为(seq_len, 4, lat, lon)
            features = np.transpose(features, (1, 0, 2, 3))

        return {"data": features, "label": label}

    def __len__(self):
        length = len(self.data_list) * self.item_len
        return length
