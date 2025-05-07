import numpy as np
import torch


class PreCorrectFormatter:
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        input_data = []
        label_data = []

        for temp in data:
            input_data.append(temp["data"])
            if mode != "test":
                label_data.append(temp["label"])

        input_data = torch.from_numpy(np.array(input_data)).float()
        # 将input_data中的nan值替换为0
        input_data = torch.nan_to_num(input_data, nan=0.0)

        if mode != "test":
            label_data = torch.from_numpy(np.array(label_data)).float()
            # 将label_data中的nan值替换为0
            label_data = torch.nan_to_num(label_data, nan=0.0)

            return {"input": input_data, "label": label_data}
        else:
            return {"input": input_data}
