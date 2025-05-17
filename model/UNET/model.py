import torch
from einops import rearrange
from torch import nn

from tools.accuracy_init import init_accuracy_function

from .DSW import DepthwiseSeparableConv


class UNetModel(nn.Module):
    def __init__(self, config, gpu_list, *args, **kwargs):
        super().__init__()
        # 读取配置
        in_ch = config.getint("model", "in_channels")
        out_ch = config.getint("model", "out_channels")
        base_ch = config.getint("model", "base_channels")  # 例如 64
        self.accuracy_function = init_accuracy_function(config, *args, **kwargs)

        # Encoder
        def conv_block(in_c, out_c):
            return nn.Sequential(
                DepthwiseSeparableConv(in_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv(out_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_ch, base_ch)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = conv_block(base_ch, base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.enc4 = conv_block(base_ch * 4, base_ch * 8)

        # Bottleneck
        self.bottleneck = conv_block(base_ch * 8, base_ch * 16)

        # Decoder
        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                DepthwiseSeparableConv(in_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.up4 = up_block(base_ch * 16, base_ch * 8)
        self.dec4 = conv_block(base_ch * 16, base_ch * 8)
        self.up3 = up_block(base_ch * 8, base_ch * 4)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)
        self.up2 = up_block(base_ch * 4, base_ch * 2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)
        self.up1 = up_block(base_ch * 2, base_ch)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        # 最终输出
        self.conv_last = nn.Conv2d(base_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, data, config, gpu_list, acc_result, mode):
        # 取序列最后一帧做输入
        x = data["input"]  # [B, C, H, W]

        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # 解码 + skip
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.conv_last(d1).squeeze(1)  # [B, H, W]

        # 计算 loss/metric
        if "label" in data:
            label = data["label"]
            loss = self.criterion(out, label)
            acc_result = self.accuracy_function(out, label, config, acc_result)
            if mode == "test":
                return {"output": acc_result}
            return {"loss": loss, "acc_result": acc_result}
        else:
            return {"output": out}

    def criterion(self, x, label):
        # 修改损失函数
        # 引入权重调整不同降水等级的损失贡献
        mask_light = (label > 0.1) & (label <= 5.0)
        mask_moderate = (label > 5.0) & (label <= 15.0)
        mask_heavy = label > 15.0

        # 基础损失计算
        loss_L1 = nn.L1Loss(reduction="none")(x, label)
        loss_MSE = nn.MSELoss(reduction="none")(x, label)

        # 对不同降水等级使用不同权重
        weighted_loss = loss_L1 * (1.0 + 5.0 * mask_light + 3.0 * mask_moderate + 2.0 * mask_heavy) + loss_MSE * (
            1.0 + 3.0 * mask_light + 2.0 * mask_moderate + 1.0 * mask_heavy
        )

        # 可以考虑移除BCE损失，因为降水是连续值
        return weighted_loss.mean()
