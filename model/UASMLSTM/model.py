import torch
from einops import rearrange
from torch import nn

from tools.accuracy_init import init_accuracy_function

from .CBAM import CBAMBlock
from .DSW import DepthwiseSeparableConv
from .EnhancedRNN import EnhancedRNN
from .GradientHighwayUnit import GHU

# 在导入部分添加
from .ResidualBlock import ResidualBlock
from .SpatioTemporalLSTMCell import SMLSTMCell

# 修改主模型部分


class EncoderBottleneck(nn.Module):
    """
    下采样模块
    输入形状: (batch, in_channel, height, width)
    """

    def __init__(self, in_channel, out_channel, stride=1, base_channel=64):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channel)
        )
        width = int(out_channel * (base_channel / 64))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, width, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x = self.relu(x_1 + x_2)
        return x


class DecoderBottleneck(nn.Module):
    """
    上采样模块
    输入形状: (batch, in_channel, height, width)
    """

    def __init__(self, in_channel, out_channel, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        self.layer = nn.Sequential(
            DepthwiseSeparableConv(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x, x_concat], dim=1)  # x (batch, channel, height, width)

        x = self.layer(x)
        return x


# 修改UASMLSTM类的初始化部分
class UASMLSTM(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super().__init__()

        img_width = config.getint("model", "img_width")
        img_height = config.getint("model", "img_height")
        in_channels = config.getint("model", "in_channels")
        out_channels = config.getint("model", "out_channels")
        num_layers = config.getint("model", "num_layers")
        num_hiddens_str = config.get("model", "num_hiddens")
        num_hiddens = [int(x) for x in num_hiddens_str.split(",")]
        seq_length = config.getint("model", "seq_length")
        layer_norm = config.getboolean("model", "layer_norm")

        self.accuracy_function = init_accuracy_function(config, *args, **params)

        # 初始卷积层 - 使用DSW提高效率
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 使用残差连接的编码器
        self.encoder1 = nn.Sequential(
            ResidualBlock(out_channels, out_channels), EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        )

        self.encoder2 = nn.Sequential(
            ResidualBlock(out_channels * 2, out_channels * 2),
            EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2),
        )

        self.encoder3 = nn.Sequential(
            ResidualBlock(out_channels * 4, out_channels * 4),
            EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2),
        )

        # 使用增强的RNN模块
        self.rnn = EnhancedRNN(
            img_channel=out_channels * 8,
            img_width=img_width // 16,
            img_height=img_height // 16,
            num_layers=num_layers,
            num_hidden=num_hiddens,
            seq_length=seq_length,
            layer_norm=layer_norm,
        )

        # 改进的中间处理层
        self.conv2 = nn.Sequential(
            ResidualBlock(out_channels * 8, out_channels * 8),
            DepthwiseSeparableConv(out_channels * 8, out_channels * 4, kernel_size=3, stride=1, padding=1),
        )
        self.bn2 = nn.BatchNorm2d(out_channels * 4)

        # 改进的解码器
        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        # 添加注意力模块到每个解码器后
        self.decoder_attention = nn.ModuleList(
            [
                CBAMBlock(out_channels * 2),
                CBAMBlock(out_channels),
                CBAMBlock(int(out_channels * 1 / 2)),
                CBAMBlock(int(out_channels * 1 / 8)),
            ]
        )

        # 通道压缩
        self.channel_linear = nn.Sequential(
            nn.Linear(int(out_channels * 1 / 8), 16), nn.ReLU(), nn.Linear(16, 1), nn.ReLU()
        )

        # 最终输出层
        self.conv_last = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)

    # 修改forward函数部分

    def forward(self, data, config, gpu_list, acc_result, mode):
        seq_x = data["input"]

        # encoder过程
        x_flat = rearrange(seq_x, "n l c h w -> (n l) c h w")  # [B*l, c, h, w]
        x = self.conv1(x_flat)
        x = self.bn1(x)
        x1 = self.relu(x)  # [B*l, out_channels, H/2, W/2]

        # 使用带有残差结构的编码器
        x2 = self.encoder1(x1)  # [B, out_channels*2, H/4, W/4]
        x3 = self.encoder2(x2)  # [B, out_channels*4, H/8, W/8]
        x = self.encoder3(x3)  # [B, out_channels*8, H/16, W/16]

        # 重组张量用于RNN处理
        seq_x_new = rearrange(x, "(n l) c h w -> n l c h w", n=seq_x.size(0), l=seq_x.size(1))

        # 通过RNN处理序列
        seq_x_new = self.rnn(seq_x_new)

        # 只取RNN输出的最后一个时间步
        # seq_x_new形状为[batch, seq_length-1, channels, height, width]
        last_step = seq_x_new[:, -1]  # 形状为[batch, channels, height, width]

        # 中间特征处理
        x = self.conv2(last_step)  #  [B, out_channels*8, H/16, W/16]
        x = self.bn2(x)
        x = self.relu(x)

        # 获取skip connections的最后一步
        x1_last = rearrange(x1, "(n l) c h w -> n l c h w", n=seq_x.size(0), l=seq_x.size(1))[:, -1]
        x2_last = rearrange(x2, "(n l) c h w -> n l c h w", n=seq_x.size(0), l=seq_x.size(1))[:, -1]
        x3_last = rearrange(x3, "(n l) c h w -> n l c h w", n=seq_x.size(0), l=seq_x.size(1))[:, -1]

        # decoder过程 - 直接使用最后一步的特征
        x = self.decoder1(x, x3_last)
        x = self.decoder_attention[0](x)

        x = self.decoder2(x, x2_last)
        x = self.decoder_attention[1](x)

        x = self.decoder3(x, x1_last)
        x = self.decoder_attention[2](x)

        x = self.decoder4(x)
        x = self.decoder_attention[3](x)

        # 通道压缩
        x_flat = rearrange(x, "n c h w -> n h w c")
        x_flat = self.channel_linear(x_flat).squeeze(-1)

        # 确保输入形状正确
        x_flat = x_flat.unsqueeze(1)  # 添加通道维度 [n, 1, h, w]

        # 最终输出
        x = self.conv_last(x_flat)

        x = x.squeeze(1)  # [n, h, w]

        if "label" in data.keys():
            label = data["label"]
            loss = self.criterion(x, label)
            acc_result = self.accuracy_function(x, label, config, acc_result)
            if mode == "test":
                return {"output": acc_result}
            else:
                return {"loss": loss, "acc_result": acc_result}
        else:
            return {"output": x}

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
