import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from tools.accuracy_init import init_accuracy_function


class ConvLSTMCell(nn.Module):
    """
    单个 ConvLSTM 单元
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # Concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size):
        height, width = spatial_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
        )


class DownsampleBlock(nn.Module):
    """
    下采样模块: 减少空间尺寸，增加通道数
    """

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=scale_factor, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UpsampleBlock(nn.Module):
    """
    上采样模块: 增加空间尺寸，减少通道数
    """

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return self.relu(self.bn(x))


class ConvLSTM(nn.Module):
    """
    多层 ConvLSTM 模块
    """

    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            self.layers.append(ConvLSTMCell(cur_input_dim, hidden_dims[i], kernel_size))

    def forward(self, input_tensor):
        # input_tensor: [batch, seq_len, channels, height, width]
        batch_size, seq_len, _, height, width = input_tensor.size()
        h, c = self.init_hidden(batch_size, (height, width))

        outputs = []
        for t in range(seq_len):
            x = input_tensor[:, t]
            for i, layer in enumerate(self.layers):
                h[i], c[i] = layer(x, (h[i], c[i]))
                x = h[i]
            outputs.append(h[-1])

        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, hidden_dim, height, width]
        return outputs

    def init_hidden(self, batch_size, spatial_size):
        h, c = [], []
        for layer in self.layers:
            h_i, c_i = layer.init_hidden(batch_size, spatial_size)
            h.append(h_i)
            c.append(c_i)
        return h, c


class ConvLSTMModel(nn.Module):
    """
    基于 ConvLSTM 的降水校正模型
    增加下采样和上采样以提高训练速度
    """

    def __init__(self, config, gpu_list, *args, **params):
        super(ConvLSTMModel, self).__init__()
        in_channels = config.getint("model", "in_channels")
        out_channels = config.getint("model", "out_channels")
        num_layers = config.getint("model", "num_layers")
        num_hiddens_str = config.get("model", "num_hiddens")
        num_hiddens = [int(x) for x in num_hiddens_str.split(",")]

        # 获取下采样倍数，默认为2
        self.scale_factor = config.getint("model", "scale_factor", fallback=2)

        self.accuracy_function = init_accuracy_function(config, *args, **params)

        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 下采样层
        self.down1 = DownsampleBlock(out_channels, out_channels * 2, scale_factor=self.scale_factor)
        self.down2 = DownsampleBlock(out_channels * 2, out_channels * 4, scale_factor=self.scale_factor)
        self.down3 = DownsampleBlock(out_channels * 4, out_channels * 8, scale_factor=self.scale_factor)
        self.down4 = DownsampleBlock(out_channels * 8, out_channels * 16, scale_factor=self.scale_factor)

        # 调整ConvLSTM的输入通道数以匹配下采样后的通道数
        self.convlstm = ConvLSTM(
            input_dim=out_channels * 16,  # 修改为正确的下采样后通道数
            hidden_dims=num_hiddens,
            kernel_size=3,
            num_layers=num_layers,
        )

        self.up1 = UpsampleBlock(num_hiddens[-1], num_hiddens[-1] // 2, scale_factor=self.scale_factor)
        self.up2 = UpsampleBlock(num_hiddens[-1] // 2, num_hiddens[-1] // 4, scale_factor=self.scale_factor)
        self.up3 = UpsampleBlock(num_hiddens[-1] // 4, num_hiddens[-1] // 8, scale_factor=self.scale_factor)
        self.up4 = UpsampleBlock(num_hiddens[-1] // 8, num_hiddens[-1] // 16, scale_factor=self.scale_factor)

        # 最终卷积层 - 修改输入通道数以匹配上采样的输出
        self.conv_last = nn.Conv2d(num_hiddens[-1] // 16, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, data, config, gpu_list, acc_result, mode):

        x = x_flat = rearrange(data["input"], "n l c h w -> (n l) c h w")

        # 初始卷积
        x = self.relu(self.bn1(self.conv1(x)))  # [batch * seq_len, out_channels, height, width]

        # 下采样
        x = self.down1(x)  # [batch * seq_len, out_channels * 2, height // 2, width // 2]
        x = self.down2(x)  # [batch * seq_len, out_channels * 4, height // 4, width // 4]
        x = self.down3(x)  # [batch * seq_len, out_channels * 8, height // 8, width // 8]
        x = self.down4(x)  # [batch * seq_len, out_channels * 16, height // 16, width // 16]

        # 重新调整维度以适应 ConvLSTM 输入格式
        x = rearrange(x, "(n l) c h w -> n l c h w", n=data["input"].shape[0], l=data["input"].shape[1])

        # ConvLSTM处理 - 在降采样的空间尺寸上运行，减少计算量
        x = self.convlstm(x)[:, -1]  # 只保留最后一个时间步 [batch, hidden_dim, reduced_height, reduced_width]

        # 上采样 - 恢复到原始空间尺寸
        x = self.up1(x)  # [batch, hidden_dim // 2, height // 8, width // 8]
        x = self.up2(x)  # [batch, hidden_dim // 4, height // 4, width // 4]
        x = self.up3(x)  # [batch, hidden_dim // 8, height, width]
        x = self.up4(x)  # [batch, hidden_dim // 16, height, width]

        # 最终卷积
        x = self.conv_last(x).squeeze(1)  # [batch, height, width]

        # 计算 loss/metric
        if "label" in data.keys():
            label = data["label"]
            loss = self.criterion(x=x, label=label)
            acc_result = self.accuracy_function(x, label, config, acc_result)
            if mode == "train":
                return {"loss": loss, "acc_result": acc_result}
            else:
                return {"loss": loss, "acc_result": acc_result, "output": x, "label": label}
        else:
            return {"loss": None, "acc_result": acc_result, "output": x}

    def criterion(self, x, label):
        # 引入权重调整不同降水等级的损失贡献
        mask_light = (label > 0.1) & (label <= 5.0)
        mask_moderate = (label > 5.0) & (label <= 15.0)
        mask_heavy = label > 15.0

        # 基础损失计算
        loss_L1 = F.l1_loss(x, label, reduction="none")
        loss_MSE = F.mse_loss(x, label, reduction="none")

        # 对不同降水等级使用不同权重
        weighted_loss = loss_L1 * (1.0 + 5.0 * mask_light + 3.0 * mask_moderate + 2.0 * mask_heavy) + loss_MSE * (
            1.0 + 3.0 * mask_light + 2.0 * mask_moderate + 1.0 * mask_heavy
        )

        return weighted_loss.mean()
