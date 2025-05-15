import torch
from torch import ne, nn

from .CBAM import CBAMBlock
from .GradientHighwayUnit import GHU
from .SpatioTemporalLSTMCell import SMLSTMCell


class EnhancedRNN(nn.Module):
    def __init__(
        self,
        img_channel,
        img_width,
        img_height,
        num_layers,
        num_hidden,
        seq_length,
        kernel_size=3,
        layer_norm=True,
        device="cuda",
    ):
        super().__init__()
        # 保留原RNN模块的基本参数
        self.width = img_width
        self.height = img_height
        self.img_channel = img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.seq_length = seq_length
        self.device = device

        # 增强型LSTM单元
        self.cell = nn.ModuleList()
        self.attentioncell = nn.ModuleList()

        # 添加全局特征提取器
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(img_channel, img_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(img_channel, img_channel, kernel_size=1),
            nn.Sigmoid(),
        )

        for i in range(num_layers):
            if i == 0:
                num_hidden_in = self.img_channel
            else:
                num_hidden_in = num_hidden[i - 1]

            # 使用原始的SMLSTMCell
            self.cell.append(
                SMLSTMCell(
                    num_hidden_in,
                    num_hidden[i],
                    self.width,
                    self.height,
                    filter_size=kernel_size,
                    layer_norm=layer_norm,
                    forget_bias=1.0,
                )
            )

            # 改进的注意力机制
            self.attentioncell.append(CBAMBlock(num_hidden[i]))

        self.ghu = GHU(num_hidden[0], img_width, img_height, kernel_size, layer_norm)

        # 输出卷积
        self.conv_last = nn.Conv2d(
            num_hidden[num_layers - 1], self.img_channel, kernel_size=1, stride=1, padding=0, bias=False
        ).to(self.device)

    def forward(self, frames_tensor):
        frames = frames_tensor.to(self.device)
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        # 初始化LSTM状态
        for i in range(self.num_layers):
            h_t.append(torch.zeros([batch, self.num_hidden[i], height, width]).to(self.device))
            c_t.append(torch.zeros([batch, self.num_hidden[i], height, width]).to(self.device))

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.device)
        z_t = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.device)

        # 增加全局上下文特征
        global_ctx = None

        for t in range(frames.shape[1] - 1):
            inputs = frames[:, t]

            # 处理全局上下文信息
            if global_ctx is None:
                global_ctx = self.global_context(inputs)
            else:
                curr_ctx = self.global_context(inputs)
                global_ctx = 0.8 * global_ctx + 0.2 * curr_ctx

            # 增强输入信息
            inputs = inputs * global_ctx

            # 第一层LSTM处理
            h_t[0], c_t[0], memory = self.cell[0](inputs, h_t[0], c_t[0], memory)
            h_t[0] = self.attentioncell[0](h_t[0])
            z_t = self.ghu(h_t[0], z_t)

            # 第二层LSTM处理
            h_t[1], c_t[1], memory = self.cell[1](z_t, h_t[1], c_t[1], memory)
            h_t[1] = self.attentioncell[1](h_t[1])

            # 深层LSTM处理
            for i in range(2, self.num_layers):
                # 添加残差连接
                prev_h = h_t[i].clone() if t > 0 else torch.zeros_like(h_t[i])
                h_t[i], c_t[i], memory = self.cell[i](h_t[i - 1], h_t[i], c_t[i], memory)
                h_t[i] = self.attentioncell[i](h_t[i])

                # 跨时间步残差连接
                if t > 0:
                    h_t[i] = h_t[i] + 0.2 * prev_h

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [batch, length, channel, height, width]
        next_frames = torch.stack(next_frames, dim=1)
        return next_frames


class RNN(nn.Module):
    def __init__(
        self,
        img_channel,
        img_width,
        img_height,
        num_layers,
        num_hidden,
        seq_length,
        kernel_size=3,
        layer_norm=True,
        device="cuda",
    ):
        """
        参数:
            img_channel: int, 输入图像的通道数
            img_width: int, 输入图像的宽度
            num_layers: int, 模型的层数
            num_hidden: list of int, 每层的隐藏单元数量
            seq_length: int, 序列的长度
            input_length: int, 输入序列的长度
            kernel_size: int, 卷积核的大小
            layer_norm: bool, 是否使用层归一化
            device: str, 运行模型的设备
        """
        super().__init__()
        self.width = img_width
        self.height = img_height
        self.img_channel = img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.seq_length = seq_length
        self.device = device
        self.cell = nn.ModuleList()
        self.attentioncell = nn.ModuleList()

        for i in range(0, num_layers):
            if i == 0:
                num_hidden_in = self.img_channel
            else:
                num_hidden_in = num_hidden[i - 1]
            self.cell.append(
                SMLSTMCell(
                    num_hidden_in,
                    num_hidden[i],
                    self.width,
                    self.height,
                    filter_size=kernel_size,
                    layer_norm=layer_norm,
                    forget_bias=1.0,
                )
            )
            self.attentioncell.append(CBAMBlock(num_hidden[i]))

        self.ghu = GHU(num_hidden[0], img_width, img_height, kernel_size, layer_norm)
        self.conv_last = nn.Conv2d(
            num_hidden[num_layers - 1], self.img_channel, kernel_size=1, stride=1, padding=0, bias=False
        ).to(self.device)

    def forward(self, frames_tensor):
        # [batch, length, channel, height, width]
        frames = frames_tensor.to(self.device)

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            h_t.append(torch.zeros([batch, self.num_hidden[i], height, width]).to(self.device))
            c_t.append(torch.zeros([batch, self.num_hidden[i], height, width]).to(self.device))

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.device)
        z_t = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.device)

        for t in range(frames.shape[1] - 1):

            inputs = frames[:, t]

            h_t[0], c_t[0], memory = self.cell[0](inputs, h_t[0], c_t[0], memory)
            h_t[0] = self.attentioncell[0](h_t[0])
            z_t = self.ghu(h_t[0], z_t)

            h_t[1], c_t[1], memory = self.cell[1](z_t, h_t[1], c_t[1], memory)
            h_t[1] = self.attentioncell[1](h_t[1])

            for i in range(2, self.num_layers):
                h_t[i], c_t[i], memory = self.cell[i](h_t[i - 1], h_t[i], c_t[i], memory)
                h_t[i] = self.attentioncell[i](h_t[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1])

            next_frames.append(x_gen)

        # [batch, length, channel, height, width]
        next_frames = torch.stack(next_frames, dim=1)

        return next_frames
