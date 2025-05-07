import torch
import torch.nn as nn

from .DSW import DepthwiseSeparableConv


class SMLSTMCell(nn.Module):
    def __init__(
        self, num_hidden_in, num_hidden, width, heigth, filter_size, layer_norm, forget_bias=1.0, device="cuda"
    ):
        """SM-LSTM 单元
        Args:
            num_hidden_in: 输入通道数
            num_hidden: 隐藏单元通道数
            seq_shape: 输入序列形状   [batch, channel, height, width]
            filter_size: 卷积核大小
            layer_norm: 是否使用LayerNorm
            forget_bias: 遗忘门偏置
        """
        super().__init__()

        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden
        self.height = heigth
        self.width = width
        self.filter_size = filter_size
        self.padding = filter_size // 2
        self.layer_norm = layer_norm
        self._forget_bias = forget_bias

        if layer_norm:
            self.conv_x = nn.Sequential(
                DepthwiseSeparableConv(
                    num_hidden_in, num_hidden * 7, kernel_size=filter_size, stride=1, padding=self.padding
                ),
                nn.LayerNorm([num_hidden * 7, self.height, self.width]),
            ).to(device)
            self.conv_h = nn.Sequential(
                DepthwiseSeparableConv(
                    num_hidden, num_hidden * 4, kernel_size=filter_size, stride=1, padding=self.padding
                ),
                nn.LayerNorm([num_hidden * 4, self.height, self.width]),
            ).to(device)
            self.conv_c = nn.Sequential(
                DepthwiseSeparableConv(
                    num_hidden, num_hidden * 3, kernel_size=filter_size, stride=1, padding=self.padding
                ),
                nn.LayerNorm([num_hidden * 3, self.height, self.width]),
            ).to(device)
            self.conv_m = nn.Sequential(
                DepthwiseSeparableConv(
                    num_hidden, num_hidden * 3, kernel_size=filter_size, stride=1, padding=self.padding
                ),
                nn.LayerNorm([num_hidden * 3, self.height, self.width]),
            ).to(device)
            self.conv_c2m = nn.Sequential(
                DepthwiseSeparableConv(
                    num_hidden, num_hidden * 4, kernel_size=filter_size, stride=1, padding=self.padding
                ),
                nn.LayerNorm([num_hidden * 4, self.height, self.width]),
            ).to(device)
            self.conv_m2o = nn.Sequential(
                DepthwiseSeparableConv(num_hidden, num_hidden, kernel_size=filter_size, stride=1, padding=self.padding),
                nn.LayerNorm([num_hidden, self.height, self.width]),
            ).to(device)
        else:
            self.conv_x = DepthwiseSeparableConv(
                num_hidden_in, num_hidden * 7, kernel_size=filter_size, stride=1, padding=self.padding
            ).to(device)
            self.conv_h = DepthwiseSeparableConv(
                num_hidden, num_hidden * 4, kernel_size=filter_size, stride=1, padding=self.padding
            ).to(device)
            self.conv_c = DepthwiseSeparableConv(
                num_hidden, num_hidden * 3, kernel_size=filter_size, stride=1, padding=self.padding
            ).to(device)
            self.conv_m = DepthwiseSeparableConv(
                num_hidden, num_hidden * 3, kernel_size=filter_size, stride=1, padding=self.padding
            ).to(device)
            self.conv_c2m = DepthwiseSeparableConv(
                num_hidden, num_hidden * 4, kernel_size=filter_size, stride=1, padding=self.padding
            ).to(device)
            self.conv_m2o = DepthwiseSeparableConv(
                num_hidden, num_hidden, kernel_size=filter_size, stride=1, padding=self.padding
            ).to(device)

        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False).to(
            device
        )

    def forward(self, x_t, h_t, c_t, m_t):
        if h_t is None:
            h_t = torch.zeros(self.batch, self.num_hidden, self.height, self.width)
        if c_t is None:
            c_t = torch.zeros(self.batch, self.num_hidden, self.height, self.width)
        if m_t is None:
            m_t = torch.zeros(self.batch, self.num_hidden, self.height, self.width)

        h_cc = self.conv_h(h_t)
        c_cc = self.conv_c(c_t)
        m_cc = self.conv_m(m_t)

        i_h, g_h, f_h, o_h = torch.chunk(h_cc, 4, dim=1)
        i_c, g_c, f_c = torch.chunk(c_cc, 3, dim=1)
        i_m, f_m, m_m = torch.chunk(m_cc, 3, dim=1)

        if x_t is None:
            i = torch.sigmoid(i_h + i_c)
            f = torch.sigmoid(f_h + f_c + self._forget_bias)
            g = torch.tanh(g_h + g_c)
        else:
            x_cc = self.conv_x(x_t)
            i_x, g_x, f_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.chunk(x_cc, 7, dim=1)

            i = torch.sigmoid(i_x + i_h + i_c)
            f = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
            g = torch.tanh(g_x + g_h + g_c)

        c_new = f * c_t + i * g

        c2m_cc = self.conv_c2m(c_new)
        i_c, g_c, f_c, o_c = torch.chunk(c2m_cc, 4, dim=1)

        if x_t is None:
            i2 = torch.sigmoid(i_c + i_m)
            f2 = torch.sigmoid(f_c + f_m + self._forget_bias)
            g2 = torch.tanh(g_c)
        else:
            i2 = torch.sigmoid(i_x_prime + i_c + i_m)
            f2 = torch.sigmoid(f_x_prime + f_c + f_m + self._forget_bias)
            g2 = torch.tanh(g_x_prime + g_c)

        m_new = f2 * torch.tanh(m_m) + i2 * g2
        o_m = self.conv_m2o(m_new)

        if x_t is None:
            o_t = torch.sigmoid(o_h + o_c + o_m)
        else:
            o_t = torch.sigmoid(o_x + o_h + o_c + o_m)

        o_cc = torch.cat([c_new, m_new], dim=1)
        o_cc = self.conv_last(o_cc)
        h_new = o_t * torch.tanh(o_cc)

        return h_new, c_new, m_new


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_t = torch.rand(2, 3, 32, 32).to(device)
    h_t = torch.rand(2, 64, 32, 32).to(device)
    c_t = torch.rand(2, 64, 32, 32).to(device)
    m_t = torch.rand(2, 64, 32, 32).to(device)

    smlstm = SMLSTMCell(3, 64, x_t.size(2), x_t.size(3), 3, True).to(device)
    h_new, c_new, m_new = smlstm(x_t, h_t, c_t, m_t)
    print(h_new.size(), c_new.size(), m_new.size())
