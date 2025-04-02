import torch
from torch import nn
from einops import rearrange
from SpatioTemporalLSTMCell import SMLSTMCell
from DSW import DepthwiseSeparableConv
from CBAM import CBAMBlock
from GradientHighwayUnit import GHU



class EncoderBottleneck(nn.Module):
    '''
    下采样模块
    输入形状: (batch, in_channel, height, width)
    '''
    def __init__(self, in_channel, out_channel, stride=1,base_channel=64):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel)
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
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x = self.relu(x_1 + x_2)
        return x

class DecoderBottleneck(nn.Module):
    '''
    上采样模块
    输入形状: (batch, in_channel, height, width)
    '''
    def __init__(self, in_channel, out_channel, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            DepthwiseSeparableConv(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x, x_concat], dim=1) # x (batch, channel, height, width)
        
        x = self.layer(x)
        return x

class RNN(nn.Module):
    def __init__(self, img_channel, img_width, num_layers, num_hidden, seq_length=20, input_length=10, kernel_size=3, layer_norm=True,device='cuda'):
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
        self.img_channel = img_channel 
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.seq_length = seq_length
        self.input_length = input_length
        self.device = device

        width = img_width 
        heigth = img_width 

        self.cell = nn.ModuleList()
        self.attentioncell = nn.ModuleList()

        for i in range(0, num_layers):
            if i == 0:
                num_hidden_in = self.img_channel
            else:
                num_hidden_in = num_hidden[i-1]
            self.cell.append(SMLSTMCell(num_hidden_in, num_hidden[i], width, heigth, filter_size=kernel_size, layer_norm=layer_norm, forget_bias=1.0))
            self.attentioncell.append(CBAMBlock(num_hidden[i]))
        
        self.ghu = GHU(num_hidden[0], width, heigth, kernel_size, layer_norm)
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.img_channel, kernel_size=1, stride=1, padding=0, bias=False).to(self.device)

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

        for t in range(frames.shape[1]-1):
            # if t < self.input_length:
            #     inputs = frames[:, t]
            # else:
            #     inputs = mask_true[:, t- self.input_length] * frames[:, t] + (1-mask_true[:, t- self.input_length]) * x_gen

            inputs = frames[:, t]

            h_t[0], c_t[0], memory = self.cell[0](inputs, h_t[0], c_t[0], memory)
            h_t[0] = self.attentioncell[0](h_t[0])
            z_t = self.ghu(h_t[0], z_t)

            h_t[1], c_t[1], memory = self.cell[1](z_t, h_t[1], c_t[1], memory)
            h_t[1] = self.attentioncell[1](h_t[1])

            for i in range(2, self.num_layers):
                h_t[i], c_t[i], memory = self.cell[i](h_t[i-1], h_t[i], c_t[i], memory)
                h_t[i] = self.attentioncell[i](h_t[i])
            
            x_gen = self.conv_last(h_t[self.num_layers-1])

            next_frames.append(x_gen)
        
        # [batch, length, channel, height, width]
        next_frames = torch.stack(next_frames, dim=1)

        return next_frames

class UASMLSTM(nn.Module): 
    def __init__(self, img_dim, in_channels, out_channels, num_layers, num_hiddens, class_num, layer_norm=False):
        '''
        config必要参数：

        img_dim (int): 输入图像的宽度。
        in_channels (int): 输入图像的通道数。
        ou_channels (int): 输出图像的通道数。
        num_layers (int): 模型的层数。
        num_hiddens (list of int): 每层的隐藏单元数量。
        class_num (int): 输出类别数。
        layer_norm (bool): 是否使用层归一化。
        '''
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False) #(batch, channel, height, width)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)
        self.rnn = RNN(img_channel= out_channels * 8, img_width= img_dim // 4, num_layers=num_layers, num_hidden=num_hiddens, layer_norm=layer_norm)
        
        self.conv2 = DepthwiseSeparableConv(out_channels * 8, out_channels * 4, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels * 4)

        self.decoder1 = DecoderBottleneck(out_channels*8, out_channels*2)
        self.decoder2 = DecoderBottleneck(out_channels*4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels*2, int(out_channels*1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels*1 / 2), int(out_channels*1 / 8))

        self.conv_last = nn.Conv2d(int(out_channels*1 / 8), class_num, kernel_size=1)

        self.to_precip_grid = nn.Sequential(
            nn.Conv2d(class_num, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, seq_x):

        #encoder过程
        x_flat = rearrange(seq_x, 'n l c h w -> (n l) c h w')
        x = self.conv1(x_flat)
        x = self.bn1(x)
        x1 = self.relu(x)
      
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)

        x = self.encoder3(x3)

        seq_x_new = rearrange(x, '(n l) c h w -> n l c h w', n=seq_x.size(0), l=seq_x.size(1))

        seq_x_new,loss = self.rnn(seq_x_new)

        x_flat =  rearrange(seq_x_new, 'n l c h w -> (n l) c h w')
        x = self.conv2(x_flat)
        x = self.bn2(x)
        x = self.relu(x)

        seq_x_new = rearrange(x, '(n l) c h w -> n l c h w', n=seq_x.size(0), l=seq_x.size(1)-1)
        x1 = rearrange(x1, '(n l) c h w -> n l c h w', n=seq_x.size(0), l=seq_x.size(1))[:,1:]
        x2 = rearrange(x2, '(n l) c h w -> n l c h w', n=seq_x.size(0), l=seq_x.size(1))[:,1:]
        x3 = rearrange(x3, '(n l) c h w -> n l c h w', n=seq_x.size(0), l=seq_x.size(1))[:,1:]
        x_flat_new = rearrange(seq_x_new, 'n l c h w -> (n l) c h w')
        x1 = rearrange(x1, 'n l c h w -> (n l) c h w')
        x2 = rearrange(x2, 'n l c h w -> (n l) c h w')
        x3 = rearrange(x3, 'n l c h w -> (n l) c h w')

        #decoder过程
        x = self.decoder1(x_flat_new, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)

        #降水MLP过程
        x = self.conv_last(x)
        # UNET最终得到的图像
        seq_x_new = rearrange(x, '(n l) c h w -> n l c h w', n=seq_x.size(0), l=seq_x.size(1)-1)
        
        return seq_x_new


if __name__ == '__main__':

    u = UASMLSTM(512,3,64,4,[64,64,64,64],1).to('cuda')
    x = torch.randn(3,7,3,512,512).to('cuda')
    y,loss = u(x)
    print(y.shape)