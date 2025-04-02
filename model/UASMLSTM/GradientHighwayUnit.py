import torch
import torch.nn as nn

from DSW import DepthwiseSeparableConv


class GHU(nn.Module):
    def __init__(self,num_features, width, height, filter_size, layer_norm=False, initializer=0.001, device='cuda'):
        super().__init__()
        self.filter_size = filter_size
        self.num_features = num_features
        self.layer_norm = layer_norm

        if layer_norm:
            self.conv_x = nn.Sequential(
                DepthwiseSeparableConv(num_features, num_features * 2, filter_size, 1, padding=filter_size//2),
                nn.LayerNorm([num_features * 2, height, width])
            ).to(device)
            self.conv_z = nn.Sequential(
                DepthwiseSeparableConv(num_features, num_features * 2, filter_size, 1, padding=filter_size//2),
                nn.LayerNorm([num_features * 2, height, width])
            ).to(device)
        else:   
            self.conv_x = DepthwiseSeparableConv(num_features, num_features * 2, filter_size, 1, padding=filter_size//2).to(device)
            self.conv_z = DepthwiseSeparableConv(num_features, num_features * 2, filter_size, 1, padding=filter_size//2).to(device)

    def forward(self,x,z):
        if z is None:
            z = torch.zeros_like(x)
        
        x_cc = self.conv_x(x)
        z_cc = self.conv_z(z)

        gates = x_cc + z_cc

        p, u = gates.chunk(2, dim=1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1-u) * z

        return z_new

if __name__ == "__main__":
    x = torch.randn(1,3,64,64)
    z = torch.randn(1,3,64,64)
    ghu = GHU(3,3)
    out = ghu(x,z)
    print(out.shape)
