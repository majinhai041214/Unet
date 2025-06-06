"""网络中的一些模块"""
import torch
from torch import nn


class conv_block(nn.Module):
    """两个卷积层+一个Relu激活函数"""

    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """上采样模块：ConvTranspose2d + crop + 拼接 + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = conv_block(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 裁剪 x2，让它和 x1 尺寸匹配（假设 x2 是 encoder 路径）
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        x2 = x2[:, :, diffY // 2: x2.size(2) - (diffY - diffY // 2), diffX // 2: x2.size(3) - (diffX - diffX // 2)]

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1)
        )

    def forward(self,x):
        return self.block(x)
