"""定义U-net模型"""
import torch
from torch import nn
from unet_part import conv_block, down, Up, OutConv
import torch.nn.functional as F



class UNet(nn.Module):
    def __init__(self, n_channals, n_classes, bilinear=False):
        super().__init__()
        self.inc = conv_block(in_channels=n_channals, out_channels=64)
        self.down1 = down(in_channels=64, out_channels=128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = torch.sigmoid(self.outc(x))
        output = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)
        return output


