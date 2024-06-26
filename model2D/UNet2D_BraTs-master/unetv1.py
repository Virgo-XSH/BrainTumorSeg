import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from DeformConv2d import DeformConv2d
# from unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from layers import CBAM
import math

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1):
        super(conv_block, self).__init__()
        # self.downsample = nn.Sequential(
        #     nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(ch_out)
        # )
        self.conv = nn.Sequential(
            DeformConv2d(ch_in, ch_out, kernel_size, stride,padding),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        self.conv_only = nn.Conv2d(ch_out, ch_out, kernel_size, stride, padding)
        # self.SE = SE_Block(ch_out)
        self.BN = nn.BatchNorm2d(ch_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # identity = self.downsample(x)
        # x1 = self.conv(x)
        # x2 = self.conv_only(x1)
        # # x = self.SE(x2)
        # res_sum = torch.add(torch.add(x2, x1), identity)
        # return self.act(self.BN(res_sum))

        x1 = self.conv(x)
        x2 = self.conv_only(x1)
        res_sum = torch.add(x2, x1)
        return self.act(self.BN(res_sum))

# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResBlock, self).__init__()
#         # self.downsample = nn.Sequential(
#         #     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
#         #     nn.BatchNorm2d(out_channels))
#         # self.double_conv = DoubleConv(in_channels, out_channels)
#         # #self.double_conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=1)
#         # # self.cbam = ECA(out_channels)
#         # # self.cbam = CBAM(out_channels)
#         # self.down_sample = nn.MaxPool2d(2)
#         # self.relu = nn.ReLU()
#
#     def forward(self, x):
#         identity = self.downsample(x)
#         out = self.double_conv(x)
#         # out = self.cbam(out)
#         # F(x)+x
#         out = self.relu(out + identity)
#         return self.down_sample(out), out

class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv = conv_block(in_channels, out_channels)
        self.cbam = CBAM(input_channels=out_channels)

    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)))
        # y = F.relu(self.bn2(self.conv2(x)))
        # x = F.max_pool2d(y, 2, stride=2)
        #
        # return x, y

        x1 = self.conv(x)
        x1 = self.cbam(x1) + x1
        x2 = self.Maxpool(x1)
        # ap = self.Avgpool(x)
        # p1 = torch.add(mp, ap)
        return x2, x1

class Upsample_block(nn.Module):
    # def __init__(self, in_channels, out_channels):
    #     super(Upsample_block, self).__init__()
    #     self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)
    #     self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    #     self.bn1 = nn.BatchNorm2d(out_channels)
    #     self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    #     self.bn2 = nn.BatchNorm2d(out_channels)
    #
    # def forward(self, x, y):
    #     x = self.transconv(x)
    #     x = torch.cat((x, y), dim=1)
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = F.relu(self.bn2(self.conv2(x)))
    #
    #     return x

    # def __init__(self, in_channels, out_channels):
    #     super(Upsample_block, self).__init__()
    #     self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)
    #     self.conv = DoubleConv(in_channels, out_channels)
    #     # self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=1)
    #     self.downsample = nn.Sequential(
    #         nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
    #         nn.BatchNorm2d(out_channels))
    #
    # def forward(self, x, y):
    #     x = self.transconv(x)
    #     x = torch.cat((x, y), dim=1)
    #     out = self.conv(x)
    #     x = self.downsample(x)
    #     return F.relu(out + x)
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1):
        super(Upsample_block, self).__init__()
        self.transconv = nn.ConvTranspose2d(ch_in, ch_out, 4, padding=1, stride=2)
        self.up_conv = conv_block(ch_in, ch_out)

    def forward(self, x, y):
        x = self.transconv(x)
        x = torch.cat((x, y), dim=1)
        d = self.up_conv(x)
        return d

class U_Net_v1(nn.Module):
    def __init__(self, args):
        in_chan = 4
        out_chan = 3
        super(U_Net_v1, self).__init__()
        self.down1 = Downsample_block(in_chan, 64)
        self.down2 = Downsample_block(64, 128)
        self.down3 = Downsample_block(128, 256)
        self.down4 = Downsample_block(256, 512)
        self.down5 = Downsample_block(512, 1024)

        # self.conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        # self.bn1 = nn.BatchNorm2d(1024)
        # self.conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        # self.bn2 = nn.BatchNorm2d(1024)

        self.up4 = Upsample_block(1024, 512)
        self.up3 = Upsample_block(512, 256)
        self.up2 = Upsample_block(256, 128)
        self.up1 = Upsample_block(128, 64)
        self.outconv = nn.Conv2d(64, out_chan, 1)
        # self.outconvp1 = nn.Conv2d(64, out_chan, 1)
        # self.outconvm1 = nn.Conv2d(64, out_chan, 1)

    def forward(self, x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        # x = F.dropout2d(F.relu(self.bn1(self.conv1(x))))
        # x = F.dropout2d(F.relu(self.bn2(self.conv2(x))))
        _, x = self.down5(x)
        x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)
        x1 = self.outconv(x)
        return x1
