import torch
from torch import nn
from torch.nn import functional as F
from mca_module import MCALayer
from HWD import Down_wt

# VGG Block 可由两层conv3或三层conv3组成，两层的感受野和一层conv5一样，三层conv3的感受野和一层conv7是一样的，但是能够减少计算量，
# 包含两层conv3的VGG block 代码, 无池化层

# class VGGBlock(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
#         super(VGGBlock, self).__init__()
#         self.act_func = act_func
#         self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(middle_channels)
#         self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.act_func(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.act_func(out)
#
#         return out


# 两个普通卷积层
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

# 由两个普通卷积层和一个跳跃连接构成的ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # self.double_conv=DoubleConv(in_channels,out_channels)

        self.one_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.two_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.downsample_two = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.down_sample = nn.MaxPool2d(2)
        # self.down_sample = Down_wt(out_channels, out_channels)
        self.mca = MCALayer(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x)
        res_one = self.relu(identity + self.one_conv(x));
        res_two = self.relu(self.two_conv(res_one) + self.downsample_two(res_one))
        mca_data = self.mca(res_two)
        # out = self.relu(mca_data + res_two)
        out = mca_data + res_two

        # identity = self.downsample(x)
        # out = self.double_conv(x)
        # out = self.mca(out)
        # # F(x)+x
        # out = self.relu(out + identity)
        return self.down_sample(out), out

class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_block, self).__init__()
        # self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, y):
        x = self.transconv(x)
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x

class MCNet(nn.Module):
    def __init__(self, args):
        in_chan = 4
        out_chan = 3
        super(MCNet, self).__init__()
        self.down1 = ResBlock(in_chan, 64)
        self.down2 = ResBlock(64, 128)
        self.down3 = ResBlock(128, 256)

        self.conv1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        # self.down4 = ResBlock(256, 512)
        # self.conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        # self.bn1 = nn.BatchNorm2d(1024)
        # self.conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        # self.bn2 = nn.BatchNorm2d(1024)
        # self.up4 = Upsample_block(1024, 512)
        self.up3 = Upsample_block(512, 256)
        self.up2 = Upsample_block(256, 128)
        self.up1 = Upsample_block(128, 64)
        self.outconv = nn.Conv2d(64, out_chan, 1)

    def forward(self, x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        # x, y4 = self.down4(x)
        x = F.dropout2d(F.relu(self.bn1(self.conv1(x))))
        x = F.dropout2d(F.relu(self.bn2(self.conv2(x))))
        # x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)
        x1 = self.outconv(x)

        return x1
