from torch import nn
from torch import cat

class pub(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(pub, self).__init__()
        inter_channels = out_channels if in_channels > out_channels else out_channels//2

        layers = [
                    nn.Conv3d(in_channels, inter_channels, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv3d(inter_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True)
                 ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm3d(inter_channels))
            layers.insert(len(layers)-1, nn.BatchNorm3d(out_channels))
        self.pub = nn.Sequential(*layers)

    def forward(self, x):
        return self.pub(x)


class unet3dEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(unet3dEncoder, self).__init__()
        self.pub = pub(in_channels, out_channels, batch_norm)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.pub(x)
        return x,self.pool(x)


class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUp, self).__init__()
        self.pub = pub(in_channels//2+in_channels, out_channels, batch_norm)
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1):
        x = self.sample(x)
        #c1 = (x1.size(2) - x.size(2)) // 2
        #c2 = (x1.size(3) - x.size(3)) // 2
        #x1 = x1[:, :, c1:-c1, c2:-c2, c2:-c2]
        x = cat((x, x1), dim=1)
        x = self.pub(x)
        return x


class unet3d(nn.Module):
    def __init__(self, args):
        super(unet3d, self).__init__()
        init_channels = 4
        class_nums = 3
        batch_norm = True
        sample = True

        self.en1 = unet3dEncoder(init_channels, 64, batch_norm)
        self.en2 = unet3dEncoder(64, 128, batch_norm)
        self.en3 = unet3dEncoder(128, 256, batch_norm)
        self.en4 = unet3dEncoder(256, 512, batch_norm)

        self.up3 = unet3dUp(512, 256, batch_norm, sample)
        self.up2 = unet3dUp(256, 128, batch_norm, sample)
        self.up1 = unet3dUp(128, 64, batch_norm, sample)
        self.con_last = nn.Conv3d(64, class_nums, 1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1,x = self.en1(x)
        x2,x= self.en2(x)
        x3,x= self.en3(x)
        x4,_ = self.en4(x)

        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.con_last(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# import torch
# import torch.nn as nn
# from torch.nn import functional as F
#
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, bath_normal=False):
#         super(DoubleConv, self).__init__()
#         channels = out_channels / 2
#         if in_channels > out_channels:
#             channels = in_channels / 2
#
#         layers = [
#             # in_channels：输入通道数
#             # channels：输出通道数
#             # kernel_size：卷积核大小
#             # stride：步长
#             # padding：边缘填充
#             nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(True),
#
#             nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(True)
#         ]
#         if bath_normal:  # 如果要添加BN层
#             layers.insert(1, nn.BatchNorm3d(channels))
#             layers.insert(len(layers) - 1, nn.BatchNorm3d(out_channels))
#
#         # 构造序列器
#         self.double_conv = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.double_conv(x)
#
# class DownSampling(nn.Module):
#     def __init__(self, in_channels, out_channels, batch_normal=False):
#         super(DownSampling, self).__init__()
#         self.maxpool_to_conv = nn.Sequential(
#             nn.MaxPool3d(kernel_size=2, stride=2),
#             DoubleConv(in_channels, out_channels, batch_normal)
#         )
#
#     def forward(self, x):
#         return self.maxpool_to_conv(x)
#
# class UpSampling(nn.Module):
#     def __init__(self, in_channels, out_channels, batch_normal=False, bilinear=True):
#         super(UpSampling, self).__init__()
#         if bilinear:
#             # 采用双线性插值的方法进行上采样
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             # 采用反卷积进行上采样
#             self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#         self.conv = DoubleConv(in_channels + in_channels / 2, out_channels, batch_normal)
#
#     # inputs1：上采样的数据（对应图中黄色箭头传来的数据）
#     # inputs2：特征融合的数据（对应图中绿色箭头传来的数据）
#     def forward(self, inputs1, inputs2):
#         # 进行一次up操作
#         inputs1 = self.up(inputs1)
#
#         # 进行特征融合
#         outputs = torch.cat([inputs1, inputs2], dim=1)
#         outputs = self.conv(outputs)
#         return outputs
#
# class LastConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(LastConv, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         return self.conv(x)
#
# class unet3d(nn.Module):
#     def __init__(self, in_channels, num_classes=3, batch_normal=False, bilinear=True):
#
#
#         super(unet3d, self).__init__()
#         self.in_channels = in_channels
#         self.batch_normal = batch_normal
#         self.bilinear = bilinear
#
#         self.inputs = DoubleConv(in_channels, 64, self.batch_normal)
#         self.down_1 = DownSampling(64, 128, self.batch_normal)
#         self.down_2 = DownSampling(128, 256, self.batch_normal)
#         self.down_3 = DownSampling(256, 512, self.batch_normal)
#
#         self.up_1 = UpSampling(512, 256, self.batch_normal, self.bilinear)
#         self.up_2 = UpSampling(256, 128, self.batch_normal, self.bilinear)
#         self.up_3 = UpSampling(128, 64, self.batch_normal, self.bilinear)
#         self.outputs = LastConv(64, num_classes)
#
#     def forward(self, x):
#         # down 部分
#         x1 = self.inputs(x)
#         x2 = self.down_1(x1)
#         x3 = self.down_2(x2)
#         x4 = self.down_3(x3)
#
#         # up部分
#         x5 = self.up_1(x4, x3)
#         x6 = self.up_2(x5, x2)
#         x7 = self.up_3(x6, x1)
#         x = self.outputs(x7)
#
#         return x
