import torch
import torch.nn as nn
import torch.nn.functional as F
from res2net import res2net50_v1b_26w_4s
import torchvision


class CNN1(nn.Module):
    def __init__(self,channel,map_size,pad):
        super(CNN1,self).__init__()
        self.weight = nn.Parameter(torch.ones(channel,channel,map_size,map_size),requires_grad=False).cuda()
        self.bias = nn.Parameter(torch.zeros(channel),requires_grad=False).cuda()
        self.pad = pad
        self.norm = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = F.conv2d(x,self.weight,self.bias,stride=1,padding=self.pad)
        out = self.norm(out)
        out = self.relu(out)
        return out


class M2SNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self):
        super(M2SNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.conv_3 = CNN1(64,3,1)
        self.conv_5 = CNN1(64, 5, 2)


        # self.x5_dem_1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # self.x5_x4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # self.x5_x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # self.x5_x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        # self.x5_dem_4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # self.x5_x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
        #                                  nn.ReLU(inplace=True))

        self.level3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # self.x5_dem_5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      # nn.ReLU(inplace=True))

        # self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # self.output2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # self.output1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x):
        input = x

        # '''
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        x1 = self.resnet.maxpool(x)      # bs, 64, 88, 88
        # ---- low-level features ----
        x2 = self.resnet.layer1(x1)      # bs, 256, 88, 88
        x3 = self.resnet.layer2(x2)     # bs, 512, 44, 44
        x4 = self.resnet.layer3(x3)     # bs, 1024, 22, 22
        x5 = self.resnet.layer4(x4)     # bs, 2048, 11, 11
        # '''


        # x5_dem_1 = self.x5_dem_1(x5)
        x4_dem_1 = self.x4_dem_1(x4)
        x3_dem_1 = self.x3_dem_1(x3)
        x2_dem_1 = self.x2_dem_1(x2)

        # x5_dem_1_up = F.upsample(x5_dem_1, size=x4.size()[2:], mode='bilinear')
        # x5_dem_1_up_map1 = self.conv_3(x5_dem_1_up)
        x4_dem_1_map1 = self.conv_3(x4_dem_1)
        # x5_dem_1_up_map2 = self.conv_5(x5_dem_1_up)
        x4_dem_1_map2 = self.conv_5(x4_dem_1)
        # x5_4 = self.x5_x4(
        #     abs(x5_dem_1_up - x4_dem_1)+abs(x5_dem_1_up_map1-x4_dem_1_map1)+abs(x5_dem_1_up_map2-x4_dem_1_map2))


        x4_dem_1_up = F.upsample(x4_dem_1, size=x3.size()[2:], mode='bilinear')
        x4_dem_1_up_map1 = self.conv_3(x4_dem_1_up)
        x3_dem_1_map1 = self.conv_3(x3_dem_1)
        x4_dem_1_up_map2 = self.conv_5(x4_dem_1_up)
        x3_dem_1_map2 = self.conv_5(x3_dem_1)
        x4_3 = self.x4_x3(
            abs(x4_dem_1_up - x3_dem_1)+abs(x4_dem_1_up_map1-x3_dem_1_map1)+abs(x4_dem_1_up_map2-x3_dem_1_map2) )


        x3_dem_1_up = F.upsample(x3_dem_1, size=x2.size()[2:], mode='bilinear')
        x3_dem_1_up_map1 = self.conv_3(x3_dem_1_up)
        x2_dem_1_map1 = self.conv_3(x2_dem_1)
        x3_dem_1_up_map2 = self.conv_5(x3_dem_1_up)
        x2_dem_1_map2 = self.conv_5(x2_dem_1)
        x3_2 = self.x3_x2(
            abs(x3_dem_1_up - x2_dem_1)+abs(x3_dem_1_up_map1-x2_dem_1_map1)+abs(x3_dem_1_up_map2-x2_dem_1_map2) )


        x2_dem_1_up = F.upsample(x2_dem_1, size=x1.size()[2:], mode='bilinear')
        x2_dem_1_up_map1 = self.conv_3(x2_dem_1_up)
        x1_map1 = self.conv_3(x1)
        x2_dem_1_up_map2 = self.conv_5(x2_dem_1_up)
        x1_map2 = self.conv_5(x1)
        x2_1 = self.x2_x1(abs(x2_dem_1_up - x1)+abs(x2_dem_1_up_map1-x1_map1)+abs(x2_dem_1_up_map2-x1_map2) )


        # x5_4_up = F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear')
        # x5_4_up_map1 = self.conv_3(x5_4_up)
        # x4_3_map1 = self.conv_3(x4_3)
        # x5_4_up_map2 = self.conv_5(x5_4_up)
        # x4_3_map2 = self.conv_5(x4_3)
        # x5_4_3 = self.x5_x4_x3(abs(x5_4_up - x4_3) +abs(x5_4_up_map1-x4_3_map1)+abs(x5_4_up_map2-x4_3_map2))


        x4_3_up = F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear')
        x4_3_up_map1 = self.conv_3(x4_3_up)
        x3_2_map1 = self.conv_3(x3_2)
        x4_3_up_map2 = self.conv_5(x4_3_up)
        x3_2_map2 = self.conv_5(x3_2)
        x4_3_2 = self.x4_x3_x2(abs(x4_3_up - x3_2)+abs(x4_3_up_map1-x3_2_map1)+abs(x4_3_up_map2-x3_2_map2) )


        x3_2_up = F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear')
        x3_2_up_map1 = self.conv_3(x3_2_up)
        x2_1_map1 = self.conv_3(x2_1)
        x3_2_up_map2 = self.conv_5(x3_2_up)
        x2_1_map2 = self.conv_5(x2_1)
        x3_2_1 = self.x3_x2_x1(abs(x3_2_up - x2_1)+abs(x3_2_up_map1-x2_1_map1)+abs(x3_2_up_map2-x2_1_map2) )


        # x5_4_3_up = F.upsample(x5_4_3, size=x4_3_2.size()[2:], mode='bilinear')
        # x5_4_3_up_map1 = self.conv_3(x5_4_3_up)
        # x4_3_2_map1 = self.conv_3(x4_3_2)
        # x5_4_3_up_map2 = self.conv_5(x5_4_3_up)
        # x4_3_2_map2 = self.conv_5(x4_3_2)
        # x5_4_3_2 = self.x5_x4_x3_x2(
        #     abs(x5_4_3_up - x4_3_2)+abs(x5_4_3_up_map1-x4_3_2_map1)+abs(x5_4_3_up_map2-x4_3_2_map2) )


        x4_3_2_up = F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear')
        x4_3_2_up_map1 = self.conv_3(x4_3_2_up)
        x4_3_2_up_map2 = self.conv_5(x4_3_2_up)
        x3_2_1_map1 = self.conv_3(x3_2_1)
        x3_2_1_map2 = self.conv_5(x3_2_1)
        x4_3_2_1 = self.x4_x3_x2_x1(
            abs(x4_3_2_up - x3_2_1) +abs(x4_3_2_up_map1-x3_2_1_map1)+abs(x4_3_2_up_map2-x3_2_1_map2))


        # x5_dem_4 = self.x5_dem_4(x5_4_3_2)
        # x5_dem_4_up = F.upsample(x5_dem_4, size=x4_3_2_1.size()[2:], mode='bilinear')
        # x5_dem_4_up_map1 = self.conv_3(x5_dem_4_up)
        # x4_3_2_1_map1 = self.conv_3(x4_3_2_1)
        # x5_dem_4_up_map2 = self.conv_5(x5_dem_4_up)
        # x4_3_2_1_map2 = self.conv_5(x4_3_2_1)
        # x5_4_3_2_1 = self.x5_x4_x3_x2_x1(
        #     abs(x5_dem_4_up - x4_3_2_1)+abs(x5_dem_4_up_map1-x4_3_2_1_map1)+abs(x5_dem_4_up_map2-x4_3_2_1_map2) )

        level3 = x4_3
        level2 = self.level2(x3_2 + x4_3_2)
        level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1)

        output3 = self.output3(F.upsample(x4_dem_1,size=level3.size()[2:], mode='bilinear') + level3)
        output2 = self.output2(F.upsample(output3,size=level2.size()[2:], mode='bilinear') + level2)
        output1 = self.output1(F.upsample(output2,size=level1.size()[2:], mode='bilinear') + level1)
        return output1

