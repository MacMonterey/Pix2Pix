import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


### RSU-7 ###
class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, shape1=240):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

        # self.upscore65 = nn.UpsamplingNearest2d(size=(int(shape1 / 16), int(shape1 / 16)))
        # self.upscore54 = nn.UpsamplingNearest2d(size=(int(shape1 / 8), int(shape1 / 8)))
        # self.upscore43 = nn.UpsamplingNearest2d(size=(int(shape1 / 4), int(shape1 / 4)))
        # self.upscore32 = nn.UpsamplingNearest2d(size=(int(shape1 / 2), int(shape1 / 2)))
        # self.upscore21 = nn.UpsamplingNearest2d(size=(int(shape1), int(shape1)))

        self.upscore65 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore54 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore43 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore32 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore21 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        hx = x  # 128
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)  # 128
        hx = self.pool1(hx1)  # 64

        hx2 = self.rebnconv2(hx)  # 64
        hx = self.pool2(hx2)  # 32

        hx3 = self.rebnconv3(hx)  # 32
        hx = self.pool3(hx3)  # 16

        hx4 = self.rebnconv4(hx)  # 16
        hx = self.pool4(hx4)  # 8

        hx5 = self.rebnconv5(hx)  # 8
        hx = self.pool5(hx5)  # 4

        hx6 = self.rebnconv6(hx)  # 4

        hx7 = self.rebnconv7(hx6)  # 4

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))  # 4
        hx6up = self.upscore65(hx6d)  # 8
        #         print(hx6up.shape,hx5.shape)
        hx5d = self.rebnconv5d(torch.cat((hx6up, hx5), 1))
        hx5dup = self.upscore54(hx5d)  # 16

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = self.upscore43(hx4d)  # 32

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = self.upscore32(hx3d)  # 64

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self.upscore21(hx2d)  # 128

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, shape1=None):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

        self.upscore54 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore43 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore32 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore21 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = self.upscore54(hx5d)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = self.upscore43(hx4d)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = self.upscore32(hx3d)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self.upscore21(hx2d)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, shape1=None):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

        self.upscore43 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore32 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore21 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = self.upscore43(hx4d)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = self.upscore32(hx3d)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self.upscore21(hx2d)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, shape1=None):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

        self.upscore32 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore21 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = self.upscore32(hx3d)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self.upscore21(hx2d)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


##### U^2-Net ####
class GeneratorUNet(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, shape1=240):
        super(GeneratorUNet, self).__init__()

        #         self.shape1=shape

        self.stage1 = RSU7(in_ch, 32, 64, int(shape1))
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        #         print("4545",self.pool12.shape)

        self.stage2 = RSU6(64, 32, 128, int(shape1 / 2))
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256, int(shape1 / 4))
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512, int(shape1 / 8))
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256, int(shape1 / 8))
        self.stage3d = RSU5(512, 64, 128, int(shape1 / 4))
        self.stage2d = RSU6(256, 32, 64, int(shape1 / 2))
        self.stage1d = RSU7(128, 16, 64, int(shape1))

        self.side1 = nn.Conv2d(64, 3, 3, padding=1)
        self.side2 = nn.Conv2d(64, 3, 3, padding=1)
        self.side3 = nn.Conv2d(128, 3, 3, padding=1)
        self.side4 = nn.Conv2d(256, 3, 3, padding=1)
        self.side5 = nn.Conv2d(512, 3, 3, padding=1)
        self.side6 = nn.Conv2d(512, 3, 3, padding=1)

        self.upscore65 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore54 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore43 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore32 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore21 = nn.UpsamplingBilinear2d(size=(int(shape1), int(shape1)))

        # self.upout=nn.UpsamplingBilinear2d(size=(int(shape1),int(shape1)))

        self.outconv = nn.Conv2d(18, 3, 1)

    def forward(self, x):
        hx = x  # 128
        #         print("hx的形状",hx.shape)
        #         DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #         self.to(DEVICE)

        # stage 1
        hx1 = self.stage1(hx)  # 128

        hx = self.pool12(hx1)  # 64
        #         print("4545",hx.shape)

        # stage 2
        hx2 = self.stage2(hx)  # 64
        hx = self.pool23(hx2)  # 32

        # stage 3
        hx3 = self.stage3(hx)  # 32
        hx = self.pool34(hx3)  # 16

        # stage 4
        hx4 = self.stage4(hx)  # 16
        hx = self.pool45(hx4)  # 8

        # stage 5
        hx5 = self.stage5(hx)  # 8
        hx = self.pool56(hx5)  # 4

        # stage 6
        hx6 = self.stage6(hx)  # 4
        hx6up = self.upscore65(hx6)  # 8

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = self.upscore54(hx5d)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = self.upscore43(hx4d)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = self.upscore32(hx3d)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self.upscore21(hx2d)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = self.upscore21(d2)

        d3 = self.side3(hx3d)
        d3 = self.upscore21(d3)

        d4 = self.side4(hx4d)
        d4 = self.upscore21(d4)

        d5 = self.side5(hx5d)
        d5 = self.upscore21(d5)

        d6 = self.side6(hx6)
        d6 = self.upscore21(d6)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.tanh(d0), torch.tanh(d1), torch.tanh(d2), torch.tanh(d3), torch.tanh(d4), torch.tanh(d5), torch.tanh(d6)


### U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, ):
        super(U2NETP, self).__init__()

        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, 1, 3, padding=1)
        self.side2 = nn.Conv2d(64, 1, 3, padding=1)
        self.side3 = nn.Conv2d(64, 1, 3, padding=1)
        self.side4 = nn.Conv2d(64, 1, 3, padding=1)
        self.side5 = nn.Conv2d(64, 1, 3, padding=1)
        self.side6 = nn.Conv2d(64, 1, 3, padding=1)

        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.outconv = nn.Conv2d(6, 1, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = self.upscore2(hx6)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = self.upscore2(hx5d)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = self.upscore2(hx4d)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = self.upscore2(hx3d)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self.upscore2(hx2d)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = self.upscore2(d2)

        d3 = self.side3(hx3d)
        d3 = self.upscore3(d3)

        d4 = self.side4(hx4d)
        d4 = self.upscore4(d4)

        d5 = self.side5(hx5d)
        d5 = self.upscore5(d5)

        d6 = self.side6(hx6)
        d6 = self.upscore6(d6)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        # d00 = d0 + self.refconv(d0)

        return F.sigmoid(d0)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


def main():
    # model = Generator()
    # model.build(input_shape=(None, 512, 512, 3))
    # model.summary()
    x = torch.rand([2, 3, 512, 512])
    model = GeneratorUNet(3, 3, 512)
    d0, d1, d2, d3, d4, d5, d6 = model(x)
    print(d0.size())


if __name__ == '__main__':
    main()
