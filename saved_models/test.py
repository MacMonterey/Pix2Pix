import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import glob
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, 2, 1, bias=False),
                  nn.Conv2d(in_size, out_size, 3, 2, 1, bias=False),
                  nn.Conv2d(in_size, out_size, 3, 2, 1, bias=False),
                  ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        return x

def main():

    x = torch.rand([2, 3, 512, 512])
    model = UNetDown(3, 32)
    d0 = model(x)
    print(type(d0))
    print(d0.size())


if __name__ == '__main__':
    main()
    # img_list = glob.glob('../data/New_Data/val/*.jpg')
    # print
    # print(len(img_list))
    # print(img_list[99])
    #
    # img = cv2.imread('../images/U2netpix/100.png')
    # R = img[512:1024, 512:1024, 1]
    # plt.imshow(R)
    # plt.show()
    # print(type(img))
    # print(img.shape)
