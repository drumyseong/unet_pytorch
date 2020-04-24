import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

"""To do:
-Pixel-wise Softmax?
-Implement more modulize way
"""

use_cuda = torch.cuda.is_available()
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        # Encoders
        self.encoder1 = nn.Sequential(double_conv_relu(3, 64), nn.MaxPool2d(kernel_size=2, stride=2))
        self.encoder2 = nn.Sequential(double_conv_relu(64, 128), nn.MaxPool2d(kernel_size=2, stride=2))
        self.encoder3 = nn.Sequential(double_conv_relu(128, 256), nn.MaxPool2d(kernel_size=2, stride=2))
        self.encoder4 = nn.Sequential(double_conv_relu(256, 512), nn.MaxPool2d(kernel_size=2, stride=2))

        # Decoders
        self.decoder1 = nn.Sequential(double_conv_relu(512, 1024), nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2))
        self.decoder2 = nn.Sequential(double_conv_relu(1024, 512), nn.ConvTranspose2d(512, 256, kernel_size=2, stride =2))
        self.decoder3 = nn.Sequential(double_conv_relu(512, 256), nn.ConvTranspose2d(256, 128, kernel_size=2, stride =2))
        self.decoder4 = nn.Sequential(double_conv_relu(256, 128), nn.ConvTranspose2d(128, 64, kernel_size=2, stride =2))
        self.decoder5 = nn.Sequential(double_conv_relu(128, 64), nn.Conv2d(64, num_classes, kernel_size = 1, stride = 1, pad = 0))

        self.encoders = nn.ModuleList[self.encoder1, self.encoder2, self.encoder3, self.encoder4]
        self.decoders = nn.ModuleList[self.decoder1, self.decoder2, self.decoder3, self.decoder4, self.decoder5]

        # need to init w ?
        """ 
        # Allocate GPU
        if use_cuda:
            self.encoders = self.encoders.cuda()
            self.decoders = self.decoders.cuda()
        """

    def forward(self, x):
        copyandcrop = [] # Data to copy and crop
        # Constracting Path
        for e in self.encoders:
            x = e(x)
            copyandcrop.append(x)
        x = copyandcrop.pop()
        # Expanding Path
        for d in self.decoders:
            x = torch.cat([x, copyandcrop.pop()], dim=1)
        out = x #(Use Pixel-wise Softmax)
        return out