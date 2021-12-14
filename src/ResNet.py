
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', stride=stride)


class BasicBlock(nn.Module):
    """ 
    Basic Block from a ResNet.
    """

    def __init__(self, in_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            conv3x3(in_channels, in_channels, stride),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            conv3x3(in_channels, in_channels, stride)
        )

    def forward(self, x):
        out = self.block(x)
        return out


class DownsamplingBlock(nn.Module):
    """ 
    ResNet Block that implements downsampling
    """

    def __init__(self, in_channels, stride=1):
        super().__init__()
        out_channels = in_channels * 2
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=2, padding='valid'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            conv3x3(out_channels, out_channels, stride)
        )
        self.downsample = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=2, padding='valid')

    def forward(self, x):
        # + self.downsample(x) #! Don't know how to match dimensions
        out = self.block(x)
        return out


class ResNet(nn.Module):
    """ 
    ResNet model for MNIST dataset containig 3 big blocks of increasing channel size (32, 64, 128)
    """

    def __init__(self, n_layers_per_block=4):

        super().__init__()
        self.layer1 = conv3x3(in_channels=1, out_channels=32, stride=1)
        self.block1 = nn.ModuleList([BasicBlock(32)
                                    for i in range(n_layers_per_block)])
        self.down1 = DownsamplingBlock(32)
        self.block2 = nn.ModuleList([BasicBlock(64)
                                    for i in range(n_layers_per_block)])
        self.down2 = DownsamplingBlock(64)
        self.block3 = nn.ModuleList([BasicBlock(128)
                                    for i in range(n_layers_per_block)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128, 10)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def block_forward(self, block, x):
        for layer in block:
            x = layer(x) + x
        return x

    def forward(self, x):
        x = self.layer1(x)
        x = self.block_forward(self.block1, x)
        x = self.down1(x)
        x = self.block_forward(self.block2, x)
        x = self.down2(x)
        x = self.block_forward(self.block3, x)
        x = self.pool(x)
        x = x.view(-1, 128)
        out = self.linear(x)
        return out


def resnet22():
    """ 
    Create a ResNet model of 22 layers
    """
    return ResNet()


def resnet30():
    """ 
    Create a ResNet model of 30 layers
    """
    return ResNet(6)


def load_pretrained_model(model: nn.Module, savePath):
    """ 
    Load parameters from a pretrained model to be used for inference. It needs the model to be
    updated as input.
    """
    if torch.cuda.is_available():
        map_location = torch.device('cuda')
    else:
        map_location = torch.device('cpu')
    state_dict = torch.load(savePath, map_location=map_location)
    model.load_state_dict(state_dict)
    model.eval()
    return model
