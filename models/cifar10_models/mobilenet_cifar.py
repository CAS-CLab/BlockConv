import torch.nn as nn
from math import floor

__all__ = ['mobilenet_cifar', 'mobilenet_025_cifar', 'mobilenet_050_cifar', 'mobilenet_075_cifar']

class MobileNet_Cifar(nn.Module):
    def __init__(self, channel_multiplier=1.0, min_channels=8):
        super(MobileNet_Cifar, self).__init__()

        if channel_multiplier <= 0:
            raise ValueError('channel multiplier must be >= 0')

        def conv_bn_relu(in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1):
            return[
                nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True)
            ]
        
        def depthwise_conv(in_planes, out_planes, stride):
            return nn.Sequential(
                    *conv_bn_relu(in_planes, in_planes, 3, stride=stride, padding=1, groups=in_planes),
                    *conv_bn_relu(in_planes, out_planes, 1, stride=1) 
            )

        base_channels = [32, 64, 128, 256, 512, 1024]
        self.channels =[max(floor(n * channel_multiplier), min_channels) for n in base_channels]

        self.features = nn.Sequential(
             nn.Sequential(*conv_bn_relu(3, self.channels[0], 3, stride=2, padding=1)),
            depthwise_conv(self.channels[0], self.channels[1], 1),
            depthwise_conv(self.channels[1], self.channels[2], 2),
            depthwise_conv(self.channels[2], self.channels[2], 1),
            depthwise_conv(self.channels[2], self.channels[3], 2),
            depthwise_conv(self.channels[3], self.channels[3], 1),
            depthwise_conv(self.channels[3], self.channels[4], 2),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[5], 2),
            depthwise_conv(self.channels[5], self.channels[5], 1),
            nn.AvgPool2d(7), 
        )
        self.fc = nn.Linear(self.channels[5], 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.channels[-1])
        x = self.fc(x)
        return x


def mobilenet_025_cifar():
    return MobileNet_Cifar(channel_multiplier=0.25)

def mobilenet_050_cifar():
    return MobileNet_Cifar(channel_multiplier=0.50)

def mobilenet_075_cifar():
    return MobileNet_Cifar(channel_multiplier=0.75)

def mobilenet_cifar():
    return MobileNet_Cifar(channel_multiplier=1.0)
