#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from math import floor
import torch.nn as nn
from ..utils import *

__all__ = ['block_mobilenet', 'block_mobilenet_025', 'block_mobilenet_050', 'block_mobilenet_075']


class BlockMobileNet(nn.Module):
    def __init__(self, channel_multiplier=1.0, min_channels=8, block_size=0, type=0):
        super(BlockMobileNet, self).__init__()

        if channel_multiplier <= 0:
            raise ValueError('channel_multiplier must be >= 0')

        def block_conv_bn_relu(n_ifm, n_ofm, kernel_size, block_size=0, type=0, stride=1, padding=0, groups=1):
            return [
                BlockConv2d(n_ifm, n_ofm, kernel_size, block_size, type, stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(n_ofm),
                nn.ReLU(inplace=True)
            ]
        def conv_bn_relu(n_ifm, n_ofm, kernel_size, stride=1, padding=0, groups=1):
            return [
                nn.Conv2d(n_ifm, n_ofm, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(n_ofm),
                nn.ReLU(inplace=True)
            ]            

        def depthwise_conv(n_ifm, n_ofm, block_size, type, stride):
            return nn.Sequential(
                *block_conv_bn_relu(n_ifm, n_ifm, 3, block_size, type, stride=stride, padding=1, groups=n_ifm),
                *conv_bn_relu(n_ifm, n_ofm, 1, stride=1)
            )

        base_channels = [32, 64, 128, 256, 512, 1024]
        self.channels = [max(floor(n * channel_multiplier), min_channels) for n in base_channels]

        self.model = nn.Sequential(
            nn.Sequential(*block_conv_bn_relu(3, self.channels[0], 3, block_size, type, stride=2, padding=1)),
            depthwise_conv(self.channels[0], self.channels[1], block_size, type, 1),
            depthwise_conv(self.channels[1], self.channels[2], block_size, type, 2),
            depthwise_conv(self.channels[2], self.channels[2], block_size, type, 1),
            depthwise_conv(self.channels[2], self.channels[3], block_size, type, 2),
            depthwise_conv(self.channels[3], self.channels[3], block_size, type, 1),
            depthwise_conv(self.channels[3], self.channels[4], block_size, type, 2),
            depthwise_conv(self.channels[4], self.channels[4], block_size, type, 1),
            depthwise_conv(self.channels[4], self.channels[4], block_size, type, 1),
            depthwise_conv(self.channels[4], self.channels[4], block_size, type, 1),
            depthwise_conv(self.channels[4], self.channels[4], block_size, type, 1),
            depthwise_conv(self.channels[4], self.channels[4], block_size, type, 1),
            depthwise_conv(self.channels[4], self.channels[5], block_size, type, 2),
            depthwise_conv(self.channels[5], self.channels[5], block_size, type, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(self.channels[5], 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.channels[-1])
        x = self.fc(x)
        return x


def block_mobilenet_025():
    return BlockMobileNet(channel_multiplier=0.25, block_size=0)


def block_mobilenet_050():
    return BlockMobileNet(channel_multiplier=0.5, block_size=0)


def block_mobilenet_075():
    return BlockMobileNet(channel_multiplier=0.75, block_size=0)


def block_mobilenet():
    return BlockMobileNet(block_size=(28,28), type=0)
