"""VGG for CIFAR10

VGG for CIFAR10, based on "Very Deep Convolutional Networks for Large-Scale
Image Recognition".
This is based on TorchVision's implementation of VGG for ImageNet, with
appropriate changes for the 10-class Cifar-10 dataset.
We replaced the three linear classifiers with a single one.
"""
import torch
import torch.nn as nn
from ..utils import *
__all__ = [
    'Block_VGG', 'block_vgg11', 'block_vgg11_bn', 'block_vgg13', 'block_vgg13_bn', 'block_vgg16',
     'block_vgg16_bn', 'block_vgg19', 'block_vgg19_bn',
]
TYPE = 0
BS = 28
class Block_VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(Block_VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, BlockConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            block_conv2d = BlockConv2d(in_channels, v[0], kernel_size=3, block_size=v[1], type=TYPE, padding=1)
            if batch_norm:
                layers += [block_conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [block_conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)


cfg = {
    'A': [[64,0], ['M'], [128,0], ['M'], [256,0], [256,0], ['M'], [512,0], [512,0], ['M'], [512,0], [512,0], ['M']],
    'B': [[64,7], [64,0], ['M'], [128,0], [128,0], ['M'], [256,0], [256,0], ['M'], [512,0], [512,0], ['M'], [512,0], [512,0], ['M']],
    'D': [[64,BS], [64,BS], ['M'], [128,BS], [128,BS], ['M'], [256,BS], [256,BS], [256,BS], ['M'], [512,BS], [512,BS], [512,BS], ['M'], [512,BS], [512,BS], [512,BS], ['M']],
    'E': [[64,0], [64,0], ['M'], [128,0], [128,0], ['M'], [256,0], [256,0], [256,0], [256,0], ['M'], [512,0], [512,0], [512,0], [512,0], ['M'], [512,0], [512,0], [512,0], [512,0], ['M']],
}


def block_vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")"""
    model = Block_VGG(make_layers(cfg['A']), **kwargs)
    return model


def block_vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = Block_VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def block_vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")"""
    model = Block_VGG(make_layers(cfg['B']), **kwargs)
    return model


def block_vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = Block_VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def block_vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")
    """
    model = Block_VGG(make_layers(cfg['D']), **kwargs)
    return model


def block_vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = Block_VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def block_vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")
    """
    model = Block_VGG(make_layers(cfg['E']), **kwargs)
    return model


def block_vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = Block_VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model
