import torch
import torch.nn as nn
from ..utils import *

__all__ = [
    'BlockVGG', 'block_vgg11', 'block_vgg11_bn', 'block_vgg13', 'block_vgg13_bn', 'block_vgg16', 'block_vgg16_bn',
    'block_vgg19_bn', 'block_vgg19',
]

class BlockVGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(BlockVGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
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

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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
                nn.init.constant_(m.bias, 0)
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
            conv2d = BlockConv2d(in_channels, v[0], kernel_size=3, block_size=v[1], padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)


cfgs = {
    'A': [[64,0], ['M'], [128,0], ['M'], [256,0], [256,0], ['M'], [512,0], [512,0], ['M'], [512,0], [512,0], ['M']],
    'B': [[64,7], [64,0], ['M'], [128,0], [128,0], ['M'], [256,0], [256,0], ['M'], [512,0], [512,0], ['M'], [512,0], [512,0], ['M']],
    'D': [[64,0], [64,4], ['M'], [128,4], [128,4], ['M'], [256,4], [256,4], [256,4], ['M'], [512,4], [512,4], [512,4], ['M'], [512,4], [512,4], [512,4], ['M']],
  #  'D': [[64,0], [64,0], ['M'], [128,0], [128,0], ['M'], [256,0], [256,0], [256,0], ['M'], [512,0], [512,0], [512,0], ['M'], [512,0], [512,0], [512,0], ['M']],
    'E': [[64,0], [64,0], ['M'], [128,0], [128,0], ['M'], [256,0], [256,0], [256,0], [256,0], ['M'], [512,0], [512,0], [512,0], [512,0], ['M'], [512,0], [512,0], [512,0], [512,0], ['M     ']],
}



def _block_vgg(arch, cfg, batch_norm, pretrained, progress, device='cpu', **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = BlockVGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        arch = arch.replace('block_', '')
        state_dict = torch.load('cifar10_state_dicts/'+arch+'.pt', map_location=device)
        model.load_state_dict(state_dict)
    return model


def block_vgg11(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _block_vgg('block_vgg11', 'A', False, pretrained, progress, **kwargs)


def block_vgg11_bn(pretrained=False, progress=True, device='cpu', **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _block_vgg('block_vgg11_bn', 'A', True, pretrained, progress, device, **kwargs)


def block_vgg13(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _block_vgg('block_vgg13', 'B', False, pretrained, progress, **kwargs)


def block_vgg13_bn(pretrained=False, progress=True, device='cpu', **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _block_vgg('block_vgg13_bn', 'B', True, pretrained, progress, device, **kwargs)


def block_vgg16(pretrained=False, progress=True, device='cpu', **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _block_vgg('block_vgg16', 'D', False, pretrained, progress, device, **kwargs)


def block_vgg16_bn(pretrained=False, progress=True, device='cpu', **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _block_vgg('block_vgg16_bn', 'D', True, pretrained, progress, device, **kwargs)


def block_vgg19(pretrained=False, progress=True, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _block_vgg('block_vgg19', 'E', False, pretrained, progress, **kwargs)


def block_vgg19_bn(pretrained=False, progress=True, device='cpu', **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _block_vgg('block_vgg19_bn', 'E', True, pretrained, progress, device, **kwargs)
