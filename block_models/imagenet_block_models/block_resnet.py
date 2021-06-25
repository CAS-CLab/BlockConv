import torch
import torch.nn as nn
from ..utils import *

__all__ = ['BlockResNet', 'block_resnet18', 'block_resnet34', 'block_resnet50', 'block_resnet101',
           'block_resnet152', 'block_resnext50_32x4d', 'block_resnext101_32x8d']

USE_BIAS = False


def block_conv3x3(in_planes, out_planes, block_size=0, type=0, stride=1, groups=1, dilation=1):
    """3x3 block convolution with padding"""
    return BlockConv2d(in_planes, out_planes, kernel_size=3, block_size=block_size, type=type, stride=stride,
                     padding=dilation, groups=groups, bias=USE_BIAS, dilation=dilation)


def block_conv1x1(in_planes, out_planes, block_size=0, type=0, stride=1):
    """1x1 block convolution"""
    return BlockConv2d(in_planes, out_planes, kernel_size=1, block_size=block_size, type=type, 
                    stride=stride, bias=USE_BIAS)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BlockBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, block_size=0, type=0, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BlockBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        #if stride == 1:
        self.conv1 = block_conv3x3(inplanes, planes, block_size, type, stride)
       # else:
        #    self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = block_conv3x3(planes, planes, block_size, type)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample[0](x)
            identity = self.downsample[1](x)
           # identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class BlockBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, block_size=0, type=0, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BlockBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = block_conv1x1(inplanes, width, block_size, type=type)
        self.bn1 = norm_layer(width)
        self.conv2 = block_conv3x3(width, width, block_size, type, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = block_conv1x1(width, planes * self.expansion, block_size, type)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            x = self.downsample[0](x)
            identity = self.downsample[1](x)
        
        out += identity
        out = self.relu(out)

        return out


class BlockResNet(nn.Module):

    def __init__(self, block, layers, block_size=0, type=0, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(BlockResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        ## CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = BlockConv2d(3, self.inplanes, kernel_size=7, block_size=block_size, type=type, stride=2, padding=3, bias=USE_BIAS)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            #    bias=False)
        ## END
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], block_size=block_size, type=type)
        self.layer2 = self._make_layer(block, 128, layers[1],block_size=block_size, type=type, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], block_size=block_size, type=type, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], block_size=block_size, type=type, stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, BlockConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BlockBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BlockBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, block_size=0, type=0, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                block_conv1x1(self.inplanes, planes * block.expansion, block_size, type, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, block_size, type, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, block_size, type, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
            
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _block_resnet(arch, block, layers, pretrained, progress, device, block_size, type=0, **kwargs):
    model = BlockResNet(block, layers, block_size, type, **kwargs)
    return model


def block_resnet18(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _block_resnet('block_resnet18', BlockBasicBlock, [2, 2, 2, 2], pretrained, progress, device,
                  28, 0, **kwargs)


def block_resnet34(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _block_resnet('block_resnet34', BlockBasicBlock, [3, 4, 6, 3], pretrained, progress, device,
                  4,  **kwargs)


def block_resnet50(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _block_resnet('block_resnet50', BlockBottleneck, [3, 4, 6, 3], pretrained, progress, device,
                  56, 0, **kwargs)


def block_resnet101(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _block_resnet('block_resnet101', BlockBottleneck, [3, 4, 23, 3], pretrained, progress, device,
                 4,   **kwargs)


def block_resnet152(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _block_resnet('block_resnet152', BlockBottleneck, [3, 8, 36, 3], pretrained, progress, device,
                  4,  **kwargs)


def block_resnext50_32x4d(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _block_resnet('block_resnext50_32x4d', BlockBottleneck, [3, 4, 6, 3],
                  0,  pretrained, progress, device, **kwargs)


def block_resnext101_32x8d(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _block_resnet('block_resnext101_32x8d', BlockBottleneck, [3, 4, 23, 3],
                  0,  pretrained, progress, device, **kwargs)
