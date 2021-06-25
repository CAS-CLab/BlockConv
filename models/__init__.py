import torch
import torchvision.models as torch_models
import logging
from . import cifar10_models
from . import imagenet_models
msglogger = logging.getLogger()

# support model name: ['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'inception_v3', 
#                     'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'squeezenet1_0', 'squeezenet1_1',
#                      'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
IMAGENET_MODEL_NAMES = sorted(name for name in torch_models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(torch_models.__dict__[name]))
EXTRA_PRETRAINED_MODELS = []

def create_model(pretrained, dataset, arch, parallel= True, device_ids=None):
    model = None

    if dataset == 'imagenet':
        if arch in IMAGENET_MODEL_NAMES:
            model = torch_models.__dict__[arch](pretrained=pretrained)
        elif arch in EXTRA_PRETRAINED_MODELS:
            model = imagenet_models.__dict__[arch]()
        else:
            raise ValueError("Model {} is not supported for dataset ImageNet".format(arch))
        msglogger.info("=> using {p} {a} model for imagenet".format(a=arch, p=('pretrained' if pretrained else ''))) 

    elif dataset == 'cifar10':
        if pretrained:
            raise ValueError("Model {} (CIFAR10) does not have a pretrained model".format(arch))
        try:
            model = cifar10_models.__dict__[arch]()
        except KeyError:
            raise ValueError("Model {} is not supported for dataset CIFAR10".format(arch))
        msglogger.info("=> creating {} model for CIFAR10".format(arch))

    else:
        raise ValueError("Could not recognize dataset {}".format(dataset))

    if torch.cuda.is_available() and device_ids != -1:
        device = 'cuda'
        if (arch.startswith('alexnet') or arch.startswith('vgg')) and parallel:
            model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
        elif parallel:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        device = 'cpu'

    return model.to(device)

