import torch
import torchvision.models as torch_models
import logging
from . import imagenet_block_models
from . import cifar10_block_models
from .utils import BlockConv2d

msglogger = logging.getLogger()

def create_block_model(args, dataset, arch, parallel= True, device_ids=None):
    model = None
    if dataset == 'imagenet':
        try:
            model = imagenet_block_models.__dict__[arch]()              
        except KeyError:
            raise ValueError("Model {} is not supported for dataset ImageNet".format(arch))
        msglogger.info("=> creating {} model for imagenet".format(arch))

    elif dataset == 'cifar10':
        try:
            model = cifar10_block_models.__dict__[arch]()                  
        except KeyError:
            raise ValueError("Model {} is not supported for dataset Cifar10".format(arch))
        msglogger.info("=> creating {} model for Cifar10".format(arch))
    else:
        raise ValueError("Could not recognize dataset {}".format(dataset))
                
    for module in model.modules():
        if isinstance(module, BlockConv2d):
            if not args.block_size is None:
                module.block_size = tuple([int(s) for s in args.block_size.split(',')])
            if not args.type is None:
                module.set_type(args.type)
            if not args.padding_mode is None:
                module.padding_mode = args.padding_mode

    if torch.cuda.is_available() and device_ids != -1:
        device = 'cuda'
        if (arch.startswith('alexnet') or arch.startswith('vgg')) and parallel:
            model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
        elif parallel:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        device = 'cpu'

    return model.to(device)

