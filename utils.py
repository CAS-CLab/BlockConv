import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import Sampler
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np
import math
import shutil
import logging
import os
from numbers import Number
from tabulate import tabulate
from errno import ENOENT
import json

msglogger = logging.getLogger()

DATASETS_NAMES = ['imagenet', 'cifar10']

def get_contents_table(d):
    def inspect_val(val):
        if isinstance(val, (Number, str)):
            return val
        elif isinstance(val, type):
            return val.__name__
        return None

    contents = [[k, type(d[k]).__name__, inspect_val(d[k])] for k in d.keys()]
    contents = sorted(contents, key=lambda entry: entry[0])
    return tabulate(contents, headers=["key", "Type", "Value"], tablefmt="fancy_grid")


def save_checkpoint(epoch, arch, model, optimizer=None, scheduler=None,
                    extras=None, is_best=False, name=None, dir='.'):
    if not os.path.isdir(dir):
        raise IOError(ENOENT, 'checkpoint directory does not exist at {}'.format(os.path.abspath(dir)))

    if extras is None:
        extras = {}
    if not isinstance(extras, dict):
        raise TypeError('extras must be either a dict or None')
    
    if name is None:
        # filename = 'ckpt_{}.pth.tar'.format(epoch)
        filename = 'ckpt.pth.tar'
    else:
        # filename = '{}_ckpt_{}.pth.tar'.format(name, epoch)
        filename = '{}_ckpt.pth.tar'.format(name)
    fullpath = os.path.join(dir, filename)
    msglogger.info("Saving checkpoint to: {}\n".format(fullpath))
    filename_best = 'best.pth.tar' if name is None else name + '_best.pth.tar'
    fullpath_best = os.path.join(dir, filename_best)

    checkpoint = {}
    checkpoint['epoch'] = epoch
    checkpoint['arch'] = arch
    checkpoint['state_dict'] = model.state_dict()
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['optimizer_type'] = type(optimizer)
    checkpoint['extras'] = extras

    torch.save(checkpoint, fullpath)
    if is_best:
        shutil.copyfile(fullpath, fullpath_best)


def load_checkpoint(model, chkpt_file, optimizer=None, model_device=None,  lean_checkpoint=False):
    if not os.path.isfile(chkpt_file):
        raise IOError(ENOENT, 'Could not find a checkpoint file at', chkpt_file)
   
    msglogger.info('=> loading checkpoint {}'.format(chkpt_file))
    checkpoint = torch.load(chkpt_file, map_location=lambda storage, lco: storage) #gpu->cpu

    msglogger.info('=> Checkpoint contents:\n{}\n'.format(get_contents_table(checkpoint)))
    extras = None 
    if 'extras' in checkpoint:
        msglogger.info("=> Checkpoint['extras'] contents:\n{}\n".format(get_contents_table(checkpoint['extras'])))
        extras = checkpoint['extras']
    if 'state_dict' not in checkpoint:
        raise ValueError("Checkpoint must contain the model parameters under the key 'state_dict'")

    checkpoint_epoch = checkpoint.get('epoch', None)
    start_epoch = checkpoint_epoch + 1 if checkpoint_epoch is not None else 0

    model.load_state_dict(checkpoint['state_dict'])
    if model_device is not None:
        model.to(model_device)

    def _load_optimizer(cls, src_state_dict, model):
        dest_optimizer = cls(model.parameters(), lr=1)
        dest_optimizer.load_state_dict(src_state_dict)
        return dest_optimizer

    try:
        optimizer = _load_optimizer(checkpoint['optimizer_type'],
                checkpoint['optimizer_state_dict'], model)
    except KeyError:
        msglogger.info('Not found optimizer in checkpoint')

    if optimizer is not None:
        msglogger.info('Optimizer of type {type} was loaded from checkpoint'.format(type=type(optimizer)))
        msglogger.info('Optimizer Args: {}'.format(dict((k,v) for k,v in optimizer.state_dict()['param_groups'][0].items() if k != 'params')))
    else:
        msglogger.warning('Optimizer could not be loaded from checkpoint.')

    msglogger.info("=> loaded checkpoint '{f}' (epoch {e}".format(f=str(chkpt_file), e=checkpoint_epoch))
    
    return (model, optimizer, start_epoch, extras)
        

def load_data(dataset, data_dir, batch_size, workers, validation_split=0.1, deterministic=False, effective_train_size=1., effective_valid_size=1., effective_test_size=1.):
    if dataset not in DATASETS_NAMES:
        raise ValueError('load_data does not support dataset {}'.format(dataset))
    if dataset == 'cifar10':
        datasets_fn = cifar10_get_datasets
    elif dataset == 'imagenet':
        datasets_fn = imagenet_get_datasets
    return get_data_loaders(datasets_fn, data_dir, batch_size, workers, validation_split, deterministic=deterministic, effective_train_size=effective_train_size,
                            effective_valid_size=effective_valid_size, effective_test_size=effective_test_size)


def cifar10_get_datasets(data_dir):
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    return train_dataset, test_dataset


def imagenet_get_datasets(data_dir):
     """
     Load the ImageNet dataset.
     """
     train_dir = os.path.join(data_dir, 'train')
     test_dir = os.path.join(data_dir, 'val')
     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

     train_transform = transforms.Compose([
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize,
     ])

     train_dataset = datasets.ImageFolder(train_dir, train_transform)

     test_transform = transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         normalize,
     ])

     test_dataset = datasets.ImageFolder(test_dir, test_transform)

     return train_dataset, test_dataset


def get_data_loaders(datasets_fn, data_dir, batch_size, num_workers, validation_split=0.1, deterministic=False, 
                    effective_train_size=1., effective_valid_size=1., effective_test_size=1.):
    train_dataset, test_dataset = datasets_fn(data_dir)
    
    worker_init_fn = __deterministic_worker_init_fn if deterministic else None

    num_train = len(train_dataset)
    indices = list(range(num_train))

    np.random.shuffle(indices)
    valid_indices, train_indices = __split_list(indices, validation_split)

    train_sampler = SwitchingSubsetRandomSampler(train_indices, effective_train_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=True,
                                               worker_init_fn=worker_init_fn)

    valid_loader = None
    if valid_indices:
        valid_sampler = SwitchingSubsetRandomSampler(valid_indices, effective_valid_size)
        valid_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=batch_size, sampler=valid_sampler,
                                                   num_workers=num_workers, pin_memory=True,
                                                   worker_init_fn=worker_init_fn)
    test_indices = list(range(len(test_dataset)))
    test_sampler = SwitchingSubsetRandomSampler(test_indices, effective_test_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=batch_size, sampler=test_sampler,
                                              num_workers=num_workers, pin_memory=True)

    input_shape = __image_size(train_dataset)

    return train_loader, valid_loader or test_loader, test_loader, input_shape

def __image_size(dataset):
    return dataset[0][0].unsqueeze(0).size()

def __deterministic_worker_init_fn(worker_id, seed=0):
    import random
    import numpy
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

def __split_list(l, ratio):
    split_idx = int(np.floor(ratio * len(l)))
    return l[:split_idx], l[split_idx:]

class SwitchingSubsetRandomSampler(Sampler):
    def  __init__(self, data_source, subset_size):
        if subset_size <= 0 or subset_size > 1:
            raise ValueError('subset_size must be in (0..1]')
        self.data_source = data_source
        self.subset_length = int(np.floor(len(self.data_source) * subset_size))

    def __iter__(self):
        indices = torch.randperm(len(self.data_source))
        subset_indices = indices[:self.subset_length]
        return (self.data_source[i] for i in subset_indices)

    def __len__(self):
        return self.subset_length


class CosineLR(_LRScheduler):
    def __init__(self, optimzier, max_epochs, last_epoch=-1, verbose=False):
        self.max_epochs = max_epochs
        super(CosineLR, self).__init__(optimzier, last_epoch, verbose)

    def get_lr(self):
        return [0.5 * base_lr * (1 + math.cos(math.pi * 
                float(self.last_epoch) / float(self.max_epochs))) 
                for base_lr in self.base_lrs]


