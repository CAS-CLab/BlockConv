import argparse
import operator

def get_parser():
    parser = argparse.ArgumentParser(description="Simple code for train and test on ImageNet and Cifar")
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', help='model architecture'+'(default: resnet18)')
    parser.add_argument('--workers', '-j', metavar='N', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', metavar='EPOCH', type=int, default=360,  help='number of total epochs to run (default: 2)')
    parser.add_argument('--batch_size', '-b', metavar='BATCH_SIZE', type=int, default=512, help='mini-batch size (default: 256)')
    parser.add_argument('--print_freq', '-p', metavar='N', type=int, default=10, help='print frequence (default: 10)')
    parser.add_argument('--gpus', metavar='DEV_ID', default=None, help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
    parser.add_argument('--cpu', action='store_true', default=False, help='Use CPU only.\n'
                                                                    'Flag not set => uses GPUs according to the --gpus flag value.'
                                                                    'Flag set => overrides the --gpus flag')

    parser.add_argument('--do_eval', action='store_true',  help='evaluate model')
    parser.add_argument('--do_train', action='store_true',  help='train model')
    parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
    parser.add_argument('--out_dir', '-o', dest='output_dir', default='logs/resnet18', help='Path to dump logs and checkpoints')
    parser.add_argument('--dataset', dest='dataset', type=str, default='cifar10', help='dataset used to train (default: cifar10)')
    parser.add_argument('--deterministic', '--det', action='store_true', help='Ensure deterministic execution for re-producible results.')
   
    parser.add_argument('--validation-split', '--valid-size', '--vs', dest='validation_split',
                                    type=float_range(exc_max=True), default=0.,  help='Portion of training dataset to set aside for validation (default: 0.0)')
    parser.add_argument('--effective-train-size', '--etrs', type=float_range(exc_min=True), default=1.,
                        help='Portion of training dataset to be used in each epoch. '
                             'NOTE: If --validation-split is set, then the value of this argument is applied '
                             'AFTER the train-validation split according to that argument')
    parser.add_argument('--effective-valid-size', '--evs', type=float_range(exc_min=True), default=1.,
                        help='Portion of validation dataset to be used in each epoch. '
                             'NOTE: If --validation-split is set, then the value of this argument is applied '
                             'AFTER the train-validation split according to that argument')
    parser.add_argument('--effective-test-size', '--etes', type=float_range(exc_min=True), default=1.,
                        help='Portion of test dataset to be used in each epoch')

    parser.add_argument('--disable_tqdm', action='store_true',  help='disable tqdm')
    parser.add_argument('--block_size', default=None, help='block size')
    parser.add_argument('--type', default=None, type=int, help='type of block size ( 0 or 1 )')
    parser.add_argument('--padding_mode', default=None, help='padding mode ("constant", "replicate", "reflect")')

    optimizer_args = parser.add_argument_group('Optimizer arguments')
    optimizer_args.add_argument('--learning_rate', '--lr', metavar='LR', type=float, default=0.1, help='initial learning rate (default: 0.1)')
    optimizer_args.add_argument('--momentum', metavar='M', type=float, default=0.9, help='momentum (default: 0.9)')
    optimizer_args.add_argument('--weight_decay', '--wd', metavar='W', type=float, default=5e-4, help='weight decay (default: 1e-4)')
    optimizer_args.add_argument('--milestones', '--ms', default=None, help='Milestones for MultiStepLR')

    load_checkpoint_group = parser.add_argument_group('Resuming arguments')
    load_checkpoint_group_exc = load_checkpoint_group.add_mutually_exclusive_group()
    load_checkpoint_group_exc.add_argument('--resume_from', dest='resumed_checkpoint_path', default='', type=str, metavar='PATH',
                                              help='path to latest checkpoint. Use to resume paused training session.')
    load_checkpoint_group.add_argument('--reset_optimizer', action='store_true', help='Flag to override optimizer if resumed from checkpoint. This will reset epochs count.')
    
    return parser

def float_range(min_val=0., max_val=1., exc_min=False, exc_max=False):
    def checker(val_str):
        val = float(val_str)
        min_op, min_op_str = (operator.gt, '>') if exc_min else (operator.ge, '>=')
        max_op, max_op_str = (operator.lt, '<') if exc_max else (operator.le, '<=')
        if min_op(val, min_val) and max_op(val, max_val):
            return val
        else:
            raise ValueError('Value must be {} {} and {} {} (received {})'.format(min_op_str, min_val, max_op_str, max_val, val))

    if min_val >= max_val:
        raise ValueError('min_val must be less than max_val')
    return checker

