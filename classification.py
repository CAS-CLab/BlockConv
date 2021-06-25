import os
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torchnet.meter as meter

from models import create_model 
from block_models import create_block_model
import utils 
import myparser
from msglogging import config_logger, log_execution_env_state

try: 
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

def train(args, model, train_loader, eval_loader, optimizer, logger, tb_writer):
    if args.reset_optimizer:
        args.start_epoch = 0
        if optimizer is not None:
            optimizer = None
            logger.info('\nreset_optimizer flag set: Overriding resumed optimizer and resetting epoch count to 0')

    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, 
                        momentum = args.momentum, weight_decay = args.weight_decay)
        logger.info('Optimzer Type: %s', type(optimizer))
        logger.info('Optimzer Args: %s\n', optimizer.defaults)

    criterion = nn.CrossEntropyLoss().to(args.device)

    if "mobilenet" in args.arch:
        scheduler = utils.CosineLR(optimizer, args.epochs, last_epoch=args.start_epoch-1)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                        milestones=[int(s) for s in args.milestones.split(',')], 
                        gamma=0.1, last_epoch=args.start_epoch-1)

    if args.start_epoch >= args.epochs:
        logger.error('epoch count is too low, starting epoch is {} but total epochs set to {}'.format(args.start_epoch, args.epochs))
        raise ValueError('Epochs parameter is too low. Nothing to do.')

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    logger.info("{} samples ({} per mini-batch)".format(total_samples, batch_size))

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        total_loss = meter.AverageValueMeter()
        classerr = meter.ClassErrorMeter(accuracy=True, topk=(1,5))

        for train_step, (inputs, target) in enumerate(train_loader):
            inputs, target = inputs.to(args.device), target.to(args.device)
            output = model(inputs)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            classerr.add(output.data, target)
            total_loss.add(loss.item())

            steps_completed = train_step + 1
            if steps_completed % args.print_freq == 0:
                top1, top5 =  classerr.value(1), classerr.value(5)
                
                logger.info('Train epoch: %d [%5d/%5d]  Top1: %.3f  Top5: %.3f  Loss: %.3f LR: %f',
                        epoch, steps_completed, steps_per_epoch,  top1, top5, total_loss.mean, scheduler.get_last_lr()[0])
                global_step = epoch * steps_per_epoch + steps_completed
                tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalar('train_loss', total_loss.mean, global_step)

        top1, top5, vloss = evaluate(args, model, eval_loader, logger)
        logger.info('==> Validation: Top1: %.3f  Top5: %.3f  Loss: %.3f',
            top1, top5, loss)
        tb_writer.add_scalar('eval_top1', top1, epoch)
        tb_writer.add_scalar('eval_top5', top5, epoch)

        is_best = top1 > args.best_top1
        args.best_top1 = top1 if is_best else args.best_top1
        args.best_epoch = epoch if is_best else args.best_epoch
        checkpoint_extras = {'current_top1': top1,
                            'best_top1': args.best_top1,
                            'best_epoch': args.best_epoch}
        logger.info('==> best epoch: %d best_top1: %.3f', args.best_epoch, args.best_top1)
        utils.save_checkpoint(epoch, args.arch, model, optimizer=optimizer, extras=checkpoint_extras, 
                            is_best=is_best, name=args.name, dir=logger.logdir)

        scheduler.step()


def evaluate(args, model, dataloader, logger):
    logger.info('----------evaluation---------')
    total_loss = meter.AverageValueMeter()
    classerr = meter.ClassErrorMeter(accuracy=True, topk=(1,5))

    total_samples = len(dataloader.sampler)
    batch_size = dataloader.batch_size
    logger.info("{} samples ({} per mini-batch)".format(total_samples, batch_size))

    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    model.eval()

    eval_iterator = tqdm(dataloader, desc="Iteration", disable=args.disable_tqdm, ncols=160)
    for _, (inputs, target) in enumerate(eval_iterator):
        with torch.no_grad():
            inputs, target = inputs.to(args.device), target.to(args.device)
            output = model(inputs)

            loss = criterion(output, target)
            total_loss.add(loss.item())
            classerr.add(output.data, target)

    return classerr.value(1), classerr.value(5), total_loss.mean

def main():
    args = myparser.get_parser().parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    script_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(script_dir)

    msglogger, _ = config_logger(os.path.join(script_dir, 'logging.conf'), 
                                    args.name, args.output_dir)
    
    # Choose CPUs or GPUs
    if args.cpu or not torch.cuda.is_available():
        args.device = 'cpu'
        args.gpus = -1
    else:
        args.device = 'cuda'
        if args.gpus is not None:
            try:
                args.gpus = [int(s) for s in args.gpus.split(',')]
            except ValueError:
                raise ValueError('ERROR: Argument --gpus must be a comma-separated list of integers only')
            available_gpus = torch.cuda.device_count()
            for dev_id in args.gpus:
                if dev_id >= available_gpus:
                    raise ValueError('ERROR: GPU device ID {0} requested, but only {1} devices available'.format(dev_id, available_gpus))
            torch.cuda.set_device(args.gpus[0])
    
    log_execution_env_state(msglogger.logdir, gitroot=module_path)
    
    # Create model
    if args.arch.startswith('block'):
        model = create_block_model(args, args.dataset, args.arch, parallel = True, device_ids = args.gpus)
    else:
        model = create_model(args.pretrained, args.dataset, args.arch, parallel = True, device_ids = args.gpus)
    msglogger.info(model)

    # Prepare dataloader
    train_loader, val_loader, test_loader, _ = utils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.validation_split, args.deterministic, 
        args.effective_train_size, args.effective_valid_size, args.effective_test_size)
    msglogger.info('Dataset sizes:\n\ttraining={}\n\tvalidation={}\n\ttest={}'.format(
                len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler)))    
    
    args.start_epoch = 0
    args.best_top1 = 0
    args.best_epoch = 0
    optimizer = None
    if args.resumed_checkpoint_path:
        model, optimizer, args.start_epoch, extras = \
            utils.load_checkpoint(model, args.resumed_checkpoint_path, None, model_device=args.device)
        if not extras is None:
            args.best_top1 = extras.get('best_top1', 0)
            args.best_epoch = extras.get('best_epoch', 0)

    if args.do_train:
        tb_writer = SummaryWriter(log_dir=msglogger.logdir)
        train(args, model, train_loader, val_loader, optimizer, msglogger, tb_writer)
        
    if args.do_eval:
        top1, top5, loss = evaluate(args, model, test_loader, msglogger)
        msglogger.info('==> Validation: Top1: %.3f  Top5: %.3f  Loss: %.3f',
                    top1, top5, loss)

if __name__ == '__main__':
    main()