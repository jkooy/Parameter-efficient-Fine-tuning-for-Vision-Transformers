from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import logging

from timm.data import Mixup
import math
import numpy as np
import torch
import torch.nn as nn
# from torch.cuda.amp import autocast

from core.evaluate import accuracy
from core.evaluate import multi_classes_accuracy
from core.mixcut import mixcut_criterion
from core.mixcut import mixcut_data
from core.mixup import mixup_criterion
from core.mixup import mixup_data
from utils.comm import comm


def mix_data(x, y, mixup, mixcut, mixup_and_mixcut):
    if mixup > 0.0 and mixcut > 0.0 and mixup_and_mixcut:
        p = np.random.random()
        if p > 0.5:
            (data_generator, l, c) = (mixup_data, mixup, mixup_criterion)
        else:
            (data_generator, l, c) = (mixcut_data, mixcut, mixcut_criterion)
        x, y_a, y_b, lam = data_generator(x, y, l)
    elif mixup > 0.0:
        x, y_a, y_b, lam = mixup_data(x, y, mixup)
        c = mixup_criterion
    elif mixcut > 0.0:
        x, y_a, y_b, lam = mixcut_data(x, y, mixcut)
        c = mixcut_criterion
    else:
        return None, None, None, None, None

    return x, y_a, y_b, lam, c


def train_one_epoch(config, train_loader, model, criterion, optimizer, epoch,
                    output_dir, tb_log_dir, writer_dict, ema_model=None,
                    scaler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    logging.info('=> switch to train mode')
    model.train()

    aug = config.AUG
    mixup_fn = Mixup(
        mixup_alpha=aug.MIXUP, cutmix_alpha=aug.MIXCUT,
        cutmix_minmax=aug.MIXCUT_MINMAX if aug.MIXCUT_MINMAX else None,
        prob=aug.MIXUP_PROB, switch_prob=aug.MIXUP_SWITCH_PROB,
        mode=aug.MIXUP_MODE, label_smoothing=config.LOSS.LABEL_SMOOTHING,
        num_classes=config.MODEL.NUM_CLASSES
    ) if aug.MIXUP_PROB > 0.0 else None
    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        if config.LOSS.LOSS in ['sigmoid', 'focal']:
            y = nn.functional.one_hot(
                y, num_classes=config.MODEL.NUM_CLASSES
            ).float()

        if mixup_fn:
            x, y = mixup_fn(x, y)

        with autocast(enabled=config.AMP.ENABLED):
            if config.AMP.ENABLED and config.AMP.MEMORY_FORMAT == 'nwhc':
                x = x.contiguous(memory_format=torch.channels_last)
                y = y.contiguous(memory_format=torch.channels_last)

            outputs = model(x)
            loss = criterion(outputs, y)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            logging.error(f'=> loss is {loss_value}, stopping training')
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            fname = os.path.join(output_dir, f'{time_str}_states.pth')
            logging.info(f'=> save error state to {fname}')
            torch.save(
                {
                    'x': x,
                    'y': y,
                    'outputs': outputs,
                    'loss': loss,
                    'states': model.module.state_dict()
                },
                fname
            )
            sys.exit(-1)

        # compute gradient and do update step
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') \
            and optimizer.is_second_order

        scaler.scale(loss).backward(create_graph=is_second_order)

        if config.TRAIN.CLIP_GRAD_NORM > 0.0:
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.TRAIN.CLIP_GRAD_NORM
            )

        scaler.step(optimizer)
        scaler.update()
        if ema_model:
            ema_model(model)

        # measure accuracy and record loss
        losses.update(loss.item(), x.size(0))

        if config.LOSS.LOSS in ['sigmoid', 'focal'] or mixup_fn:
            y = torch.argmax(y, dim=1)

        if config.LOSS.LOSS == 'multisoftmax':
            prec1 = multi_classes_accuracy(outputs, y, fast=True)
            prec5 = 0
        else:
            prec1, prec5 = accuracy(outputs, y, (1, 5))

        top1.update(prec1, x.size(0))
        top5.update(prec5, x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = '=> Epoch[{0}][{1}/{2}]: ' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Accuracy@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time,
                      speed=x.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1, top5=top5)
            logging.info(msg)

        torch.cuda.synchronize()

    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', losses.avg, global_steps)
        writer.add_scalar('train_top1', top1.avg, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1


@torch.no_grad()
def test(config, val_loader, model, criterion, output_dir, tb_log_dir,
         writer_dict=None, distributed=False, real_labels=None,
         valid_labels=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    logging.info('=> switch to eval mode')
    model.eval()

    end = time.time()
    for i, (x, y) in enumerate(val_loader):
        # compute output
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        if config.LOSS.LOSS in ['sigmoid', 'focal']:
            num_classes = config.MODEL.NUM_CLASSES \
                if not valid_labels else sum(valid_labels)
            y = nn.functional.one_hot(
                y, num_classes=num_classes
            ).float()

        outputs = model(x)
        if valid_labels:
            outputs = outputs[:, valid_labels]

        loss = criterion(outputs, y)

        if real_labels and not distributed:
            real_labels.add_result(outputs)

        # measure accuracy and record loss
        losses.update(loss.item(), x.size(0))
        if config.LOSS.LOSS in ['sigmoid', 'focal']:
            y = torch.argmax(y, dim=1)

        if config.LOSS.LOSS == 'multisoftmax':
            prec1 = multi_classes_accuracy(outputs, y, fast=False)
            prec5 = 0
        else:
            prec1, prec5 = accuracy(outputs, y, (1, 5))
        top1.update(prec1, x.size(0))
        top5.update(prec5, x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logging.info('=> synchronize...')
    comm.synchronize()
    top1_acc, top5_acc, loss_avg = map(
        _meter_reduce if distributed else lambda x: x.avg,
        [top1, top5, losses]
    )

    if real_labels and not distributed:
        real_top1 = real_labels.get_accuracy(k=1)
        real_top5 = real_labels.get_accuracy(k=5)
        msg = '=> TEST using Reassessed labels:\t' \
            'Error@1 {error1:.3f}%\t' \
            'Error@5 {error5:.3f}%\t' \
            'Accuracy@1 {top1:.3f}%\t' \
            'Accuracy@5 {top5:.3f}%\t'.format(
                top1=real_top1,
                top5=real_top5,
                error1=100-real_top1,
                error5=100-real_top5
            )
        logging.info(msg)

    if comm.is_main_process():
        msg = '=> TEST:\t' \
            'Loss {loss_avg:.4f}\t' \
            'Error@1 {error1:.3f}%\t' \
            'Error@5 {error5:.3f}%\t' \
            'Accuracy@1 {top1:.3f}%\t' \
            'Accuracy@5 {top5:.3f}%\t'.format(
                loss_avg=loss_avg, top1=top1_acc,
                top5=top5_acc, error1=100-top1_acc,
                error5=100-top5_acc
            )
        logging.info(msg)

    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', loss_avg, global_steps)
        writer.add_scalar('valid_top1', top1_acc, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    logging.info('=> switch to train mode')
    model.train()

    return top1_acc


def _meter_reduce(meter):
    rank = comm.local_rank
    meter_sum = torch.FloatTensor([meter.sum]).cuda(rank)
    meter_count = torch.FloatTensor([meter.count]).cuda(rank)
    torch.distributed.all_reduce(meter_sum)
    torch.distributed.all_reduce(meter_count)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
