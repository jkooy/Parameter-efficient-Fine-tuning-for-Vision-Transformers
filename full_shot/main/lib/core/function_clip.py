from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import logging

from timm.data import Mixup
import math
import numpy as np
import torch
from torch.cuda.amp import autocast

from core.evaluate import accuracy
from core.function import AverageMeter
from core.function import _meter_reduce
from dataset.languages import SimpleTokenizer
from dataset.languages import HFPTTokenizer
from dataset.imagenet.constants import IMAGENET_CLASSES
from dataset.imagenet.constants import IMAGENET_DEFAULT_TEMPLATES
from utils.comm import comm


def train_one_epoch(config, train_loader, model, criterion, optimizer, epoch,
                    output_dir, tb_log_dir, writer_dict, ema_model=None,
                    scaler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

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

        if mixup_fn:
            x, y = mixup_fn(x, y)

        # ========deepspeed use its own mixprecision========
        if not config.USE_DEEPSPEED:
            with autocast(enabled=config.AMP.ENABLED):
                if config.AMP.ENABLED and config.AMP.MEMORY_FORMAT == 'nwhc':
                    x = x.contiguous(memory_format=torch.channels_last)
                    y = y.contiguous(memory_format=torch.channels_last)

                logits = model(x, y)
                loss = criterion(logits)
        else:
            logits = model(x, y)
            loss = criterion(logits)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            logging.error(f'=> loss is {loss_value}, stopping training')
            sys.exit(-1)

        # compute gradient and do update step
        if not config.USE_DEEPSPEED:
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
        else:
            model.backward(loss)
            model.step(dict(epoch=epoch))

        if ema_model:
            ema_model(model)

        # measure accuracy and record loss
        losses.update(loss.item(), x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = '=> Epoch[{0}][{1}/{2}]: ' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time,
                      speed=x.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logging.info(msg)

        torch.cuda.synchronize()

    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', losses.avg, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1


@torch.no_grad()
def evaluate_retrieval(config, test_loader, model, output_dir, tb_log_dir):
    logging.info('=> switch to eval mode')
    model.eval()

    features_image = []
    features_text = []
    num_captions_per_img = getattr(config.DATASET, 'NUM_CAPTIONS', 1)
    if 'coco-caption' in config.DATASET.TEST_SET:
        num_captions_per_img = 5
    model_without_ddp = model.module if hasattr(model, 'module') else model
    for i, (x, y) in enumerate(test_loader):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        features_image.append(
            model_without_ddp.encode_image(x).cpu()
        )
        if num_captions_per_img > 1:
            B, N, C = y.shape
            y = y.reshape(B*N, C)
        features_text.append(
            model_without_ddp.encode_text(y).cpu()
        )

    features_image = torch.cat(features_image)
    features_text = torch.cat(features_text)

    i2t_similarities = features_image @ features_text.t()

    i2t_ranks = []
    for i, sim in enumerate(i2t_similarities.cpu().numpy()):
        inds = np.argsort(sim)[::-1]
        for r, ind in enumerate(inds):
            if i*num_captions_per_img <= ind < (i+1)*num_captions_per_img:
                rank = r
                break
        i2t_ranks.append(rank)

    rank = [1, 5]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]

    t2i_similarities = features_text @ features_image.t()

    t2i_ranks = []
    for i, sim in enumerate(t2i_similarities.cpu().numpy()):
        inds = np.argsort(sim)[::-1]
        for r, ind in enumerate(inds):
            if i//num_captions_per_img == ind:
                rank = r
                break
        t2i_ranks.append(rank)

    rank = [1, 5]
    t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]

    logging.info(
        '=> I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5'
        .format(i2t_accs[0], i2t_accs[1])
    )
    logging.info(
        '=> T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5'
        .format(t2i_accs[0], t2i_accs[1])
    )

    return i2t_accs[0], t2i_accs[0]


@torch.no_grad()
def zeroshot_classifier(classnames, templates, tokenizer, model):
    zeroshot_weights = []
    for classname in classnames:
        texts = [template.format(classname) for template in templates]
        texts = tokenizer(texts).cuda()
        class_embeddings = model.encode_text(texts)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

    return zeroshot_weights


@torch.no_grad()
def evaluate_zeroshot(
    config, dataset, test_loader, model, output_dir, tb_log_dir, distributed, tokenizer
):
    logging.info('=> switch to eval mode')
    model_without_ddp = model.module if hasattr(model, 'module') else model
    model_without_ddp.eval()

    classnames = ''
    templates = ''
    if dataset == 'imagenet':
        classnames = IMAGENET_CLASSES
        templates = IMAGENET_DEFAULT_TEMPLATES
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    """
    if config.MODEL.SPEC.TEXT.TOKENIZER == 'clip':
        tokenizer = SimpleTokenizer()
    elif 'hf_' in config.MODEL.SPEC.TEXT.TOKENIZER:
        tokenizer = HFPTTokenizer(config.MODEL.SPEC.TEXT.TOKENIZER[3:])
    else:
        raise ValueError(
            'Unknown tokenizer: {}'.format(config.MODEL.SPEC.TEXT.TOKENIZER)
        )
    """
    zeroshot_weights = zeroshot_classifier(
        classnames, templates, tokenizer, model_without_ddp
    )

    top1 = AverageMeter()
    top5 = AverageMeter()
    for i, (x, y) in enumerate(test_loader):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        features_image = model_without_ddp.encode_image(x)
        logits = 100. * features_image @ zeroshot_weights

        prec1, prec5 = accuracy(logits, y, (1, 5))
        top1.update(prec1, x.size(0))
        top5.update(prec5, x.size(0))

    logging.info('=> synchronize...')
    comm.synchronize()
    top1_acc, top5_acc = map(
        _meter_reduce if distributed else lambda x: x.avg, [top1, top5]
    )

    msg = '=> TEST:\t' \
        'Error@1 {error1:.3f}%\t' \
        'Error@5 {error5:.3f}%\t' \
        'Accuracy@1 {top1:.3f}%\t' \
        'Accuracy@5 {top5:.3f}%\t'.format(
            top1=top1_acc, top5=top5_acc,
            error1=100-top1_acc, error5=100-top5_acc
        )
    logging.info(msg)

    return top1_acc
