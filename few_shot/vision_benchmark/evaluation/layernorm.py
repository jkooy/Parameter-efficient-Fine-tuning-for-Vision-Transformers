"""
Linear classifier implemented with Pytorch Linear class
"""
import time
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .feature import FeatureData, get_model
from ..optim import build_optimizer
from ..evaluation.metric import get_metric

import pdb



from functools import partial
from itertools import repeat

import logging
import os

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_

import collections.abc as container_abcs

MULTILABEL_DATASETS = {"voc-2007-classification","chestx-ray8"}


np.random.seed(3)



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


class Classifier(torch.nn.Module):
    """
    Linear classifier.
    """

    def __init__(self, config, l2_lambda):
        super(Classifier, self).__init__()

        feature_type="image"
        self.backbone = get_model(config, feature_type=feature_type)


        for n, param in self.backbone.named_parameters():
            if 'norm' in n:
                param.requires_grad = True
                logging.info(f'{n}, shape: {param.shape}, require grad.')
            else:
                param.requires_grad = False
                logging.info(f"=> Paramters in the model not require grad: {n}.")
            


        input_dim, output_dim = config.MODEL.SPEC.EMBED_DIM, config.DATASET.NUM_CLASSES
        self.optim = None
        self.l2_lambda = l2_lambda
        self.channel_bn = torch.nn.BatchNorm1d(
            input_dim,
            affine=False,
        )
        self.layers = torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))

    def forward(self, img):
        pdtype = img.dtype
        feature = self.backbone(img).to(pdtype)
        outputs = self.channel_bn(feature)

        # pdb.set_trace()
        outputs = self.layers(outputs)
        return outputs


def hyperparameter_sweep(train_dataloader, val_dataloader, config):
    logging.info(f"=> Learning rate {config.TRAIN.LR}: tuning l2 regularization strength.")
    start = time.time()
    l2_lambda_list = np.logspace(-6, 6, num=97).tolist()
    # l2_lambda_list = np.logspace(-3, 3, num=97).tolist()
    l2_lambda_init_idx = [i for i, val in enumerate(l2_lambda_list) if val in set(np.logspace(-6, 6, num=7))]
    peak_idx = -1
    peak_score = 0
    iter_num = 0
    for idx in l2_lambda_init_idx:
        config.defrost()
        config.TRAIN.WD = l2_lambda_list[idx]

        # best_score_ = train_task(train_dataloader, val_dataloader, config)
        try:
            best_score_ = train_task(train_dataloader, val_dataloader, config)
        except:
            best_score_ = 0.0
            continue       

        if best_score_ > peak_score:
            peak_idx = idx
            peak_score = best_score_
    logging.info(f"Iteration {iter_num}: l2_lambda: {l2_lambda_list[peak_idx]}, best score {best_score_}")

    step_span = 8
    while step_span > 0:
        left, right = max(peak_idx - step_span, 0), min(peak_idx + step_span, len(l2_lambda_list) - 1)
        search_idx = []
        if left != peak_idx:
            search_idx.append(left)
        if right != peak_idx:
            search_idx.append(right)
        for idx in search_idx:
            config.TRAIN.WD = l2_lambda_list[left]
            
            # best_score_ = train_task(train_dataloader, val_dataloader, config)
            try:
                best_score_ = train_task(train_dataloader, val_dataloader, config)
            except:
                best_score_ = 0.0
                continue

            if best_score_ > peak_score:
                peak_idx = idx
                peak_score = best_score_
        iter_num += 1
        logging.info(f"Iteration {iter_num}: l2_lambda: {l2_lambda_list[peak_idx]}, best score {best_score_}")
        step_span //= 2

    logging.info(f"=> Learning rate {config.TRAIN.LR}: The best l2 lambda is {l2_lambda_list[peak_idx]}")
    logging.info('=> Learning rate {}: l2 regularization strength tuning duration time: {:.2f}s'.format(config.TRAIN.LR, time.time() - start))
    return l2_lambda_list[peak_idx], peak_score


def train_task(train_dataloader, test_dataloader, config):
    best_acc1 = 0

    model = Classifier(config, 0)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    logging.info(f'Total number of trainable params: {pytorch_total_params / 1000000}M.')


    gpu = config.GPUS

    if len(gpu) == 1:
        torch.cuda.set_device(gpu[0])
        model = model.cuda(gpu[0])

    # define loss function (criterion) and optimizer
    if config.DATASET.DATASET in MULTILABEL_DATASETS:
        criterion = torch.nn.BCEWithLogitsLoss().cuda(gpu)
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda(gpu)

    optimizer = build_optimizer(config, model)

    cudnn.benchmark = True

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        adjust_learning_rate(optimizer, epoch, config)

        # train for one epoch
        train_one(train_dataloader, model, criterion, optimizer, epoch, config)

        # evaluate on validation set
        acc1 = validate(test_dataloader, model, criterion, epoch, config)

        # remember best acc@1 and save checkpoint
        best_acc1 = max(acc1, best_acc1)

    logging.info(f'=> Learning rate {config.TRAIN.LR}, L2 lambda {config.TRAIN.WD}: Best score: Acc@1 {best_acc1:.3f}')
    return best_acc1



def train_one(train_loader, model, criterion, optimizer, epoch, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    metric = get_metric(config.TEST.METRIC)

    if config.TEST.METRIC == "11point_mAP":
        mAP = AverageMeter()
    else:
        top1 = AverageMeter()
        top5 = AverageMeter()


    end = time.time()
    for _,  batch in enumerate(train_loader):

        images, target = batch[:2]

        # measure data loading time
        data_time.update(time.time() - end)

        if len(config.GPUS) == 1:
            images = images.cuda(config.GPUS[0], non_blocking=True)

        if target.shape[-1] == 1: target = target[:,0]
        target = target.cuda(config.GPUS[0], non_blocking=True)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        output = model.forward(images)

        # pdb.set_trace()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), images.size(0))

        # measure accuracy and record loss
        if config.TEST.METRIC == "11point_mAP":
            target_np = target.cpu().detach().numpy() 
            output_np = output.cpu().detach().numpy() 
            map_score = metric(target_np, output_np)
            mAP.update(map_score * 100.0, images.size(0))
        else:
            if config.DATASET.NUM_CLASSES < 5:
                topk_large = config.DATASET.NUM_CLASSES
            else:
                topk_large = 5
            acc1, acc5 = accuracy(output, target, topk=(1, topk_large))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))     

        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    if config.TEST.METRIC == "11point_mAP":
        logging.info(f'[Epoch {epoch}] Train: mAP {mAP.avg:.3f}')
    else:
        logging.info(f'[Epoch {epoch}] Train: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')


def validate(val_loader, model, criterion, epoch, config):
    batch_time = AverageMeter()
    losses = AverageMeter()
    metric = get_metric(config.TEST.METRIC)

    if config.TEST.METRIC == "11point_mAP":
        mAP = AverageMeter()
    else:
        top1 = AverageMeter()
        top5 = AverageMeter()


    model.eval()
    with torch.no_grad():
        end = time.time()
        for _, batch in enumerate(val_loader):

            images, target = batch[:2]

            if len(config.GPUS) == 1:
                images = images.cuda(config.GPUS[0], non_blocking=True)
            target = target.cuda(config.GPUS[0], non_blocking=True)
            if target.shape[-1] == 1: target = target[:,0]
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            if config.TEST.METRIC == "11point_mAP":
                target_np = target.cpu().detach().numpy() 
                output_np = output.cpu().detach().numpy() 
                map_score = metric(target_np, output_np)
                mAP.update(map_score * 100.0, images.size(0))
            else:
                if config.DATASET.NUM_CLASSES < 5:
                    topk_large = config.DATASET.NUM_CLASSES
                else:
                    topk_large = 5
                acc1, acc5 = accuracy(output, target, topk=(1, topk_large))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if config.TEST.METRIC == "11point_mAP":
            logging.info(f'[Epoch {epoch}] Val: mAP {mAP.avg:.3f}')
        else:
            logging.info(f'[Epoch {epoch}] Val: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')

    return mAP.avg if config.TEST.METRIC == "11point_mAP" else top1.avg 


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate based on schedule"""
    lr = config.TRAIN.LR
    for milestone in config.TRAIN.SCHEDULE:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def hyperparameter_sweep_lr(train_dataloader, val_dataloader, config):
    logging.info("=> Start hyperparameter tuning.")
    start = time.time()
    learning_rate_list = np.logspace(-6, -1, num=6).tolist()
    best_score = 0
    best_lr = 0
    best_l2_lambda = 0
    for lr_one in learning_rate_list:
        config.defrost()
        config.TRAIN.LR = lr_one
        config.freeze()
        l2_lambda, best_score_one = hyperparameter_sweep(train_dataloader, val_dataloader, config)
        logging.info(f"=> Learning rate: {lr_one}, best_score {best_score_one}")
        if best_score < best_score_one:
            best_score = best_score_one
            best_lr = lr_one
            best_l2_lambda = l2_lambda
    logging.info(f"Hyper parameter tuning result: learning rate {best_lr}, l2_lambda {best_l2_lambda}")
    logging.info('=> Hyperparameter tuning duration time: {:.2f}s'.format(time.time() - start))
    logging.info('=> Finished hyperparameter tuning.')
    return best_lr, best_l2_lambda


def layernormt(train_dataloader, val_dataloader, test_dataloader, no_hyperparameter_tuning, lr, l2, config):

    if no_hyperparameter_tuning:
        best_lr = lr
        best_l2_lambda = l2
    else:
        best_lr, best_l2_lambda = hyperparameter_sweep_lr(train_dataloader, val_dataloader, config)

    logging.info("=> The final classifier is on training ...")
    logging.info(f"Hyperparameters: learning_rate = {best_lr}, l2_lambda = {best_l2_lambda}")
    config.defrost()
    config.TRAIN.LR = best_lr
    config.TRAIN.WD = best_l2_lambda
    config.TRAIN.END_EPOCH += config.TRAIN.EXTRA_FINAL_TRAIN_EPOCH
    config.freeze()

    # TODO: correct train_dataloader to include both train and val
    train_task(train_dataloader, test_dataloader, config)

