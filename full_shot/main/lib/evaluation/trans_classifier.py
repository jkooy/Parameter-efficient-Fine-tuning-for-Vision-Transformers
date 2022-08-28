"""
Linear classifer implemented with Pytorch Linear class
"""
import time
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from evaluation.feature import FeatureData
from core.function import AverageMeter
from optim import build_optimizer
import torch.nn.functional as F

from torch import nn

from functools import partial

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 res_score=False):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.res_score = res_score

    def forward(self, x, prev=None):
        B, C = x.shape
        qkv = self.qkv(x) \
                  .reshape(B, 3, self.num_heads, C // self.num_heads) \
                  .permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn_score = (q @ k.transpose(-2, -1)) * self.scale

        if prev is not None and self.res_score:
            attn_score = attn_score + prev

        if self.res_score:
            prev = attn_score

        attn = F.softmax(attn_score, dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, prev

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
class Classifier(torch.nn.Module):
    """
    Linear classifier.
    """
    def __init__(self, input_dim, output_dim, l2_lambda):
        super(Classifier, self).__init__()
        self.optim = None
        self.l2_lambda = l2_lambda
        self.channel_bn = torch.nn.BatchNorm1d(
            input_dim,
            affine=False,
        )
        self.layers = torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))
        self.norm1 = partial(nn.LayerNorm, eps=1e-6)
        self.attn = Attention(
            dim = 512, num_heads=8, qkv_bias=False, qk_scale=None,
            attn_drop=0, proj_drop=0, res_score=False
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        dim = 512
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=512, hidden_features=int(dim * 4.),
            act_layer=nn.GELU, drop=0.
        )
    def forward(self, feature):
        # outputs = self.channel_bn(feature)
        # outputs = self.layers(outputs)
        prev=None
        attn, prev = self.attn(feature, prev)
        feature = feature + self.drop_path(attn)
        feature = feature + self.drop_path(self.mlp(self.norm2(feature)))

        outputs = self.layers(feature)
        return outputs


def hyperparameter_sweep(train_features, train_labels, val_features, val_labels, config):
    logging.info(f"=> Learning rate {config.TRAIN.LR}: tuning l2 regularization strength.")
    start = time.time()
    l2_lambda_list = np.logspace(-6, 6, num=97).tolist()
    l2_lambda_init_idx = [i for i,val in enumerate(l2_lambda_list) if val in set(np.logspace(-6, 6, num=7))]
    peak_idx = -1
    peak_score = 0
    iter_num = 0
    for idx in l2_lambda_init_idx:
        config.defrost()
        config.TRAIN.WD = l2_lambda_list[idx]
        best_score_, training_time =  train_task(train_features, train_labels, val_features, val_labels, config)
        if best_score_ > peak_score:
            peak_idx = idx
            peak_score = best_score_
    logging.info(f"Iteration {iter_num}: l2_lambda: {l2_lambda_list[peak_idx]}, best score {best_score_}")

    step_span = 8
    while step_span > 0:
        left, right = max(peak_idx - step_span, 0), min(peak_idx + step_span, len(l2_lambda_list)-1)
        search_idx = []
        if left != peak_idx:
            search_idx.append(left)
        if right != peak_idx:
            search_idx.append(right)
        for idx in search_idx:
            config.TRAIN.WD = l2_lambda_list[left]
            best_score_, training_time =  train_task(train_features, train_labels, val_features, val_labels, config)
            if best_score_ > peak_score:
                peak_idx = idx
                peak_score = best_score_
        iter_num += 1
        logging.info(f"Iteration {iter_num}: l2_lambda: {l2_lambda_list[peak_idx]}, best score {best_score_}")
        step_span //= 2
    
    logging.info(f"=> Learning rate {config.TRAIN.LR}: The best l2 lambda is {l2_lambda_list[peak_idx]}")
    logging.info('=> Learning rate {}: l2 regularization strength tuning duration time: {:.2f}s'.format(config.TRAIN.LR, time.time()-start))
    return l2_lambda_list[peak_idx], peak_score

def count_trainable_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params  # output in million

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters())
    return params  # output in million

def train_task(train_features, train_labels, test_features, test_labels, config):
    best_acc1 = 0
    train_loader_feature = torch.utils.data.DataLoader(
        FeatureData(train_features,train_labels), batch_size=config.TRAIN.BATCH_SIZE_PER_GPU)
    test_loader_feature = torch.utils.data.DataLoader(
        FeatureData(test_features,test_labels), batch_size=config.TEST.BATCH_SIZE_PER_GPU)

    model = Classifier(config.MODEL.SPEC.EMBED_DIM, config.DATASET.NUM_CLASSES, 0)
    num_trainable_par = count_trainable_parameters(model)
    num_par = count_parameters(model)
    logging.info(f"num_trainable_par = {num_trainable_par}, num_par = {num_par}")
    gpu = config.GPUS

    if len(gpu) == 1:
        torch.cuda.set_device(gpu[0])
        model = model.cuda(gpu[0])

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda(gpu)
    optimizer = build_optimizer(config, model)

    cudnn.benchmark = True
    

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        adjust_learning_rate(optimizer, epoch, config)

        # train for one epoch
        start = time.time()
        train_one(train_loader_feature, model, criterion, optimizer, epoch, config)
        training_time = time.time()-start
        # evaluate on validation set
        vstart = time.time()
        acc1 = validate(test_loader_feature, model, criterion, epoch, config)
        test_time = time.time()-vstart

        # remember best acc@1 and save checkpoint
        best_acc1 = max(acc1, best_acc1)

    
    logging.info(f'=> Learning rate {config.TRAIN.LR}, L2 lambda {config.TRAIN.WD}: Best score: Acc@1 {best_acc1:.3f}')
    logging.info(f'=> Training cost time {training_time}, Testing cost time {test_time}')
    return best_acc1, training_time


def train_one(train_loader, model, criterion, optimizer, epoch, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for _, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if len(config.GPUS)==1:
            images = images.cuda(config.GPUS[0], non_blocking=True)
        target = target.cuda(config.GPUS[0], non_blocking=True)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    logging.info(f'[Epoch {epoch}] Train: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')


def validate(val_loader, model, criterion, epoch, config):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        end = time.time()
        for _, (images, target) in enumerate(val_loader):
            if len(config.GPUS)==1:
                images = images.cuda(config.GPUS[0], non_blocking=True)
            target = target.cuda(config.GPUS[0], non_blocking=True)
            # compute output
            output = model(images)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        logging.info(f'[Epoch {epoch}] Val:   Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')

    return top1.avg


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


def hyperparameter_sweep_lr(train_features, train_labels, val_features, val_labels, config):
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
        l2_lambda, best_score_one = hyperparameter_sweep(train_features, train_labels, val_features, val_labels, config)
        logging.info(f"=> Learning rate: {lr_one}, best_score {best_score_one}")
        if best_score < best_score_one:
            best_score = best_score_one
            best_lr = lr_one
            best_l2_lambda = l2_lambda
    logging.info(f"Hyper parameter tuning result: learning rate {best_lr}, l2_lambda {best_l2_lambda}")
    logging.info('=> Hyperparameter tuning duration time: {:.2f}s'.format(time.time()-start))
    logging.info('=> Finished hyperparameter tuning.')
    return best_lr, best_l2_lambda

def trans_classifier(train_features, train_labels, val_features, val_labels, test_features, test_labels, no_hyperparameter_tuning, lr, l2, config):
    if no_hyperparameter_tuning:
        best_lr = lr
        best_l2_lambda = l2
    else:
        best_lr, best_l2_lambda = hyperparameter_sweep_lr(train_features, train_labels, val_features, val_labels, config)

    logging.info("=> The final classifier is on training ...")
    logging.info(f"Hyperparameters: learning_rate = {best_lr}, l2_lambda = {best_l2_lambda}")
    config.defrost()
    config.TRAIN.LR = best_lr
    config.TRAIN.WD = best_l2_lambda
    config.TRAIN.END_EPOCH += config.TRAIN.EXTRA_FINAL_TRAIN_EPOCH
    config.freeze()
    best_acc1, training_time = train_task(np.concatenate((train_features, val_features)), np.concatenate((train_labels, val_labels)), test_features, test_labels, config)
    return best_acc1, training_time