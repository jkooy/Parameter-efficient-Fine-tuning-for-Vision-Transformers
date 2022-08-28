"""
Finetune full model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import logging
import os
import pprint
import numpy as np
import torch
import torch.nn.parallel
from torch.utils.collect_env import get_pretty_env_info
# These 2 lines are a wordaround for "Too many open files error". Refer: https://github.com/pytorch/pytorch/issues/11201
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import _init_paths
from utils.comm import comm
from utils.utils import create_logger
from evaluation.feature import extract_features
from evaluation.linear_classifier import linear_classifier
from evaluation.logistic_classifier import lr_classifier
from evaluation.multi_label import multlabel_lr_classifier
from config import config
from config import update_config
from torch import nn
import torchvision
import clip

from torchvision import transforms
import models

import json
from pathlib import Path
import timm
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F

from optim import build_optimizer
import torch.backends.cudnn as cudnn
from core.function import AverageMeter
from sklearn.model_selection import train_test_split
from PIL import Image
#########################################
# The following 2 lines are to solve PIL "IOError: image file truncated" with big images. 
# Refer to https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#########################################

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

#pip install ptflops
from ptflops import get_model_complexity_info

cfg_files_dataset = {
    "caltech101": "experiments/eval/dataset/caltech101.yaml",
    "cifar10": "experiments/eval/dataset/cifar10.yaml",
    "cifar100": "experiments/eval/dataset/cifar100.yaml",
    "dtd": "experiments/eval/dataset/dtd.yaml",
    "fer2013": "experiments/eval/dataset/fer2013.yaml",
    "fgvc-aircraft-2013b": "experiments/eval/dataset/fgvc-aircraft-2013b.yaml",
    "flower102": "experiments/eval/dataset/flower102.yaml",
    "food101": "experiments/eval/dataset/food101.yaml",
    "gtsrb": "experiments/eval/dataset/gtsrb.yaml",
    "hatefulmemes": "experiments/eval/dataset/hatefulmemes.yaml",
    "mnist": "experiments/eval/dataset/mnist.yaml",
    "patchcamelyon": "experiments/eval/dataset/patchcamelyon.yaml",
    "pet37": "experiments/eval/dataset/pet37.yaml",
    "stanfordcar": "experiments/eval/dataset/stanfordcar.yaml",
    "stl10": "experiments/eval/dataset/stl10.yaml",
    "sun397": "experiments/eval/dataset/sun397.yaml",
    "ucf101": "experiments/eval/dataset/ucf101.yaml",
    "voc2007classification": "experiments/eval/dataset/voc2007classification.yaml",
    # "imagenet": "experiments/eval/dataset/imagenet.yaml",
    }

MULTILABEL_DATASETS = set(["voc2007classification", "voc2007", "chestx-ray8"])

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test a classification model.')

    parser.add_argument('--ds',
                        required=True,
                        help='Evaluation dataset configure file name.',
                        type=str)

    parser.add_argument('--model',
                        required=True,
                        help='Evaluation model configure file name',
                        type=str)

    parser.add_argument('-c', '--classifier',
                        choices=['logistic', 'linear'], #logistic - sklearn logistic_regression, linear - torch.nn.linear
                        help='Classifier used for linear probe',
                        default='logistic',
                        type=str)

    parser.add_argument('--save-feature',
                        help='Flag to save feature or not',
                        default=False,
                        type=str)

    parser.add_argument('--no-tuning',
                        help='No hyperparameter-tuning.',
                        default=False,
                        type=str)

    parser.add_argument('--lr-range',
                        help='learning rate search range.',
                        default=None,
                        type=float)

    parser.add_argument('--l2',
                        help='(Inverse) L2 regularization strength. This option is only useful when option --no-tuning is True.',
                        default=0.316,
                        type=float)

    parser.add_argument('--lr',
                        help='Test with a specific learning rate. This option is only useful when option --no-tuning is True.',
                        default=0.001,
                        type=float)

    parser.add_argument('--dintrinsic',
                        help='Test with a different intrinsic dimension.',
                        default=100,
                        type=int)

    parser.add_argument('--layerType',
                        help='Test with a different layerType, e.g. mlp, attention, adapter.',
                        default= "mlp",
                        type=str)

    parser.add_argument('--layernum',
                        help='Test intrinsic dimension in the given layer.',
                        type=int)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args

# count parameters
def count_trainable_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params  # output in million

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters())
    return params  # output in million




class FastFoodWrap(nn.Module):
    def __init__(self, module, intrinsic_dimension, device=0):
        """
        Wrapper to estimate the intrinsic dimensionality of the
        objective landscape for a specific task given a specific model using FastFood transform
        :param module: pytorch nn.Module
        :param intrinsic_dimension: dimensionality within which we search for solution
        :param device: cuda device id
        """
        super(FastFoodWrap, self).__init__()

        # Hide this from inspection by get_parameters()
        self.m = [module]

        self.name_base_localname = []

        # Stores the initial value: \theta_{0}^{D}
        self.initial_value = dict()

        # Fastfood parameters
        self.fastfood_params = {}

        # Parameter vector that is updated
        # Initialised with zeros as per text: \theta^{d}
        V = nn.Parameter(torch.zeros((intrinsic_dimension)).to(device))
        self.register_parameter("V", V)
        v_size = (intrinsic_dimension,)

        # Iterate over layers in the module
        for name, param in module.named_parameters():
            # If param requires grad update
            if param.requires_grad:

                # Saves the initial values of the initialised parameters from param.data and sets them to no grad.
                # (initial values are the 'origin' of the search)
                self.initial_value[name] = v0 = (
                    param.clone().detach().requires_grad_(False).to(device)
                )

                # Generate fastfood parameters
                DD = np.prod(v0.size())
                self.fastfood_params[name] = fastfood_vars(DD, device)

                base, localname = module, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = base.__getattr__(prefix)
                self.name_base_localname.append((name, base, localname))

        for name, base, localname in self.name_base_localname:
            delattr(base, localname)

    def forward(self, x):
        # Iterate over layers
        for name, base, localname in self.name_base_localname:

            init_shape = self.initial_value[name].size()
            DD = np.prod(init_shape)

            # Fastfood transform te replace dence P
            ray = fastfood_torched(self.V, DD, self.fastfood_params[name]).view(
                init_shape
            )

            param = self.initial_value[name] + ray

            setattr(base, localname, param)

        # Pass through the model, by getting hte module from a list self.m
        module = self.m[0]
        x = module(x)
        return x


def fast_walsh_hadamard_torched(x, axis=0, normalize=False):
    """
    Performs fast Walsh Hadamard transform
    :param x:
    :param axis:
    :param normalize:
    :return:
    """
    orig_shape = x.size()
    assert axis >= 0 and axis < len(orig_shape), (
        "For a vector of shape %s, axis must be in [0, %d] but it is %d"
        % (orig_shape, len(orig_shape) - 1, axis)
    )
    h_dim = orig_shape[axis]
    h_dim_exp = int(round(np.log(h_dim) / np.log(2)))
    assert h_dim == 2 ** h_dim_exp, (
        "hadamard can only be computed over axis with size that is a power of two, but"
        " chosen axis %d has size %d" % (axis, h_dim)
    )

    working_shape_pre = [int(np.prod(orig_shape[:axis]))]  # prod of empty array is 1 :)
    working_shape_post = [
        int(np.prod(orig_shape[axis + 1 :]))
    ]  # prod of empty array is 1 :)
    working_shape_mid = [2] * h_dim_exp
    working_shape = working_shape_pre + working_shape_mid + working_shape_post

    ret = x.view(working_shape)

    for ii in range(h_dim_exp):
        dim = ii + 1
        arrs = torch.chunk(ret, 2, dim=dim)
        assert len(arrs) == 2
        ret = torch.cat((arrs[0] + arrs[1], arrs[0] - arrs[1]), axis=dim)

    if normalize:
        ret = ret / torch.sqrt(float(h_dim))

    ret = ret.view(orig_shape)

    return ret


def fastfood_vars(DD, device=0):
    """
    Returns parameters for fast food transform
    :param DD: desired dimension
    :return:
    """
    ll = int(np.ceil(np.log(DD) / np.log(2)))
    LL = 2 ** ll

    # Binary scaling matrix where $B_{i,i} \in \{\pm 1 \}$ drawn iid
    BB = torch.FloatTensor(LL).uniform_(0, 2).type(torch.LongTensor)
    BB = (BB * 2 - 1).type(torch.FloatTensor).to(device)
    BB.requires_grad = False

    # Random permutation matrix
    Pi = torch.LongTensor(np.random.permutation(LL)).to(device)
    Pi.requires_grad = False

    # Gaussian scaling matrix, whose elements $G_{i,i} \sim \mathcal{N}(0, 1)$
    GG = torch.FloatTensor(LL,).normal_().to(device)
    GG.requires_grad = False

    divisor = torch.sqrt(LL * torch.sum(torch.pow(GG, 2)))

    return [BB, Pi, GG, divisor, LL]


def fastfood_torched(x, DD, param_list=None, device=0):
    """
    Fastfood transform
    :param x: array of dd dimension
    :param DD: desired dimension
    :return:
    """
    dd = x.size(0)

    if not param_list:

        BB, Pi, GG, divisor, LL = fastfood_vars(DD, device=device)

    else:

        BB, Pi, GG, divisor, LL = param_list

    # Padd x if needed
    dd_pad = F.pad(x, pad=(0, LL - dd), value=0, mode="constant")

    # From left to right HGPiH(BX), where H is Walsh-Hadamard matrix
    mul_1 = torch.mul(BB, dd_pad)
    # HGPi(HBX)
    mul_2 = fast_walsh_hadamard_torched(mul_1, 0, normalize=False)

    # HG(PiHBX)
    mul_3 = mul_2[Pi]

    # H(GPiHBX)
    mul_4 = torch.mul(mul_3, GG)

    # (HGPiHBX)
    mul_5 = fast_walsh_hadamard_torched(mul_4, 0, normalize=False)

    ret = torch.div(mul_5[:DD], divisor * np.sqrt(float(DD) / LL))

    return ret



# add clip wrapper
class CLIP_wrapper(nn.Module):
  '''
  class that takes the pretrained backbone of clip model
  and can be then trained as in image classifier
  '''

  def __init__(
      self,
      clip_model,
      model_name,
      input_dim, 
      output_dim,
      freeze_visual=False,
      dintrinsic = 100,
      layerType = "mlp",
      layernum = 100
    ):
    super().__init__()
    model = clip_model.to(dtype=torch.float32)
    torch.save(model.state_dict(), "vit.pth") 
    self.model = torch.load('vit.pth')
    self.adapter_model = models.cls_intrinsic_dimension.get_cls_model(config, dintrinsic, layerType, layernum)
    adapter_model_dict = self.adapter_model.state_dict()
    state_dict = {k:v for k,v in self.model.items() if k in adapter_model_dict.keys()}
    adapter_model_dict.update(state_dict)
    self.adapter_model.load_state_dict(adapter_model_dict)
    # self.layers = FastFoodWrap(torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim)), intrinsic_dimension=200)
    self.layers = torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))
    self.model_name = model_name
    
    for n, param in self.adapter_model.named_parameters():
        print(f'{n}, shape: {param.shape}')
        if 'intrinsic' in n:
            print(f'intrinsic parameter, {n}, shape: {param.shape}')
        else:
            param.requires_grad = False
            
    
    '''
    if model_name == "vit_base_patch16_224":
        # for block in self.adapter_model.blocks:
        #     for param in block.adapter.parameters():
        #         param.requires_grad = True
        for param in self.adapter_model.blocks[0].adapter.parameters():
            param.requires_grad = True
        # for param in self.adapter_model.adapter.parameters():
        #     param.requires_grad = True
    else:
        for param in self.adapter_model.visual.ln_pre.parameters():
            param.requires_grad = True

        for param in self.adapter_model.visual.ln_post.parameters():
            param.requires_grad = True
    '''


  def forward(self, batch):
    if self.model_name == "clip_openai":
        image_features = self.adapter_model.visual(batch)
        outputs = self.layers(image_features)
    else:
        image_features = self.adapter_model(batch)
        outputs = self.layers(image_features)
    return outputs


def load_oneclassification_model(config):
    logging.info(f"Loading evaluated model {config.MODEL.NAME}.")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = eval('models.' + config.MODEL.NAME + '.get_cls_model')(config)
    model_file = config.TEST.MODEL_FILE
    logging.info('=> load model file: {}'.format(model_file))
    ext = model_file.split('.')[-1]
    # add .pt model support
    if ext == 'pt':
        model = torch.jit.load(model_file, map_location="cpu")
    else:
        if ext == 'pth':
            state_dict = torch.load(model_file, map_location="cpu")
        
        elif ext == 'pkl':
            logging.info('=> load pkl model')
            with open(model_file, 'rb') as f:
                state_dict = pickle.load(f)['model']

            for k, v in state_dict.items():
                state_dict[k] = torch.from_numpy(v)
        else:
            raise ValueError("Unknown model file")
        if config.TEST.MODEL_KEY:
            state_dict = state_dict[config.TEST.MODEL_KEY]
        msg = model.load_state_dict(state_dict, strict=False)

        logging.info(msg)
    return model


def get_model(config, feature_type="image"):
    model_name = config.MODEL.NAME
    if model_name in dir(torchvision.models):
        model_pretrained = eval("torchvision.models."+ model_name)(pretrained=True)
        model = EvalModel(model_pretrained)
        logging.info(f"Using Pytorch pretrained model {model_name}")
    elif model_name in timm.list_models(pretrained=True):
        model = timm.create_model(model_name, pretrained=True)
        if model_name.startswith("efficientnet"):
            model = EvalModel(model)
        elif model_name.startswith("vit"):
            model.forward = model.forward_features
        else:
            raise Exception("Please define Timm feature extraction model.")
        logging.info(f"Using Timm pretrained model {model_name}")
    elif model_name in dir(models):
        logging.info(f"Using Pytorch pretrained model {model_name}")
        model = load_oneclassification_model(config)
        if model_name == "clip_openai":
            model.forward = model.encode_image
        else:
            model.forword = model.forward_features
    elif model_name in clip.available_models():
        model, _ = clip.load(model_name, jit=False)
        if feature_type == "image":
            model.forward = model.encode_image
        elif feature_type == "text":
            model.forward = model.encode_text
        else:
            raise Exception("Incorrect model type.")
        logging.info(f"Using CLIP pretrained model {model_name}")
    else:
        raise Exception("Wrong model name.")
    return model, model_name

def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate based on schedule"""
    lr = config.TRAIN.LR
    for milestone in config.TRAIN.SCHEDULE:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_task(train_dataloader, val_dataloader, config, args):
    best_acc1 = 0

    model, model_name = get_model(config, feature_type="image")

    if args.layerType == "attention":
        logging.info('=> intrinsic attention mode')
    elif args.layerType == "adapter":
        logging.info('=> intrinsic adapter mode')
    elif args.layerType == "mlp":
        logging.info('=> intrinsic mlp mode')

    model = CLIP_wrapper(model, model_name, config.MODEL.SPEC.EMBED_DIM, config.DATASET.NUM_CLASSES, freeze_visual=False, dintrinsic = args.dintrinsic, layerType = args.layerType, layernum = args.layernum)
    logging.info(model)

    num_trainable_par = count_trainable_parameters(model)
    num_par = count_parameters(model)
    logging.info(f"num_trainable_par = {num_trainable_par}, num_par = {num_par}")
    
    gpu = config.GPUS

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # define loss function (criterion) and optimizer
    if config.DATASET.DATASET in MULTILABEL_DATASETS:
        criterion = torch.nn.MultiLabelSoftMarginLoss().cuda(gpu)
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda(gpu)
    optimizer = build_optimizer(config, model)

    cudnn.benchmark = True
    

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        logging.info(f'=> epoch={epoch}')
        adjust_learning_rate(optimizer, epoch, config)


        start = time.time()
        # train for one epoch
        train_one(train_dataloader, model, criterion, optimizer, epoch, config)
        training_time = time.time()-start

        # evaluate on validation set
        vstart = time.time()
        acc1 = validate(val_dataloader, model, criterion, epoch, config)
        test_time = time.time()-vstart

        # remember best acc@1 and save checkpoint
        best_acc1 = max(acc1, best_acc1)
        logging.info(f'=> best_acc1={best_acc1}')

    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True)
    logging.info(f'=> Learning rate {config.TRAIN.LR}, L2 lambda {config.TRAIN.WD}: Best score: Acc@1 {best_acc1:.3f}')
    logging.info(f'=> Flops: {flops}, Params: {params}')
    logging.info(f'=> Training cost time {training_time}, Testing cost time {test_time}')
    return best_acc1, training_time, num_trainable_par, num_par


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
            if config.DATASET.DATASET in MULTILABEL_DATASETS:
                acc1 = compute_mAP(target.data,output.data)
                '''
                mAP_sum = 0
                for i, estimator in enumerate(target.data[0, :]):
                    acc = mAP_11points(target.data[:, i],output.data[:, i])
                mAP_sum += acc
                acc1 = mAP_sum/len(target.data[0, :])
                '''
                top1.update(acc1, images.size(0))
                top5.update(0, images.size(0))
            else:
                if config.DATASET.DATASET == "hatefulmemes" or config.DATASET.DATASET == "patchcamelyon":
                    #binary classification dataset
                    acc1, acc5 = accuracy(output, target, topk=(1, 2))
                else:
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
            losses.update(loss.item(), images.size(0))
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        logging.info(f'[Epoch {epoch}] Val:   Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')

    return top1.avg


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

def compute_mAP(labels,outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i],y_pred[i]))
    return np.mean(AP)
    
def mAP_11points(y_label, y_pred_proba):
    y_label = y_label.cpu().numpy()
    y_pred_proba = y_pred_proba.cpu().numpy()
    precision, recall, _ = precision_recall_curve(y_label, y_pred_proba)
    recall_thresholds = np.linspace(1, 0, 11, endpoint=True).tolist()
    precision_sum = 0
    recall_idx = 0
    precision_tmp = 0
    for threshold in recall_thresholds:
        while recall_idx < len(recall) and threshold <= recall[recall_idx]:
            precision_tmp = max(precision_tmp, precision[recall_idx])
            recall_idx += 1
        precision_sum += precision_tmp
    return precision_sum/11

def train_one(train_loader, model, criterion, optimizer, epoch, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # logging.info(f'=> device is {device}')
    end = time.time()
    for _, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if device == torch.device("cuda"):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output, target)
        # logging.info(f'=> loss={loss}')
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        if config.DATASET.DATASET in MULTILABEL_DATASETS:
            acc1 = compute_mAP(target.data,output.data)
            '''
            mAP_sum = 0
            for i, estimator in enumerate(target.data[0, :]):
                acc = mAP_11points(target.data[:, i],output.data[:, i])
            mAP_sum += acc
            acc1 = mAP_sum/len(target.data[0, :])
            '''
            top1.update(acc1, images.size(0))
            top5.update(0, images.size(0))
        else:
            if config.DATASET.DATASET == "hatefulmemes" or config.DATASET.DATASET == "patchcamelyon":
                #binary case dataset
                acc1, acc5 = accuracy(output, target, topk=(1, 2))
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))        
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
        losses.update(loss.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    logging.info(f'[Epoch {epoch}] Train: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')

def hyperparameter_sweep(train_dataloader, val_dataloader, config, args):
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
        best_score_, training_time, num_trainable_par, num_par =  train_task(train_dataloader, val_dataloader, config, args)
        if best_score_ > peak_score:
            peak_idx = idx
            peak_score = best_score_
        torch.cuda.empty_cache()
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
            best_score_, training_time, num_trainable_par, num_par =  train_task(train_dataloader, val_dataloader, config, args)
            if best_score_ > peak_score:
                peak_idx = idx
                peak_score = best_score_
        iter_num += 1
        logging.info(f"Iteration {iter_num}: l2_lambda: {l2_lambda_list[peak_idx]}, best score {best_score_}")
        step_span //= 2
    
    logging.info(f"=> Learning rate {config.TRAIN.LR}: The best l2 lambda is {l2_lambda_list[peak_idx]}")
    logging.info('=> Learning rate {}: l2 regularization strength tuning duration time: {:.2f}s'.format(config.TRAIN.LR, time.time()-start))
    return l2_lambda_list[peak_idx], peak_score
    
def hyperparameter_sweep_lr(train_dataloader, val_dataloader, config, args):
    logging.info("=> Start hyperparameter tuning.")
    start = time.time()
    # add learning rate search range
    if args.lr_range:
        learning_rate_list = [args.lr_range]
    else:
        learning_rate_list = np.logspace(-6, -1, num=6).tolist()
    # learning_rate_list = np.logspace(-6, -1, num=6).tolist()
    best_score = 0
    best_lr = 0
    best_l2_lambda = 0
    for lr_one in learning_rate_list:
        config.defrost()
        config.TRAIN.LR = lr_one
        config.freeze()
        l2_lambda, best_score_one = hyperparameter_sweep(train_dataloader, val_dataloader, config, args)
        logging.info(f"=> Learning rate: {lr_one}, best_score {best_score_one}")
        if best_score < best_score_one:
            best_score = best_score_one
            best_lr = lr_one
            best_l2_lambda = l2_lambda
    logging.info(f"Hyper parameter tuning result: learning rate {best_lr}, l2_lambda {best_l2_lambda}")
    logging.info('=> Hyperparameter tuning duration time: {:.2f}s'.format(time.time()-start))
    logging.info('=> Finished hyperparameter tuning.')
    return best_lr, best_l2_lambda


def classifier(train_dataloader, val_dataloader, test_dataloader, no_hyperparameter_tuning, lr, l2, config, args):

    best_lr = args.lr_range
    best_l2_lambda = config.TRAIN.WD
    '''
    if no_hyperparameter_tuning is True:
        best_lr = lr
        best_l2_lambda = l2
    else:
        best_lr, best_l2_lambda = hyperparameter_sweep_lr(train_dataloader, val_dataloader, config, args)
    '''

    logging.info("=> The classifier is on training ...")
    logging.info(f"Hyperparameters: learning_rate = {best_lr}, l2_lambda = {best_l2_lambda}")
    config.defrost()
    config.TRAIN.LR = best_lr
    config.TRAIN.WD = best_l2_lambda
    config.TRAIN.END_EPOCH += config.TRAIN.EXTRA_FINAL_TRAIN_EPOCH
    config.freeze()
    best_acc1, training_time, num_trainable_par, num_par = train_task(train_dataloader, val_dataloader, config, args)
    return best_acc1, training_time, num_trainable_par, num_par 


def get_dataloader(dataset, val_split=0, batch_size_per_gpu=64, workers=6, pin_memory=True):
    workers = 2
    if val_split == 0:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            sampler=None,
            drop_last=False,
        )
        return data_loader
    else:
        def train_val_dataset(dataset, val_split):
            train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
            datasets = {}
            datasets['train'] = Subset(dataset, train_idx)
            datasets['val'] = Subset(dataset, val_idx)
            return datasets
        datasets = train_val_dataset(dataset, val_split)
        train_loader = torch.utils.data.DataLoader(
            datasets['train'],
            batch_size=batch_size_per_gpu,
            shuffle=True,
            num_workers=workers,
            pin_memory=pin_memory,
            sampler=None,
            drop_last=False,
        )
        val_loader = torch.utils.data.DataLoader(
            datasets['val'],
            batch_size=batch_size_per_gpu,
            shuffle=True,
            num_workers=workers,
            pin_memory=pin_memory,
            sampler=None,
            drop_last=False,
        )
        return train_loader, val_loader

def linear_probe():
    args = parse_args()
    args.cfg = args.ds
    update_config(config, args)
    args.cfg = args.model
    update_config(config, args)
    config.defrost()
    config.NAME = ""

    config.DATASET.ROOT = os.path.join(config.DATASET.ROOT, config.DATASET.DATASET)

    config.freeze()

    final_output_dir = create_logger(config, args.cfg, 'intrinsic_dimension')
    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> saving logging info into: {}".format(final_output_dir))

    transform_CLIP = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.INPUT.MEAN, std=config.INPUT.STD),
        ])
    if config.DATASET.DATASET == "voc2007classification" or config.DATASET.DATASET == "voc2007" :
        from evaluation.dataset import Voc2007Classification
        train_dataloader = get_dataloader(Voc2007Classification(config.DATASET.ROOT, image_set="train", transform=transform_CLIP))
        val_dataloader = get_dataloader(Voc2007Classification(config.DATASET.ROOT, image_set="val", transform=transform_CLIP))
        test_dataloader = get_dataloader(Voc2007Classification(config.DATASET.ROOT, image_set="test", transform=transform_CLIP))
    elif config.DATASET.DATASET == "hatefulmemes":
        from evaluation.dataset import HatefulMemes
        train_dataloader, val_dataloader = get_dataloader(HatefulMemes(config.DATASET.ROOT, image_set="train", transform=transform_CLIP), val_split=0.2)
        test_dataloader = get_dataloader(HatefulMemes(config.DATASET.ROOT, image_set="val", transform=transform_CLIP))
    elif config.DATASET.DATASET == "chestx-ray8":
        from evaluation.dataset import ChestXRay8
        train_dataloader, val_dataloader = get_dataloader(ChestXRay8(config.DATASET.ROOT, image_set="train", transform=transform_CLIP), val_split=0.2)
        test_dataloader = get_dataloader(ChestXRay8(config.DATASET.ROOT, image_set="test", transform=transform_CLIP))           
    else:
        if config.DATASET.VAL_SET:
            train_dataloader = get_dataloader(torchvision.datasets.ImageFolder(os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET), transform=transform_CLIP))
            val_dataloader = get_dataloader(torchvision.datasets.ImageFolder(os.path.join(config.DATASET.ROOT, config.DATASET.VAL_SET), transform=transform_CLIP))
        else:
            train_dataloader, val_dataloader = get_dataloader(torchvision.datasets.ImageFolder(os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET), transform=transform_CLIP), val_split=0.2)
        test_dataloader = get_dataloader(torchvision.datasets.ImageFolder(os.path.join(config.DATASET.ROOT, config.DATASET.TEST_SET), transform=transform_CLIP))

    best_acc, training_time, num_trainable_par, num_par = classifier(train_dataloader, val_dataloader, test_dataloader, args.no_tuning, args.lr, args.l2, config, args)


    log_stats = {'best_acc': best_acc.cpu().tolist(), 'training_time': training_time, 'num_training_parameters': num_trainable_par, 'num_parameters': num_par}
    
    args.output_dir = config.OUTPUT_DIR
    args.output_dir = os.path.join(args.output_dir, config.DATASET.DATASET)
    with (Path(args.output_dir) / "log.txt").open("a") as f:
        f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":
    linear_probe()
