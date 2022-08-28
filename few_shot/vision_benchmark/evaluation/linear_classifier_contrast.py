"""
Linear classifier implemented with Pytorch Linear class
"""
import os
import time
import logging
import pickle
import numpy as np
import sys, json
import random

from torch import nn
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from .metric import get_metric
from .feature import FeatureData, get_model
from ..optim import build_optimizer
from ..evaluation.metric import get_metric
from .criterion import HybridContrastive

from ..common.constants import get_dataset_hub, VISION_DATASET_STORAGE
from ..models import *
from ..datasets import class_map, template_map

from vision_benchmark.datasets import SimpleTokenizer, HFPTTokenizer
from vision_benchmark.evaluation import clip_zeroshot_evaluator

import pdb

from tqdm import tqdm
from vision_datasets import ManifestDataset
from nltk.corpus import wordnet as wn
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')

MULTILABEL_DATASETS = {"voc-2007-classification","chestx-ray8"}

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

        for name, param in self.backbone.named_parameters():
            if name.startswith('transformer') or name.startswith('token_embedding') or name.startswith('ln_final'): # image encoder names: [visual ResidualAttentionBlock]; text encoder names: [transformers ResidualAttentionBlock]
                param.requires_grad = False
                # print(name, param.requires_grad)

            if config.TRAIN.FREEZE_IMAGE_BACKBONE:
                if name.startswith('visual.conv1') or name.startswith('visual.ln_pre') or name.startswith('visual.transformer'): # image encoder names: [visual ResidualAttentionBlock]; text encoder names: [transformers ResidualAttentionBlock]
                    param.requires_grad = False

        # pdb.set_trace()
        input_dim, output_dim = config.MODEL.SPEC.EMBED_DIM, config.DATASET.NUM_CLASSES
        self.optim = None
        self.l2_lambda = l2_lambda
        self.logit_scale = nn.Parameter(torch.ones([]))

    def forward(self, image, text):
        pdtype = image.dtype

        features_image = self.backbone.encode_image(image)
        features_text = self.backbone.encode_text(text)

        # cosine similarity as logits
        T = self.logit_scale.exp()

        return features_image, features_text, T


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
    logging.info(f'Number of trainable params: {pytorch_total_params / 1000000}M.')


    gpu = config.GPUS

    if len(gpu) == 1:
        torch.cuda.set_device(gpu[0])
        model = model.cuda(gpu[0])

    # define loss function (criterion) and optimizer
    if config.LOSS.LOSS == 'contrast':
        criterion = HybridContrastive().cuda(gpu)
    else:
        if config.DATASET.DATASET in MULTILABEL_DATASETS:
            criterion = torch.nn.BCEWithLogitsLoss().cuda(gpu)
        else:
            criterion = torch.nn.CrossEntropyLoss().cuda(gpu)

    optimizer = build_optimizer(config, model)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC

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

    # construct text templates
    class_names = class_map.get(config.DATASET.DATASET)
    if not class_names:
        hub = get_dataset_hub()
        from vision_datasets import Usages
        manifest = hub.create_dataset_manifest(VISION_DATASET_STORAGE, None, config.DATASET.DATASET, usage=Usages.TEST_PURPOSE)
        if manifest:
            class_names = manifest[0].labelmap

    templates = template_map.get(config.DATASET.DATASET, ['a photo of a {}'])

    if config.MODEL.SPEC.TEXT.TOKENIZER == 'clip':
        tokenizer = SimpleTokenizer()
    elif 'hf_' in config.MODEL.SPEC.TEXT.TOKENIZER:
        tokenizer = HFPTTokenizer(pt_name=config.MODEL.SPEC.TEXT.TOKENIZER[3:])
    else:
        tokenizer = None


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

        if target.shape[-1] == 1: 
            target = target[:,0]
        target = target.cuda(config.GPUS[0], non_blocking=True)
        
        texts = []
        for t in target:
            text = class_names[t]
            template = templates[ random.randint(0,len(templates)-1) ]
            texts.append( template.format(text) )
        texts = tokenizer(texts).cuda(config.GPUS[0], non_blocking=True) 

        
        

        # compute gradient and do SGD step
        optimizer.zero_grad()

        features_image, features_text, T = model.forward(images, texts)
        logits_image_text = T * features_image @ features_text.t()
        targets = (target.view(-1, 1) == target.view(1, -1)).float()
        
        loss = criterion(logits_image_text, targets)

        # pdb.set_trace()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        

def validate(val_loader, model, criterion, epoch, config):
    batch_time = AverageMeter()
    losses = AverageMeter()
    metric = get_metric(config.TEST.METRIC)

    if config.TEST.METRIC == "11point_mAP":
        mAP = AverageMeter()
    else:
        top1 = AverageMeter()
        top5 = AverageMeter()


    # Step1: image features
    image_features = []
    image_labels = []

    model.eval()
    with torch.no_grad():
        end = time.time()
        for batch in tqdm(val_loader, f'Extracting image features with model {config.MODEL.NAME}.'):

            images, target = batch[:2]
            if len(config.GPUS) == 1:
                images = images.cuda(config.GPUS[0], non_blocking=True)

            if target.shape[-1] == 1: 
                target = target[:,0]
            target = target.cuda(config.GPUS[0], non_blocking=True)
            

            image_features.append(model.backbone.encode_image(images))
            image_labels.append(target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    image_features = torch.cat(image_features)
    image_labels = torch.cat(image_labels)

    # Step2: text features

    if config.MODEL.SPEC.TEXT.TOKENIZER == 'clip':
        tokenizer = SimpleTokenizer()
    elif 'hf_' in config.MODEL.SPEC.TEXT.TOKENIZER:
        tokenizer = HFPTTokenizer(pt_name=config.MODEL.SPEC.TEXT.TOKENIZER[3:])
    else:
        tokenizer = None
    text_features = extract_text_features_for_current_model(model, config, tokenizer)


    # Step3: evaluation and metrics
    metric = get_metric(config.TEST.METRIC)
    # Normalize image_features
    image_features = F.normalize(image_features)

    # Compute logits
    logits = (100. * image_features @ text_features).softmax(dim=-1)
    result = metric(image_labels.squeeze().cpu().detach().numpy(), logits.cpu().detach().numpy())
    metric_name = metric.__name__

    msg = f'=> TEST: {metric_name} {100 * result:.3f}% '
    logging.info(msg)

    return result



@torch.no_grad()
def extract_text_features_for_current_model(model, config, tokenizer):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    class_names = class_map.get(config.DATASET.DATASET)
    if not class_names:
        hub = get_dataset_hub()
        from vision_datasets import Usages
        manifest = hub.create_dataset_manifest(VISION_DATASET_STORAGE, None, config.DATASET.DATASET, usage=Usages.TEST_PURPOSE)
        if manifest:
            class_names = manifest[0].labelmap

    if config.KNOWLEDGE.WIKITIONARY.USE_DEFINITION:
        
        if not config.KNOWLEDGE.WIKITIONARY.PRE_EXTRACTED:
            wiki_path = config.KNOWLEDGE.WIKITIONARY.WIKI_DB_PATH
            sys.path.append(wiki_path)
            from get_description import resolve_meaning

            wikdict_json_path = os.path.join(wiki_path, 'wik_dict.json') 
            wik_dict = json.load(open(wikdict_json_path, encoding='utf-8'))
        else:
            wiki_path = config.KNOWLEDGE.WIKITIONARY.WIKI_DICT_PATH
            wiki_tsv_path = os.path.join(wiki_path,  'WIKI_' + config.DATASET.DATASET + '.tsv') 
            wiki_anwser_list = json.load(open(wiki_tsv_path, encoding='utf-8'))

            count_has_wiki_knowledge = 0
            wiki_dict = {}
            for k2v in wiki_anwser_list:
                wiki_dict[ k2v['classname'] ] = k2v['wiki']   
                if k2v['wiki']:
                    count_has_wiki_knowledge += 1
            logging.info(f'coverage is {count_has_wiki_knowledge} / {len(wiki_dict)}')
                   
        
    if config.KNOWLEDGE.GPT3.USE_GPT3:
        gpt3_path = config.KNOWLEDGE.GPT3.GPT3_DICT_PATH
        gpt3_tsv_path = os.path.join(gpt3_path,  'GPT3_' + config.DATASET.DATASET + '.tsv') 
        gpt3_anwser_list = json.load(open(gpt3_tsv_path, encoding='utf-8'))

        gpt3_dict = {}
        for k2v in gpt3_anwser_list:
            gpt3_dict[ k2v['classname'] ] = k2v['gpt3']
    
    templates = template_map.get(config.DATASET.DATASET, ['a photo of a {}'])
    
    start = time.time()
    model.to(device)
    model.eval()

    zeroshot_weights = []
    wiki_count, gpt3_count = 0, 0
    for classname in tqdm(class_names, f'Extracting text features with model {config.MODEL.NAME}.'):
        if type(classname) == list: classname = classname[0]

        knowledge_text_list = []
        if config.KNOWLEDGE.WIKITIONARY.USE_DEFINITION:
            try:
                if not config.KNOWLEDGE.WIKITIONARY.PRE_EXTRACTED:
                    knowledge_text = resolve_meaning(classname, wik_dict)
                    if not knowledge_text:
                        knowledge_text_list.append(knowledge_text)
                else:
                    if wiki_dict[classname]:
                        knowledge_text_list.append(wiki_dict[classname])
                        wiki_count += 1
                        print(f'wiki: {wiki_dict[classname]}')
            except:
                knowledge_text = None


        if config.KNOWLEDGE.GPT3.USE_GPT3:

            if config.KNOWLEDGE.AGGREGATION.MEHTOD == 'WIKI_AND_GPT3':
                for knowledge_text in gpt3_dict[classname][:config.KNOWLEDGE.AGGREGATION.NUM_GPT3_ITEMS]:
                    # knowledge_text = gpt3_dict[classname][-1]
                    knowledge_text_list.append(knowledge_text)
                    print(f'gpt3: {knowledge_text}')  
                gpt3_count += 1          
            
            elif config.KNOWLEDGE.AGGREGATION.MEHTOD == 'WIKI_THEN_GPT3' and  len(knowledge_text_list) == 0:
                for knowledge_text in gpt3_dict[classname][:config.KNOWLEDGE.AGGREGATION.NUM_GPT3_ITEMS]:
                    # knowledge_text = gpt3_dict[classname][-1]
                    knowledge_text_list.append(knowledge_text)
                    print(f'gpt3: {knowledge_text}')
                gpt3_count += 1

        knowledge_text_list_aug = []
        for knowledge_text in knowledge_text_list:
            knowledge_text = f' ; {classname} , ' + knowledge_text if knowledge_text is not None else ''
            knowledge_text = ' '.join(word_tokenize(knowledge_text.lower()))
            knowledge_text_list_aug.append(knowledge_text)

        if len(knowledge_text_list_aug) == 0:
            texts = [template.format(classname) for template in templates ]
        else:
            texts = [template.format(classname) + knowledge_text for knowledge_text in knowledge_text_list_aug for template in templates ]

        # pdb.set_trace()
        texts = tokenizer(texts).to(device)
        class_embeddings = model.backbone.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    logging.info(f'=> Feature extraction duration time: {time.time() - start:.2f}s')
    logging.info(f'=> Knowledge source count | wiki_count: {wiki_count} | gpt3_count {gpt3_count} ')

    return zeroshot_weights




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


def linear_classifier_contrast(train_dataloader, val_dataloader, test_dataloader, no_hyperparameter_tuning, lr, l2, config):

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

