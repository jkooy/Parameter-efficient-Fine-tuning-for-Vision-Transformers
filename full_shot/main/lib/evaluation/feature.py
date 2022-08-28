"""
This module extract features with model to be evaluated and given dataset.
"""
from numpy.core.fromnumeric import transpose
from numpy.lib.shape_base import split
import os
import time
import logging
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import timm
import clip
import torch
from torch import nn
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import torchvision.models
import models
from dataset.languages.simple_tokenizer import SimpleTokenizer

# from pytorch_pretrained_vit import ViT

from PIL import Image
#########################################
# The following 2 lines are to solve PIL "IOError: image file truncated" with big images. 
# Refer to https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#########################################

class EvalModel(nn.Module):
    def __init__(self, model_cls):
        super().__init__()
        for param in model_cls.parameters():
            param.requires_grad = False
        self.feature_model = nn.Sequential(*list(model_cls.children())[:-1])

    def forward(self, x):
        features = self.feature_model(x)
        return features


class FeatureData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx]


def get_dataloader(dataset, val_split=0, batch_size_per_gpu=64, workers=6, pin_memory=True):
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
        logging.info(model)
    # elif ext == 'npz':
    #     model = ViT('B_32_imagenet1k', pretrained=True)
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
    elif model_name == "vit_b_32":
        model = ViT('B_32_imagenet1k', pretrained=True)
    elif model_name in timm.list_models(pretrained=True):
        model = timm.create_model(model_name, pretrained=True)
        if model_name.startswith("efficientnet"):
            model = EvalModel(model)
        elif model_name.startswith("vit"):
            model.forward = model.forward_features
        else:
            raise Exception("Please define Timm feature extraction model.")
        logging.info(f"Using Timm pretrained model {model_name}")
    elif model_name in clip.available_models():
        model, _ = clip.load(model_name, jit=False)
        if feature_type == "image":
            model.forward = model.encode_image
        elif feature_type == "text":
            model.forward = model.encode_text
        else:
            raise Exception("Incorrect model type.")
        logging.info(f"Using CLIP pretrained model {model_name}")
    elif model_name in dir(models):
        model = load_oneclassification_model(config)
        if model_name == "clip_openai":
            model.forward = model.encode_image
        else:
            model.forword = model.forward_features
    else:
        raise Exception("Wrong model name.")
    return model


def extract_feature(data_loader, config, feature_type="image"):
    model = get_model(config, feature_type=feature_type)
    logging.info(f"Extracting features with evaluated model {config.MODEL.NAME}.")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    start = time.time()
    model.to(device)
    model.eval()

    all_features = []
    all_labels = []
    with torch.no_grad():
        for _, (x, y) in enumerate(data_loader):
            # compute output
            if device == torch.device("cuda"):
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
            outputs = model(x)
            all_features.append(outputs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    features =  np.concatenate((all_features))
    labels = np.concatenate((all_labels))
    logging.info('=> Feature extraction duration time: {:.2f}s'.format(time.time()-start))
    return np.reshape(features, (features.shape[0], -1)), np.reshape(labels, (labels.shape[0], -1))


def extract_features(config):
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

    train_features, train_labels = extract_feature(train_dataloader, config)
    val_features, val_labels = extract_feature(val_dataloader, config)
    test_features, test_labels = extract_feature(test_dataloader, config)

    return train_features, train_labels, val_features, val_labels, test_features, test_labels