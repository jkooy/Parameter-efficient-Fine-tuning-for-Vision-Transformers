import os
import json
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models
import torchvision.datasets
import clip
from dataset.languages.simple_tokenizer import SimpleTokenizer
import logging
from vision_datasets import DatasetHub
from vision_datasets import ManifestDataset
import pathlib
import numpy as np
import sys
from collections import Counter
import math
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import torchvision.models
import torchvision.datasets
import torch.nn.functional as F
from .metric import get_metric

VISION_DATASET_STORAGE = 'https://irisdatasets.blob.core.windows.net/share'
def get_dataset_hub():
    vision_dataset_json = (pathlib.Path(__file__).resolve().parents[1] / 'resources' / 'datasets' / 'vision_datasets.json').read_text()
    hub = DatasetHub(vision_dataset_json)

    return hub

def multilabel_to_vec(indices, n_classes):
    vec = np.zeros(n_classes)
    for x in indices:
        vec[x] = 1
    return vec


def multiclass_to_int(indices):
    return indices[0]

def get_dataloader(dataset, val_split=0.0, batch_size_per_gpu=64, workers=6, pin_memory=True):
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
            # this implementation does not generate class-balanced splits.
            # train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)

            # quick fetch labels without accessing images / transformations
            def quick_fetch_labels(dataset):
                dataset_info = dataset.dataset_info
                dataset_manifest = dataset.dataset.dataset_manifest
                from vision_datasets import DatasetTypes
                if dataset_info.type == DatasetTypes.IC_MULTILABEL:
                    labels = [multilabel_to_vec(x.labels, len(dataset.labels)) for x in dataset_manifest.images]
                elif dataset_info.type == DatasetTypes.IC_MULTICLASS:
                    labels = [multiclass_to_int(x.labels) for x in dataset_manifest.images]
                else:
                    raise NotImplementedError
                return np.asarray(labels)

            logging.info('Quick fetch label starts.')
            labels = quick_fetch_labels(dataset)
            logging.info('Quick fetch label finished.')
            # logging.info('Full fetch label starts.')
            # labels_all_fetch = np.asarray([x[1] for x in dataset])
            # logging.info('Full fetch label finished.')
            # assert (labels == labels_all_fetch).all()
            # logging.info('Quick fetch label same as full fetch.')

            # FIX: class-balanced split generation
            if len(labels.shape) == 1:
                # single-class IC datasets
                cls_to_count = Counter(labels)
                val_indices = []

                for label in cls_to_count:
                    n_samples = math.ceil(cls_to_count[label] * val_split)
                    samples = np.where(labels == label)[0][:n_samples]      # TODO: not doing random. confirm that it is unnecessary
                    val_indices.append(samples)
                val_idx = set(np.concatenate(val_indices).tolist())
                train_idx = set(list(range(len(dataset)))) - val_idx
                train_idx, val_idx = list(train_idx), list(val_idx)
            elif len(labels.shape) == 2:
                # multi-class IC datasets
                val_target_count = np.ceil(np.sum(labels, axis=0) * val_split)
                next_targets = np.where(val_target_count > 0)[0]
                val_idx = []

                while next_targets.size > 0:
                    target_cls = next_targets[0]
                    next_sample = np.where(labels[:, target_cls] > 0)[0][0]
                    val_idx.append(next_sample)
                    val_target_count -= labels[next_sample]
                    labels[next_sample] = 0
                    next_targets = np.where(val_target_count > 0)[0]

                val_idx = np.asarray(val_idx).tolist()
                train_idx = set(list(range(len(dataset)))) - set(val_idx)
                train_idx = list(train_idx)
            else:
                raise NotImplementedError

            # val_idx, train_idx = np.split(list(range(len(dataset))), [int(len(dataset)*val_split)])
            # train_idx, val_idx = [x.tolist() for x in (train_idx, val_idx)]
            return {'train': Subset(dataset, train_idx), 'val': Subset(dataset, val_idx)}

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

        
def construct_dataloader(config, feature_type="image", test_split_only=False):

    logging.info('no center crop')
    transform_clip = transforms.Compose([
        transforms.Resize(config.TRAIN.IMAGE_SIZE, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.INPUT.MEAN, std=config.INPUT.STD),
    ])

    if config.DATASET.DATASET == 'chestx-ray8':
        from .dataset import ChestXRay8
        if not test_split_only:
            train_dataloader, val_dataloader = get_dataloader(ChestXRay8(config.DATASET.ROOT, image_set='train', transform=transform_clip), val_split=0.2)
        test_dataloader = get_dataloader(ChestXRay8(config.DATASET.ROOT, image_set='test', transform=transform_clip))
    else:
        from vision_datasets import Usages, DatasetTypes
        from vision_datasets.pytorch import TorchDataset
        hub = get_dataset_hub()
        dataset_names = set([x['name'] for x in hub.list_data_version_and_types()])
        if config.DATASET.DATASET in dataset_names:
            vision_dataset_storage = 'https://irisdatasets.blob.core.windows.net/share'
            local_temp = config.DATASET.ROOT

            # return [manifest, dataset_info, downloader_resources]
            results = hub.create_dataset_manifest(vision_dataset_storage, local_temp, config.DATASET.DATASET, usage=Usages.TEST_PURPOSE)
            if results:
                test_set, test_set_dataset_info, _ = results
            logging.info(f'Test size is {len(test_set.images)}.')
            
            # re-define transform_clip to organize the labels
            if test_set_dataset_info.type == DatasetTypes.IC_MULTILABEL:
                previous_transform = transform_clip

                def transform_clip(x, y):
                    test_set_ = ManifestDataset(test_set_dataset_info, test_set)
                    return (previous_transform(x), multilabel_to_vec(y, len(test_set_.labels)))
            elif test_set_dataset_info.type == DatasetTypes.IC_MULTICLASS:
                previous_transform = transform_clip

                def transform_clip(x, y):
                    return (previous_transform(x), multiclass_to_int(y))

            test_dataloader = get_dataloader(TorchDataset(   ManifestDataset(test_set_dataset_info, test_set), transform=transform_clip) )
            # download train/val split only if test_split_only is False
            if not test_split_only:
                train_set_results = hub.create_dataset_manifest(vision_dataset_storage, local_temp, config.DATASET.DATASET, usage=Usages.TRAIN_PURPOSE)
                if train_set_results:
                    train_set, train_set_dataset_info, _ = train_set_results

                val_set = None
                val_set_results = hub.create_dataset_manifest(vision_dataset_storage, local_temp, config.DATASET.DATASET, usage=Usages.VAL_PURPOSE)
                if val_set_results:
                    val_set, val_set_dataset_info, _ = val_set_results

                    
                val_split=0.2
                train_dataloader, val_dataloader = get_dataloader(TorchDataset( ManifestDataset(train_set_dataset_info, train_set), transform=transform_clip), val_split=val_split)
                logging.info(f'Val split from Train set: Train size is {len(train_set.images)*(1-val_split)}, and validation size is {len(train_set.images)*val_split}.')
        else:
            if not test_split_only:
                if config.DATASET.VAL_SET:
                    train_dataloader = get_dataloader(torchvision.datasets.ImageFolder(os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET), transform=transform_clip))
                    val_dataloader = get_dataloader(torchvision.datasets.ImageFolder(os.path.join(config.DATASET.ROOT, config.DATASET.VAL_SET), transform=transform_clip))
                else:
                    train_dataloader, val_dataloader = get_dataloader(torchvision.datasets.ImageFolder(os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET), transform=transform_clip),
                                                                      val_split=0.2)
            test_dataloader = get_dataloader(torchvision.datasets.ImageFolder(os.path.join(config.DATASET.ROOT, config.DATASET.TEST_SET), transform=transform_clip))

    return train_dataloader, val_dataloader, test_dataloader


class Voc2007Classification(torch.utils.data.Dataset):
    def __init__(self, data_root, image_set="train", transform=None):
        """
        Pascal voc2007 training/validation data: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
        test data: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
        """
        self.data_root = self._update_path(data_root, image_set)
        self.transform = transform
        self.labels = self._read_annotation(image_set)
        self.images = list(self.labels.keys())

    def _update_path(self, data_root, image_set):
        if image_set == "train" or image_set == "val":
            data_root = os.path.join(data_root, "train/VOCdevkit/VOC2007")
        elif image_set == "test":
            data_root = os.path.join(data_root, "test/VOCdevkit 2/VOC2007")
        else:
            raise Exception("Incorrect image set!")
        return data_root

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, 'JPEGImages/'+self.images[index]+'.jpg')
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        label = self.labels[self.images[index]]
        label = torch.LongTensor(label)
        return image,label

    def __len__(self):
        return len(self.images)

    def _read_annotation(self, image_set="train"):
        """
        Annotation interpolation, refer to: 
        http://host.robots.ox.ac.uk/pascal/VOC/voc2007/htmldoc/voc.html#SECTION00093000000000000000
        """
        object_categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', \
                            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', \
                            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        annotation_folder = os.path.join(self.data_root, "ImageSets/Main/")
        files = [file_name for file_name in os.listdir(annotation_folder) if file_name.endswith("_"+image_set+".txt")]
        labels_all = dict()
        for file_name in files:
            label_str = file_name.split("_")[0]
            label_int = object_categories.index(label_str)
            with open(annotation_folder+"/"+file_name, "r") as fread:
                for line in fread.readlines():
                    index = line[:6]
                    if index not in labels_all.keys():
                        labels_all[index] = [0]*len(object_categories)
                    flag = 1
                    if line[7:9] and int(line[7:9]) != 1:
                        flag = -1
                    if flag == 1:
                        labels_all[index][label_int] = 1
        return labels_all


class HatefulMemes(torch.utils.data.Dataset):    
    def __init__(self, data_root, image_set="train", transform=None, tokenizer=None, context_length=77):
        """
        Facebook Hateful Memes: Phase 1 dataset: https://www.drivendata.org/competitions/64/hateful-memes/data/
        """
        self.data_root = data_root
        self.transform = transform
        self.images = self._read_annotation(image_set)
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.images[index]["image_file"])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        label = self.images[index]["label"]
        label = torch.tensor(label)
        text = self.images[index]["text"]
        if self.tokenizer:
            text = self.tokenizer(text, context_length=self.context_length)[0,:]
        return image, label

    def __len__(self):
        return len(self.images)

    def _read_annotation(self, image_set="train"):
        """
        Annotation interpolation, refer to: 
        https://www.drivendata.org/competitions/64/hateful-memes/page/206/
        """
        if image_set == "train":
            label_file = os.path.join(self.data_root, "train.jsonl")
        elif image_set == "val":
            label_file = os.path.join(self.data_root, "dev_seen.jsonl")
        else:
            raise Exception(f"Incorrect image_set value: {image_set}!")
        image_records = []
        with open(label_file, "r") as fread_file:
            for line in fread_file.readlines():
                record = json.loads(line)
                image_records.append({"image_file":record["img"], "text":record["text"], "label": record["label"]})
        return image_records



class ChestXRay8(torch.utils.data.Dataset):
    def __init__(self, data_root, image_set="train", transform=None):
        """
        ChestX-ray dataset: https://paperswithcode.com/dataset/chestx-ray8
        """
        self.data_root = data_root
        self.transform = transform
        self.image_set = image_set
        self.labels = self._read_annotation()
        self.images = self._read_split_file()

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, "images", self.images[index])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        label = self.labels[self.images[index]]
        label = torch.LongTensor(label)
        return image,label

    def __len__(self):
        return len(self.images)

    def _read_split_file(self):
        if self.image_set == "train":
            split_file = "train_val_list.txt"
        elif self.image_set == "test":
            split_file = "test_list.txt"
        else:
            raise Exception("Incorrect image set!")
        file_list = []
        with open(os.path.join(self.data_root, split_file), "r") as fread:
            for line in fread.readlines():
                file_list.append(line.replace("\n", ""))
        return file_list

    def _read_annotation(self):
        """
        Annotation interpolation, refer to: 
        http://host.robots.ox.ac.uk/pascal/VOC/voc2007/htmldoc/voc.html#SECTION00093000000000000000
        """
        object_categories = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',\
            'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', \
            'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding']
        annotation_file = os.path.join(self.data_root, "Data_Entry_2017_v2020.csv")
        image2labels = dict()
        with open(annotation_file, "r") as fread:
            for i, line in enumerate(fread.readlines()):
                if i==0:
                    continue
                image_name, labels_raw, _, _, _, _, _, _, _, _, _= line.split(",")
                labels = labels_raw.split('|')
                labels_int = [0]*(len(object_categories) - 1)
                for label in labels:
                    if label == "No Finding":
                        continue
                    labels_int[object_categories.index(label)] = 1
                image2labels[image_name] = labels_int
        return image2labels
