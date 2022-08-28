"""
Image caption evaluation
"""
import time
import logging
import pickle

import torch
from torchvision import transforms
from PIL import Image
from vision_evaluation.evaluators import BleuScoreEvaluator, METEORScoreEvaluator, ROUGELScoreEvaluator, CIDErScoreEvaluator, SPICEScoreEvaluator

from ..models import *
from ..common.constants import get_dataset_hub, VISION_DATASET_STORAGE

def image_caption_evaluator(config, image_caption_result, ground_truth):

    evaluators = []
    results = {}
    metrics =  config.TEST.METRIC.split(',')
    if 'Bleu' in metrics:
        evaluators.append(BleuScoreEvaluator())
    if 'METEOR' in metrics:
        evaluators.append(METEORScoreEvaluator())
    if 'ROUGE_L' in metrics:
        evaluators.append(ROUGELScoreEvaluator())
    if 'CIDEr' in metrics:
        evaluators.append(CIDErScoreEvaluator())
    if 'SPICE' in metrics:
        evaluators.append(SPICEScoreEvaluator())
    for evaluator in evaluators:
        evaluator.add_predictions(predictions=image_caption_result, targets=ground_truth)
        report = evaluator.get_report()
        results.update(report)

    return results


def load_custom_image_caption_model(config):
    logging.info(f'=> Loading custom model {config.MODEL.NAME}.')
    torch.device("cuda")

    model = eval(config.MODEL.NAME + '.get_image_caption_model')(config)
    model_file = config.TEST.MODEL_FILE
    logging.info(f'=> load model file: {model_file}')
    ext = model_file.split('.')[-1]
    if ext == 'pth':
        state_dict = torch.load(model_file, map_location="cpu")
    elif ext == 'pkl':
        logging.info('=> load pkl model')
        with open(model_file, 'rb') as f:
            state_dict = pickle.load(f)['model']

        for k, v in state_dict.items():
            state_dict[k] = torch.from_numpy(v)
    else:
        raise ValueError(f'=> Unknown model file, with ext {ext}')
    model.load_state_dict(state_dict)
    return model


def collate_fn(data):
    """
       data: is a list of tuples with (image tensor, caption list, id)
    """
    images, labels, ids = zip(*data)
    return torch.stack(images), labels, ids


def get_model(config):
    model_name = config.MODEL.NAME
    if model_name.startswith('image_caption_'):
        model = load_custom_image_caption_model(config)
        model.forward = model.get_caption
    elif model_name.startswith('faked_'):
        model = faked_image_caption_model.load_faked_image_caption_model(config)
        model.forward = model.get_caption
    else:
        raise ValueError(f'=> Unknown model name: {model_name}')
    return model


def get_dataloader(dataset, batch_size_per_gpu=64, workers=6, pin_memory=True):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None,
        drop_last=False,
        collate_fn=collate_fn
    )
    return data_loader


def get_caption(model, data_loader, config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    start = time.time()
    imcap_predictions = []
    imcap_targets = []

    with torch.no_grad():
        for batch in data_loader:
            x, y = batch[:2]

            # compute output
            if device == torch.device('cuda'):
                x = x.cuda(non_blocking=True)

            model_name = config.MODEL.NAME
            if model_name.startswith('image_caption_'):
                caption_results_per_batch = model(x)
            elif model_name.startswith('faked_'):
                caption_results_per_batch = model(x, y)
            else:
                raise ValueError(f'=> Unknown model name: {model_name}')
            imcap_predictions.extend(caption_results_per_batch)
            imcap_targets.extend(y)

    logging.info(f'=> Generating image caption duration time: {time.time() - start:.2f}s')
    return imcap_predictions, imcap_targets


def image_caption_generator(config):
    transform = transforms.Compose([
        transforms.Resize(config.TRAIN.IMAGE_SIZE, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.INPUT.MEAN, std=config.INPUT.STD),
    ])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(config)
    if model:
        model.to(device)
        model.eval()
    hub = get_dataset_hub()
    dataset_names = set([x['name'] for x in hub.list_data_version_and_types()])
    if config.DATASET.DATASET not in dataset_names:
        raise ValueError(f"Unsupported dataset: {config.DATASET.DATASET}. The supported dataset list is {dataset_names}") 
    from vision_datasets import Usages
    from vision_datasets.pytorch import TorchDataset
    local_temp = config.DATASET.ROOT
    test_set = hub.create_manifest_dataset(VISION_DATASET_STORAGE, local_temp, config.DATASET.DATASET, usage=Usages.TEST_PURPOSE)
    test_dataloader = get_dataloader(TorchDataset(test_set, transform=transform))
    return get_caption(model, test_dataloader, config)
