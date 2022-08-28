"""
Linear Probe with sklearn Logistic Regression or linear model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

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
from evaluation.trans_classifier import trans_classifier
from evaluation.multi_label import multlabel_lr_classifier
from config import config
from config import update_config

import json
from pathlib import Path

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
                        choices=['logistic', 'linear', 'Transformer'], #logistic - sklearn logistic_regression, linear - torch.nn.linear
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

    parser.add_argument('--l2',
                        help='(Inverse) L2 regularization strength. This option is only useful when option --no-tuning is True.',
                        default=0.316,
                        type=float)

    parser.add_argument('--lr',
                        help='Test with a specific learning rate. This option is only useful when option --no-tuning is True.',
                        default=0.001,
                        type=float)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


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

    final_output_dir = create_logger(config, args.cfg, 'linear_probe')
    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> saving logging info into: {}".format(final_output_dir))

    # Load or extrac features.
    feature_file = os.path.join(config.OUTPUT_DIR, "features_" + config.MODEL.NAME.replace("/", "") +".npy")
    if os.path.exists(feature_file):
        logging.info("Loading features from an existing file.")
        with open(feature_file, 'rb') as fread:
            train_features = np.load(fread)
            train_labels = np.load(fread)
            val_features = np.load(fread)
            val_labels = np.load(fread)
            test_features = np.load(fread)
            test_labels = np.load(fread)
    else:
        train_features, train_labels, val_features, val_labels, test_features, test_labels = extract_features(config)
        if args.save_feature:
            logging.info("Saving features to a file.")
            with open(feature_file, 'wb') as fwrite:
                np.save(fwrite, train_features)
                np.save(fwrite, train_labels)
                np.save(fwrite, val_features)
                np.save(fwrite, val_labels)
                np.save(fwrite, test_features)
                np.save(fwrite, test_labels)
    logging.info(f"Train size is {train_features.shape[0]} and validation size is {val_features.shape[0]} .")
    logging.info(f"Test size is {test_features.shape[0]}.")

    # Run linear probe
    if args.classifier == "logistic":
        logging.info("Linear probe with logistic regression classifier. ")
        logging.info("This may take several minutes to hours depending on the size of your data. You could turn on verbose to see more details.")
        if config.DATASET.DATASET in MULTILABEL_DATASETS:

            print("================== multlabel_lr_classifier CHOSEN ==================")
            # multilabel classification
            best_acc, training_time = multlabel_lr_classifier(train_features, train_labels, val_features, val_labels, test_features, test_labels, args.no_tuning, args.l2, config)
        else:
            best_acc, training_time = lr_classifier(train_features, np.ravel(train_labels), val_features, np.ravel(val_labels), test_features, np.ravel(test_labels), args.no_tuning, args.l2, config)
    elif args.classifier == "linear":
        logging.info("Linear probe with linear classifier. This may take several minutes to hours depending on the size of your data.")
        train_features = train_features.astype(np.float32)
        train_labels = train_labels.squeeze(1).astype(np.int_)
        val_features = val_features.astype(np.float32)
        val_labels = val_labels.squeeze(1).astype(np.int_)
        test_features = test_features.astype(np.float32)
        test_labels = test_labels.squeeze(1).astype(np.int_)
        best_acc1, training_time = linear_classifier(train_features, train_labels, val_features, val_labels, test_features, test_labels, args.no_tuning, args.lr, args.l2, config)
        logging.info(f"Training_time is {training_time}.")
    elif args.classifier == "Transformer":
        logging.info("Linear probe with additional transformer block. This may take several minutes to hours depending on the size of your data.")
        train_features = train_features.astype(np.float32)
        print('shape', np.shape(train_labels))
        train_labels = train_labels.squeeze(1).astype(np.int_)
        val_features = val_features.astype(np.float32)
        val_labels = val_labels.squeeze(1).astype(np.int_)
        test_features = test_features.astype(np.float32)
        test_labels = test_labels.squeeze(1).astype(np.int_)
        best_acc1, training_time = trans_classifier(train_features, train_labels, val_features, val_labels, test_features, test_labels, args.no_tuning, args.lr, args.l2, config)
        logging.info(f"Training_time is {training_time}.")

    else:
        raise Exception("Incorrect classifier option.")

    log_stats = {'best_acc': best_acc, 'training_time': training_time}
    
    args.output_dir = config.OUTPUT_DIR
    args.output_dir = os.path.join(args.output_dir, config.DATASET.DATASET)
    with (Path(args.output_dir) / "log.txt").open("a") as f:
        f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":
    linear_probe()
