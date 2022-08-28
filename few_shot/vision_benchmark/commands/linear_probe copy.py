"""
Linear Probe with sklearn Logistic Regression or linear model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

import numpy as np
import os
import random
from pathlib import Path

from vision_datasets import DatasetTypes
from vision_benchmark.common.constants import get_dataset_hub
from vision_benchmark.utils import comm, create_logger
from vision_benchmark.evaluation import extract_features, linear_classifier, lr_classifier, multlabel_lr_classifier
from vision_benchmark.evaluation import construct_dataloader, full_model_finetune
from vision_benchmark.config import config, update_config
# These 2 lines are a walk-around for "Too many open files error". Refer: https://github.com/pytorch/pytorch/issues/11201
import torch.multiprocessing
from vision_benchmark.common.utils import log_arg_env_config, submit_predictions

torch.multiprocessing.set_sharing_strategy('file_system')

MULTILABEL_DATASETS = {"chestx-ray8"}


def add_linear_probing_args(parser):
    parser.add_argument('--ds', required=False, help='Evaluation dataset configure file name.', type=str)
    parser.add_argument('--model', required=True, help='Evaluation model configure file name', type=str)
    parser.add_argument('--submit-predictions', help='submit predictions and model info to leaderboard.', default=False, action='store_true')
    parser.add_argument('--submit-by', help='Person who submits the results.', type=str)

    parser.add_argument('--target', help='target of run. local or azureml', choices=['local', 'azureml'], default='local')

    parser.add_argument('-c', '--classifier', choices=['logistic', 'linear'], default='logistic', type=str)
    # logistic - sklearn logistic_regression, linear - torch.nn.linear help='Classifier used for linear probe',
    parser.add_argument('--save-feature', help='Flag to save feature or not', default=False, type=str)
    parser.add_argument('--no-tuning', help='No hyperparameter-tuning.', default=False, type=str)
    parser.add_argument('--emulate-zeroshot', help='Emulate zero shot learning.', default=False, type=str)
    parser.add_argument('--l2', help='(Inverse) L2 regularization strength. This option is only useful when option --no-tuning is True.', default=0.316, type=float)
    parser.add_argument('--lr', help='Test with a specific learning rate. This option is only useful when option --no-tuning is True.', default=0.001, type=float)
    parser.add_argument('--run', help='Run id', default=1, type=int)
    parser.add_argument('--fix_seed', help='Fix the random seed. [-1] not fixing the seeds', default=0, type=int)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)    

def load_or_extract_features(args, cfg):
    # Load or extract features.

    n_samples = str(cfg.DATASET.NUM_SAMPLES_PER_CLASS) if cfg.DATASET.NUM_SAMPLES_PER_CLASS > 0 else 'full'
    feature_path = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.DATASET)

    feature_path_dir = Path(feature_path)
    print('=> creating {} ...'.format(feature_path_dir))
    feature_path_dir.mkdir(parents=True, exist_ok=True)

    feature_file = os.path.join(feature_path_dir, "linearprobe_features_" + n_samples + "_" + cfg.MODEL.NAME.replace("/", "") + ".npy")
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
        train_features, train_labels, val_features, val_labels, test_features, test_labels = extract_features(cfg)
        if args.save_feature:
            logging.info("Saving features to a file.")
            with open(feature_file, 'wb') as fwrite:
                np.save(fwrite, train_features)
                np.save(fwrite, train_labels)
                np.save(fwrite, val_features)
                np.save(fwrite, val_labels)
                np.save(fwrite, test_features)
                np.save(fwrite, test_labels)
    logging.info(f'Train size is {train_features.shape[0]} and validation size is {val_features.shape[0]}.')
    logging.info(f'Test size is {test_features.shape[0]}.')
    return train_features, train_labels, val_features, val_labels, test_features, test_labels


def _construct_command(args):
    """Build a commandline from the parsed arguments"""

    cmd = ['vb_linear_probe', '--ds', args.ds, '--model', args.model, '--l2', str(args.l2), '--lr', str(args.lr), '--target', str(args.target)]

    if args.submit_predictions:
        assert args.submit_by
        cmd.extend(['--submit-predictions', '--submit-by', args.submit_by])

    if args.no_tuning:
        cmd.append('--no-tuning')

    return cmd


def main():
    parser = argparse.ArgumentParser(description='Test a classification model, with linear probing.')
    add_linear_probing_args(parser)
    args = parser.parse_args()

    args.cfg = args.ds
    update_config(config, args)
    args.cfg = args.model
    update_config(config, args)
    config.defrost()
    config.NAME = ''
    config.freeze()

    if args.submit_predictions:
        assert args.submit_by

    if args.fix_seed != -1:
        random.seed(args.fix_seed)
        np.random.seed(args.fix_seed)
        torch.manual_seed(args.fix_seed)
        torch.cuda.manual_seed_all(args.fix_seed)

    if args.emulate_zeroshot:
        args.no_tuning = True
        config.defrost()
        config.TRAIN.END_EPOCH = 1
        config.TRAIN.EXTRA_FINAL_TRAIN_EPOCH = 0
        config.DATASET.NUM_SAMPLES_PER_CLASS = 0
        config.TRAIN.EMULATE_ZERO_SHOT = True
        config.freeze()

    n_samples = str(config.DATASET.NUM_SAMPLES_PER_CLASS) if config.DATASET.NUM_SAMPLES_PER_CLASS >= 0 else 'full'
    exp_name = 'linear_probe_' + n_samples

    if config.DATASET.NUM_SAMPLES_PER_CLASS == 1:
        config.defrost()
        config.DATASET.NUM_SAMPLES_PER_CLASS = 2
        config.DATASET.MERGE_TRAIN_VAL_FINAL_RUN = False
        config.freeze()

    if config.DATASET.DATASET == 'patch-camelyon' and config.DATASET.NUM_SAMPLES_PER_CLASS == -1:
        # deal with patch camelyon large dataset (search using 10000-shot subset, final run with the full dataset)
        logging.info(f'Detecting large dataset with {config.DATASET.NUM_SAMPLES_PER_CLASS}-shot.')
        config.defrost()
        config.DATASET.NUM_SAMPLES_PER_CLASS = 10000
        config.freeze()
        logging.info(f'Used the subset ({config.DATASET.NUM_SAMPLES_PER_CLASS}-shot) to train the model.')

    final_output_dir = create_logger(config, exp_name)
    if comm.is_main_process():
        log_arg_env_config(args, config, final_output_dir)

    if args.target == 'azureml':
        from vision_benchmark.common.run_aml import run_aml
        setattr(args, 'target', 'local')
        run_aml(args, _construct_command, 'linear_probing')
        return

    

    # Run linear probe
    if args.classifier == 'logistic':
        train_features, train_labels, val_features, val_labels, test_features, test_labels = load_or_extract_features(args, config)

        logging.info('Linear probe with logistic regression classifier. ')
        logging.info('This may take several minutes to hours depending on the size of your data. You could turn on verbose to see more details.')
        hub = get_dataset_hub()
        dataset_info = hub.dataset_registry.get_dataset_info(config.DATASET.DATASET)
        if config.DATASET.DATASET in MULTILABEL_DATASETS or (dataset_info and dataset_info.type == DatasetTypes.IC_MULTILABEL):
            logging.info(f'{config.DATASET.DATASET} is a multilabel dataset.')
            # multilabel classification
            test_predictions = multlabel_lr_classifier(train_features, train_labels, val_features, val_labels, test_features, test_labels, args.no_tuning, args.l2, config)
        else:
            test_predictions = lr_classifier(train_features, np.ravel(train_labels), val_features, np.ravel(val_labels), test_features, np.ravel(test_labels), args.no_tuning, args.l2, config)
    elif args.classifier == 'linear':
        # logging.info('Linear probe with linear classifier. This may take several minutes to hours depending on the size of your data.')
        # linear_classifier(train_features, train_labels, val_features, val_labels, test_features, test_labels, args.no_tuning, args.lr, args.l2, config)

        train_dataloader, val_dataloader, test_dataloader = construct_dataloader(config)

        # print([x[1] for x in train_dataloader.dataset])
        # print([x[1] for x in val_dataloader.dataset])
        # print(np.asarray([x[1] for x in train_dataloader.dataset]).sum(axis=0))
        # print(np.asarray([x[1] for x in val_dataloader.dataset]).sum(axis=0))
        # import sys
        # sys.exit(0)

        full_model_finetune(train_dataloader, val_dataloader, test_dataloader, args.no_tuning, args.lr, args.l2, config)

        test_predictions = None  # submission not supported yet
        dataset_info = None
    else:
        raise ValueError('Incorrect classifier option.')

    if args.submit_predictions and dataset_info and test_predictions:
        submit_predictions(test_predictions.tolist(), args.submit_by, config, 'linear_probing', dataset_info.type)


if __name__ == '__main__':
    main()
