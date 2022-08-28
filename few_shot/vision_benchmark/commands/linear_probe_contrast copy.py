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
from pathlib import Path

from vision_datasets import DatasetTypes
from vision_benchmark.common.constants import get_dataset_hub
from vision_benchmark.utils import comm, create_logger
from vision_benchmark.evaluation import extract_features, linear_classifier, lr_classifier, multlabel_lr_classifier, construct_dataloader, linear_classifier_contrast
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
    parser.add_argument('--l2', help='(Inverse) L2 regularization strength. This option is only useful when option --no-tuning is True.', default=0.316, type=float)
    parser.add_argument('--lr', help='Test with a specific learning rate. This option is only useful when option --no-tuning is True.', default=0.001, type=float)

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

    n_samples = str(config.DATASET.NUM_SAMPLES_PER_CLASS) if config.DATASET.NUM_SAMPLES_PER_CLASS > 0 else 'full'
    exp_name = 'linear_probe_' + n_samples

    final_output_dir = create_logger(config, exp_name)
    if comm.is_main_process():
        log_arg_env_config(args, config, final_output_dir)

    if args.target == 'azureml':
        from vision_benchmark.common.run_aml import run_aml
        setattr(args, 'target', 'local')
        run_aml(args, _construct_command, 'linear_probing')
        return

    logging.info(f'{config.DATASET.DATASET} is a dataset.')
    train_dataloader, val_dataloader, test_dataloader = construct_dataloader(config)

    # Run full model finetuning
    logging.info('Finetuning with full model. This may take several minutes to hours depending on the size of your data.')
    linear_classifier_contrast(train_dataloader, val_dataloader, test_dataloader, args.no_tuning, args.lr, args.l2, config)
    test_predictions = None  # submission not supported yet
    dataset_info = None

    if args.submit_predictions and dataset_info and test_predictions:
        submit_predictions(test_predictions.tolist(), args.submit_by, config, 'finetuning', dataset_info.type)


if __name__ == '__main__':
    main()
