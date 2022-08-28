"""
Benchmark a model against a list of datasets.
"""

import argparse
import os
import subprocess
import logging
import pprint
from torch.utils.collect_env import get_pretty_env_info
import _init_paths
from utils.comm import comm
from utils.utils import create_logger
from config import config
from config import update_config


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


def parse_args():
    """Parse arguments: a list of dataset names and a model"""
    parser = argparse.ArgumentParser(
        description='Test classification network')

    parser.add_argument('--ds',
                        help='Evaluation dataset configure file name.',
                        type=str)

    parser.add_argument('--model',
                        required=True,
                        help='Evaluation model configure file name',
                        type=str)

    parser.add_argument('--save-feature',
                        help='Flag to save feature or not',
                        default=False,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args_batch = parser.parse_args()

    return args_batch


def run_jobs():
    """
    Queue a list of evaluation jobs.
    """
    args = parse_args()
    
    if args.ds is None:
        datasets = list(cfg_files_dataset.keys())
    else:
        datasets = args.ds.split(",")
    # Check dataset availability
    for dataset_name in datasets:
        if not os.path.exists(dataset_name) and (not os.path.exists(cfg_files_dataset[dataset_name])):
            raise Exception(f"Dataset {dataset_name} does not exist.")

    for dataset_name in datasets:
        if os.path.exists(dataset_name):
            cfg_file_ds = dataset_name
        else:
            cfg_file_ds = cfg_files_dataset[dataset_name]
        cfg_file_model = args.model
        cmd = ["python", "tools/linear_probe.py", "--ds", cfg_file_ds, "--model", cfg_file_model]
        subprocess.run(cmd)
      

if __name__ == "__main__":
    run_jobs()

