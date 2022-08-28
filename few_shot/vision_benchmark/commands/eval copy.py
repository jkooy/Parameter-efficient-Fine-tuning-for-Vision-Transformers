"""
Benchmark a model against a list of datasets.
"""

import argparse
import logging
import os
import subprocess

from vision_benchmark.commands.linear_probe import add_linear_probing_args
from vision_benchmark.commands.zeroshot_eval import add_zero_shot_args

alexandar_leaderboard = {
    "caltech-101": "resources/datasets/caltech101.yaml",
    "cifar-10": "resources/datasets/cifar10.yaml",
    "cifar-100": "resources/datasets/cifar100.yaml",
    "country211": "resources/datasets/country211.yml",
    "dtd": "resources/datasets/dtd.yaml",
    "eurosat-clip": "resources/datasets/eurosat-clip.yaml",
    "fer-2013": "resources/datasets/fer2013.yaml",
    "fgvc-aircraft-2013b-variants102": "resources/datasets/fgvc-aircraft-2013b.yaml",
    "oxford-flower-102": "resources/datasets/flower102.yaml",
    "food-101": "resources/datasets/food101.yaml",
    "gtsrb": "resources/datasets/gtsrb.yaml",
    "hatefulmemes": "resources/datasets/hateful-memes.yaml",
    "kitti-distance": "resources/datasets/kitti-distance.yaml",
    "mnist": "resources/datasets/mnist.yaml",
    "patch-camelyon": "resources/datasets/patchcamelyon.yaml",
    "oxford-iiit-pets": "resources/datasets/oxford-iiit-pets.yaml",
    "ping-attack-on-titan-plus": "resources/datasets/ping-attack-on-titan-plus.yml",
    "ping-whiskey-plus": "resources/datasets/ping-whiskey-plus.yml",
    "rendered-sst2": "resources/datasets/rendered-sst2.yml",
    "resisc45-clip": "resources/datasets/resisc45-clip.yaml",
    "stanford-cars": "resources/datasets/stanfordcar.yaml",
    # "stl10": "resources/datasets/stl10.yaml",
    # "sun397": "resources/datasets/sun397.yaml",
    # "ucf101": "resources/datasets/ucf101.yaml",
    "voc-2007-classification": "resources/datasets/voc2007classification.yaml",
    "b92-regular-ic-benchmark": "resources/datasets/b92-regular-ic-benchmark.yaml",
    "imagenet-1k": "resources/datasets/imagenet-1k.yaml"
}


def create_parser():
    parser = argparse.ArgumentParser(description='Evaluate a model against a pre-defined set of datasets.')
    subparsers = parser.add_subparsers(dest='evaltrack', required=True)
    linear_probing_parser = subparsers.add_parser('linear_probing')
    add_linear_probing_args(linear_probing_parser)
    zero_shot_parser = subparsers.add_parser('zero_shot')
    add_zero_shot_args(zero_shot_parser)

    return parser


def main():
    """
    Queue a list of evaluation jobs.
    """

    parser = create_parser()
    args = parser.parse_args()

    if args.ds is None:
        datasets = list(alexandar_leaderboard.keys())
    else:
        datasets = args.ds.split(',')

    for dataset_name in datasets:
        logging.info(f'run eval for {dataset_name}')
        if os.path.exists(dataset_name):
            cfg_file_ds = dataset_name
        else:
            cfg_file_ds = alexandar_leaderboard[dataset_name]

        setattr(args, 'ds', cfg_file_ds)
        if args.evaltrack == 'linear_probing':
            from vision_benchmark.commands.linear_probe import _construct_command
        else:
            from vision_benchmark.commands.zeroshot_eval import _construct_command

        cmd = _construct_command(args)
        if args.target == 'azureml':
            subprocess.Popen(cmd)
        else:
            subprocess.run(cmd)


if __name__ == '__main__':
    main()
