from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pprint

import tensorwatch as tw
from torch.utils.collect_env import get_pretty_env_info

import _init_paths
import models
from config import config
from config import update_config
from utils.modelsummary import get_model_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model summary')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('-o', '--output',
                        help='output model summary directory',
                        required=True,
                        type=str)
    parser.add_argument('-n', '--name',
                        help='model summary file',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    args.distributed = False

    update_config(config, args)
    print("=> collecting env info (might take some time)")
    print("\n" + get_pretty_env_info())
    print(pprint.pformat(args))
    print(config)

    model = eval('models.' + config.MODEL.NAME + '.get_cls_model')(config)
    print(model)
    import torch
    dump_input = torch.rand(
        (1, 3, config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    )
    print(get_model_summary(model, dump_input))

    df = tw.model_stats(
        model, (1, 3, config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    )
    df.to_html(os.path.join(args.output, '{}.html'.format(args.name)))
    df.to_csv(os.path.join(args.output, '{}.csv'.format(args.name)))
    msg = '*'*20 + ' Model summary ' + '*'*20
    print('\n{msg}\n{summary}\n{msg}'.format(msg=msg, summary=df.iloc[-1]))


if __name__ == '__main__':
    main()
