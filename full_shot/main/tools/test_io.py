from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pprint
import shutil
import sys
import time
from datetime import timedelta

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
from torch.utils.collect_env import get_pretty_env_info
from tensorboardX import SummaryWriter

import _init_paths
import models
from config import config
from config import update_config
from config import save_config
from dataset import build_dataloader
from utils.comm import comm
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='IO performance test')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) \
        if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        print("=> init process group start")
        os.environ['NCCL_BLOCKING_WAIT'] = '1'
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=60))
        comm.synchronize()
        comm.local_rank = args.local_rank
        print("=> init process group end")

    update_config(config, args)
    final_output_dir = create_logger(
        config, args.cfg, 'train')

    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(num_gpus))

        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        logging.info("=> saving config into: {}".format(output_config_path))
        # save overloaded model config in the output directory
        save_config(config, output_config_path)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    last_epoch = config.TRAIN.BEGIN_EPOCH
    train_loader = build_dataloader(config, True, args.distributed)

    logging.info('=> start IO performance testing')
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        logging.info('Epoch: [{}] start'.format(epoch))
        start = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        end = time.time()
        duration = 0.0
        num_samples = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            duration += time.time() - end

            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            num_samples += inputs.size(0)
            end = time.time()

            if batch_idx % config.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Speed: {speed: .1f} samples/s'.format(
                        epoch, batch_idx, len(train_loader),
                        speed=num_samples/duration
                    )
                logging.info(msg)
                end = time.time()
                duration = 0.0
                num_samples = 0

        logging.info('Epoch: [{}] end'.format(epoch))
        logging.info('Epoch: [{}] duration time: {}s'.format(epoch, time.time()-start))

    logging.info('=> finish IO performance testing')


if __name__ == '__main__':
    main()
