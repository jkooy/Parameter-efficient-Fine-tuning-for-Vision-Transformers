from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pprint
import time

import torch
import torch.nn.parallel
import torch.optim
from torch.utils.collect_env import get_pretty_env_info

import _init_paths
import models
from core.function_clip import evaluate_retrieval
from config import config
from config import update_config
from config import save_config
from dataset import build_dataloader
from utils.comm import comm
from utils.utils import create_logger
from utils.utils import init_distributed
from utils.utils import setup_cudnn
from utils.utils import summary_model_on_master


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train clip network')

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

    init_distributed(args)
    setup_cudnn(config)

    update_config(config, args)

    final_output_dir = create_logger(config, args.cfg, 'clip training')
    tb_log_dir = final_output_dir

    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(args.num_gpus))

        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        logging.info("=> saving config into: {}".format(output_config_path))
        save_config(config, output_config_path)

    model = eval('models.' + config.MODEL.NAME + '.get_clip_model')(config)

    model_file = config.TEST.MODEL_FILE if config.TEST.MODEL_FILE \
        else os.path.join(final_output_dir, 'final_state.pth')
    logging.info('=> load model file: {}'.format(model_file))
    state_dict = torch.load(model_file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(torch.device('cuda'))

    # copy model file
    summary_model_on_master(model, config, final_output_dir, True)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    test_loader = build_dataloader(config, False, False)

    total_batch = config.TRAIN.BATCH_SIZE_PER_GPU * comm.world_size
    logging.info(f'=> total batch: {total_batch}')

    steps_per_epoch = len(test_loader.dataset) // total_batch
    logging.info(f'=> steps per epoch: {steps_per_epoch}')

    # scaler = torch.cuda.amp.GradScaler(enabled=config.AMP.ENABLED)

    start = time.time()
    logging.info('=> start evaluating')
    evaluate_retrieval(
        config, test_loader, model, final_output_dir, tb_log_dir
    )
    logging.info('=> evaluation duration: {:.2f}s'.format(time.time()-start))
    logging.info('=> finish evaluating')


if __name__ == '__main__':
    main()
