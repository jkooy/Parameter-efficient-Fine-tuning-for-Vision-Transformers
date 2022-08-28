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
import torchvision
from torch.utils.collect_env import get_pretty_env_info

import _init_paths
import models
from core.function_clip import evaluate_zeroshot
from config import config
from config import update_config
from config import save_config
from dataset import build_dataloader
from dataset.transforms import build_transforms
from utils.comm import comm
from utils.utils import create_logger
from utils.utils import init_distributed
from utils.utils import setup_cudnn
from utils.utils import summary_model_on_master

from dataset.languages import SimpleTokenizer
from dataset.languages import HFPTTokenizer

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train clip network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--zeroshot-dataset',
                        help='zero-shot test dataset (Default: imagenet)',
                        default='imagenet')
    parser.add_argument('--root',
                        help='dataset root (Default: DATASET/)',
                        default='DATASET/')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def build_zeroshot_loader(config, args):
    transforms = build_transforms(config, False)
    if args.zeroshot_dataset == 'imagenet':
        images = torchvision.datasets.ImageFolder(
            os.path.join(args.root, args.zeroshot_dataset, 'val'),
            transform=transforms
        )
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    sampler = torch.utils.data.distributed.DistributedSampler(
        images, shuffle=False
    ) if args.distributed else None

    data_loader = torch.utils.data.DataLoader(
        images,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
        sampler=sampler,
        drop_last=False,
    )

    return data_loader


def main():
    args = parse_args()

    init_distributed(args)
    setup_cudnn(config)

    update_config(config, args)

    final_output_dir = create_logger(config, args.cfg, 'clip test')
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

    #model = eval('models.' + config.MODEL.NAME + '.get_clip_model')(config)
    if config.MODEL.SPEC.TEXT.TOKENIZER == 'clip':
        tokenobj = SimpleTokenizer()
    elif 'hf_' in config.MODEL.SPEC.TEXT.TOKENIZER:
        tokenobj = HFPTTokenizer(pt_name = config.MODEL.SPEC.TEXT.TOKENIZER[3:])
    else:
        tokenobj = None
    vocab_size = tokenobj.get_vocab_size()
    eot_token = tokenobj.get_eot_token()
    
    model = eval('models.' + config.MODEL.NAME + '.get_clip_model')(config, vocab_size = vocab_size, eot_token=eot_token)

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

    data_loader = build_zeroshot_loader(config, args)

    start = time.time()
    logging.info('=> start evaluating')
    evaluate_zeroshot(
        config, args.zeroshot_dataset, data_loader, model, final_output_dir,
        tb_log_dir, args.distributed, tokenobj
    )
    logging.info('=> evaluation duration: {:.2f}s'.format(time.time()-start))
    logging.info('=> finish evaluating')


if __name__ == '__main__':
    main()
