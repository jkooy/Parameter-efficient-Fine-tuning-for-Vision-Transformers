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
from tensorboardX import SummaryWriter

import _init_paths
import models
from core.function_clip import train_one_epoch
from core.function_clip import evaluate_retrieval
from core.function_clip import evaluate_zeroshot
from core.loss import build_criterion
from config import config
from config import update_config
from config import save_config
from config import export_deepspeed_config
from dataset import build_dataloader
from optim import build_optimizer
from scheduler import build_lr_scheduler
from utils.comm import comm
from utils.utils import create_logger
from utils.utils import init_distributed
from utils.utils import resume_checkpoint
from utils.utils import save_checkpoint_on_master
from utils.utils import save_model_on_master
from utils.utils import setup_cudnn
from utils.utils import summary_model_on_master

import deepspeed
# For huggingface tokenizers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from test_clip_zeroshot import build_zeroshot_loader

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train clip network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--zeroshot-flag", type=int, default=1)
    parser.add_argument('--zeroshot-dataset',
                        help='zero-shot test dataset (Default: imagenet)',
                        default='imagenet')
    parser.add_argument('--root',
                        help='dataset root (Default: DATASET/)',
                        default='DATASET/')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)

    parser = deepspeed.add_config_arguments(parser)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    init_distributed(args)
    update_config(config, args)
    setup_cudnn(config)

    final_output_dir = create_logger(config, args.cfg, 'clip training')
    tb_log_dir = final_output_dir

    args.total_batch = config.TRAIN.BATCH_SIZE_PER_GPU * comm.world_size
    logging.info(f'=> total batch: {args.total_batch}')

    deepspeed_config_path = os.path.join(final_output_dir, 'ds_config.json')

    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(args.num_gpus))

        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        logging.info("=> saving config into: {}".format(output_config_path))
        save_config(config, output_config_path)

        if config.USE_DEEPSPEED:        
            logging.info("=> saving deepspeed config into : {}".format(deepspeed_config_path))
            export_deepspeed_config(config, args, deepspeed_config_path)

    # synchronize to set deepspeed
    if config.USE_DEEPSPEED:
        comm.synchronize()        
        args.deepspeed = True
        args.deepspeed_config = deepspeed_config_path

    train_loader = build_dataloader(config, True, args.distributed)
    # valid_loader = build_dataloader(config, False, args.distributed)

    vocab_size = train_loader.dataset.tokenize.get_vocab_size()
    eot_token = train_loader.dataset.tokenize.get_eot_token()
    logging.info("=> using vocab size {}".format(vocab_size))
    logging.info("=> using eot_token {}".format(eot_token))
    model = eval('models.' + config.MODEL.NAME + '.get_clip_model')(config, vocab_size = vocab_size, eot_token=eot_token)
    model.to(torch.device('cuda'))

    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # copy model file
    summary_model_on_master(model, config, final_output_dir, True)

    best_perf = 0.0
    begin_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = build_optimizer(config, model)
    best_perf, begin_epoch = resume_checkpoint(
        model, None, optimizer, config, final_output_dir, True
    )    
    
    criterion = build_criterion(config)
    criterion.cuda()
    criterion_val = build_criterion(config, train=False)
    criterion_val.cuda()

    #train_loader = build_dataloader(config, True, args.distributed)
    retrieval_loader = build_dataloader(config, False, False)

    if (args.zeroshot_flag > 0):
        zeroshot_loader = build_zeroshot_loader(config, args)

    logging.info('=> start training')
    steps_per_epoch = len(train_loader.dataset) // args.total_batch
    logging.info(f'=> steps per epoch: {steps_per_epoch}')

    lr_scheduler = build_lr_scheduler(config, optimizer, begin_epoch)

    if config.USE_DEEPSPEED:
        logging.info('=> Use deepspeed')
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler)
    elif args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True
        )
        

    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP.ENABLED)

    logging.info('=> train start')
    logging.info('=> switch to train mode')
    top1_acc = 0.0
    for epoch in range(begin_epoch, config.TRAIN.END_EPOCH):
        head = 'Epoch[{}]:'.format(epoch)
        logging.info('=> {} epoch start'.format(head))

        start = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        logging.info('=> {} train start'.format(head))

        train_one_epoch(config, train_loader, model, criterion, optimizer,
                        epoch, final_output_dir, tb_log_dir, writer_dict,
                        scaler=scaler)
        logging.info(
            '=> {} train end, duration: {:.2f}s'
            .format(head, time.time()-start)
        )

        logging.info('*'*20 + ' retrieval evaluation ' + '*'*20)
        start = time.time()
        i2t_acc, t2i_acc = evaluate_retrieval(
            config, retrieval_loader, model, final_output_dir, tb_log_dir
        )
        logging.info(
            '=> {} retrieval evaluation end, duration: {:.2f}s'
            .format(head, time.time()-start)
        )
        # ======== deepspeed will do lr_scheduler.step() in model.step()========
        if not config.USE_DEEPSPEED:
            lr_scheduler.step(epoch=epoch+1)
            
        logging.info('*'*20 + ' retrieval evaluation ' + '*'*20)

        if (args.zeroshot_flag > 0):
            logging.info('*'*20 + ' zero-shot evaluation ' + '*'*20)
            start = time.time()
            top1_acc = evaluate_zeroshot(
                config, args.zeroshot_dataset, zeroshot_loader, model,
                final_output_dir, tb_log_dir, args.distributed, train_loader.dataset.tokenize
            )
            logging.info(
                '=> {} zeroshot evaluation duration: {:.2f}s'
                .format(head, time.time()-start)
            )
            if top1_acc > best_perf:
                best_perf = top1_acc
                save_model_on_master(
                    model, args.distributed, final_output_dir, 'model_best.pth'
                )

            logging.info(
                '=> {} zeroshot best top1 acc: {:.2f}'.format(head, best_perf)
            )
            logging.info('*'*20 + ' zero-shot evaluation ' + '*'*20)

        if writer_dict and comm.is_main_process():
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('i2t_acc', i2t_acc, global_steps)
            writer.add_scalar('t2i_acc', t2i_acc, global_steps)
            writer.add_scalar('zero-shot top1 acc', top1_acc, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

        if config.TRAIN.LR_SCHEDULER.METHOD == 'timm':
            lr = lr_scheduler.get_epoch_values(epoch+1)[0]
        else:
            lr = lr_scheduler.get_last_lr()[0]
        logging.info(f'=> lr: {lr}')

        save_checkpoint_on_master(
            model=model,
            ema_model=None,
            swa_model=None,
            distributed=args.distributed,
            model_name=config.MODEL.NAME,
            optimizer=optimizer,
            output_dir=final_output_dir,
            in_epoch=True,
            epoch_or_step=epoch,
            best_perf=best_perf,
            update_swa=False
        )

        if config.TRAIN.SAVE_ALL_MODELS and comm.is_main_process():
            save_model_on_master(
                model, args.distributed, final_output_dir, f'model_{epoch}.pth'
            )

        logging.info(
            '=> {} epoch end, duration : {:.2f}s'
            .format(head, time.time()-start)
        )

    save_model_on_master(
        model, args.distributed, final_output_dir, 'final_state.pth'
    )

    writer_dict['writer'].close()
    logging.info('=> finish training')


if __name__ == '__main__':
    main()
