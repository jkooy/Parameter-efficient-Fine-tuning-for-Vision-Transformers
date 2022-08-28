from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pprint
import time

from timm.data import Mixup
import torch
import torch.nn.parallel
import torch.optim
from torch.cuda.amp import autocast
from torch.utils.collect_env import get_pretty_env_info
from tensorboardX import SummaryWriter

import _init_paths
import models
from core.function import test
from core.loss import build_criterion
from config import config
from config import update_config
from config import save_config
from dataset import build_dataloader
from optim import build_optimizer
from utils.comm import comm
from utils.utils import create_logger
from utils.utils import init_distributed
from utils.utils import resume_checkpoint
from utils.utils import save_checkpoint_on_master
from utils.utils import save_model_on_master
from utils.utils import setup_cudnn
from utils.utils import summary_model_on_master


def parse_args():
    parser = argparse.ArgumentParser(
        description='Finetune classification network')

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


def recycle(iterable):
    """Variant of itertools.cycle that does not save iterates."""
    epoch = 0
    logging.info(f'=> set epoch to {epoch}')
    iterable.sampler.set_epoch(epoch)
    while True:
        for i in iterable:
            yield i
    epoch += 1


def get_schedule(dataset_size):
    if dataset_size < 20_000:
        return [100, 200, 300, 400, 500]
    elif dataset_size < 500_000:
        return [500, 3000, 6000, 9000, 10_000]
    else:
        return [500, 6000, 12_000, 18_000, 20_000]


def get_lr(step, dataset_size, base_lr=0.003):
    """Returns learning-rate for `step` or None at the end."""
    supports = get_schedule(dataset_size)
    # Linear warmup
    if step < supports[0]:
        return base_lr * step / supports[0]
    # End of training
    elif step >= supports[-1]:
        return None
    # Staircase decays by factor of 10
    else:
        for s in supports[1:]:
            if s < step:
                base_lr /= 10
        return base_lr


def check_bit_finetune_setting(config):
    finetune_batch = config.FINETUNE.BATCH_SIZE
    total_batch = config.TRAIN.BATCH_SIZE_PER_GPU * comm.world_size
    if total_batch > finetune_batch:
        print('total batch should less than {} ({} vs {})'
              .format(finetune_batch, total_batch, finetune_batch))
        return False
    elif finetune_batch % total_batch != 0:
        print('total batch should be devided by finetune batch')
        return False

    return True


def main():
    args = parse_args()

    init_distributed(args)
    setup_cudnn(config)

    update_config(config, args)
    if not check_bit_finetune_setting(config):
        return

    final_output_dir = create_logger(
        config, args.cfg, 'bit_finetune')
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

    model = eval('models.' + config.MODEL.NAME + '.get_cls_model')(config)
    model.to(torch.device('cuda'))

    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # copy model file
    summary_model_on_master(model, config, final_output_dir, True)

    best_perf = 0.0
    step = 0
    best_model = True
    optimizer = build_optimizer(config, model)

    best_perf, step = resume_checkpoint(
        model, None, optimizer, config, final_output_dir, False
    )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    criterion = build_criterion(config)
    criterion.cuda()
    criterion_val = build_criterion(config, train=False)
    criterion_val.cuda()

    train_loader = build_dataloader(config, True, args.distributed)
    valid_loader = build_dataloader(config, False, args.distributed)

    finetune_batch = config.FINETUNE.BATCH_SIZE
    total_batch = config.TRAIN.BATCH_SIZE_PER_GPU * comm.world_size
    batch_split = finetune_batch // total_batch
    logging.info('=> batch split: {}'.format(batch_split))

    logging.info('=> start training')
    # for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
    accum_steps = 0
    base_lr = config.FINETUNE.BASE_LR
    num_steps = len(train_loader.dataset)
    aug = config.AUG
    mixup_fn = Mixup(
        mixup_alpha=aug.MIXUP, cutmix_alpha=aug.MIXCUT,
        cutmix_minmax=aug.MIXCUT_MINMAX if aug.MIXCUT_MINMAX else None,
        prob=aug.MIXUP_PROB, switch_prob=aug.MIXUP_SWITCH_PROB,
        mode=aug.MIXUP_MODE, label_smoothing=config.LOSS.LABEL_SMOOTHING,
        num_classes=config.MODEL.NUM_CLASSES
    ) if aug.MIXUP_PROB > 0.0 else None

    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP.ENABLED)
    logging.info('=> train start')
    logging.info('=> switch to train mode')
    optimizer.zero_grad()
    model.train()
    for x, y in recycle(train_loader):
        # train for one epoch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        lr = get_lr(step, num_steps, base_lr)
        if lr is None:
            break

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if mixup_fn:
            x, y = mixup_fn(x, y)

        with autocast(enabled=config.AMP.ENABLED):
            logits = model(x)
            c = criterion(logits, y)

        c_num = float(c.data.cpu().numpy())

        scaler.scale(c / batch_split).backward()
        accum_steps += 1
        accstep = f" ({accum_steps}/{batch_split})" if batch_split > 1 else ""
        if step % config.PRINT_FREQ == 0:
            logging.info(f"[step {step}/{accstep}: loss={c_num:.5f} (lr={lr:.5e})]")

        if accum_steps == batch_split:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1
            accum_steps = 0

            if step % config.FINETUNE.EVAL_EVERY == 0:
                val_start = time.time()
                perf = test(
                    config, valid_loader, model, criterion_val,
                    final_output_dir, tb_log_dir, writer_dict,
                    args.distributed
                )
                logging.info(
                    '=> validate end, duration: {:.2f}s'
                    .format(time.time()-val_start)
                )

                if perf > best_perf:
                    best_perf = perf
                    best_model = True
                else:
                    best_model = False

                save_checkpoint_on_master(
                    model=model,
                    ema_model=None,
                    swa_model=None,
                    distributed=args.distributed,
                    model_name=config.MODEL.NAME,
                    optimizer=optimizer,
                    output_dir=final_output_dir,
                    in_epoch=False,
                    epoch_or_step=step,
                    best_perf=best_perf,
                    update_swa=False
                )
                if best_model:
                    save_model_on_master(
                        model, args.distributed, final_output_dir,
                        'model_best.pth'
                    )

    save_model_on_master(
        model, args.distributed, final_output_dir, 'final_state.pth'
    )

    writer_dict['writer'].close()
    logging.info('=> finish training')


if __name__ == '__main__':
    main()
