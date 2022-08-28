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
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
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
from layers.batch_norm import FrozenBatchNorm2d
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


def main():
    args = parse_args()

    init_distributed(args)
    setup_cudnn(config)

    update_config(config, args)

    final_output_dir = create_logger(config, args.cfg, 'swa_finetune')
    tb_log_dir = final_output_dir

    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(num_gpus))

        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        logging.info("=> saving config into: {}".format(output_config_path))
        save_config(config, output_config_path)

    model = eval('models.' + config.MODEL.NAME + '.get_cls_model')(config)

    if config.SWA.FROZEN_BN:
        logging.info('=> froze bn layers')
        FrozenBatchNorm2d.convert_frozen_batchnorm(model)

    model.to(torch.device('cuda'))

    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # copy model file
    summary_model_on_master(model, config, final_output_dir, True)

    best_perf = 0.0
    best_model = False
    step = 0
    optimizer = build_optimizer(config, model)
    best_perf, step = resume_checkpoint(
        model, None, optimizer, config, final_output_dir, False
    )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    swa_model = None
    if config.SWA.ENABLED:
        logging.info('=> init swa model')
        swa_model = AveragedModel(model, device=config.SWA.DEVICE)

    criterion = build_criterion(config)
    criterion.cuda()
    criterion_val = build_criterion(config, train=False)
    criterion_val.cuda()

    train_loader = build_dataloader(config, True, args.distributed)
    valid_loader = build_dataloader(config, False, args.distributed)

    total_batch = config.TRAIN.BATCH_SIZE_PER_GPU * comm.world_size
    logging.info(f'=> total batch: {total_batch}')

    logging.info('=> start training')
    # for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
    steps_per_epoch = len(train_loader.dataset)
    logging.info(f'=> steps per epoch: {steps_per_epoch}')
    steps_update = steps_per_epoch

    if config.TRAIN.LR_SCHEDULER.METHOD == 'CyclicLR':
        cyclic_ratio = 1.0
        if 'CYCLIC_RATIO' in config.TRAIN.LR_SCHEDULER:
            cyclic_ratio = config.TRAIN.LR_SCHEDULER.CYCLIC_RATIO

        steps_update = int(steps_per_epoch * cyclic_ratio)
        logging.info(f'=> update steps_update to {steps_update}')
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config.TRAIN.LR_SCHEDULER.BASE_LR*comm.world_size,
            max_lr=config.TRAIN.LR_SCHEDULER.MAX_LR*comm.world_size,
            step_size_up=1,
            step_size_down=steps_update-1,
            last_epoch=step-1
        )
    elif config.TRAIN.LR_SCHEDULER.METHOD == 'SWALR':
        lr_scheduler = SWALR(
            optimizer,
            swa_lr=config.TRAIN.LR,
            anneal_epochs=steps_update,
        )

    aug = config.AUG
    mixup_fn = Mixup(
        mixup_alpha=aug.MIXUP, cutmix_alpha=aug.MIXCUT,
        cutmix_minmax=aug.MIXCUT_MINMAX if aug.MIXCUT_MINMAX else None,
        prob=aug.MIXUP_PROB, switch_prob=aug.MIXUP_SWITCH_PROB,
        mode=aug.MIXUP_MODE, label_smoothing=config.LOSS.LABEL_SMOOTHING,
        num_classes=config.MODEL.NUM_CLASSES
    ) if aug.MIXUP_PROB > 0.0 else None

    optimizer.step()
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP.ENABLED)

    logging.info('=> test pre-trained model')
    test(
        config, valid_loader, model, criterion,
        final_output_dir, tb_log_dir, writer_dict,
        args.distributed
    )

    logging.info('=> train start')
    logging.info('=> switch to train mode')
    epoch = 0
    model.train()
    for x, y in recycle(train_loader):
        if step >= (steps_per_epoch * config.TRAIN.END_EPOCH):
            break

        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        if mixup_fn:
            x, y = mixup_fn(x, y)

        with autocast(enabled=config.AMP.ENABLED):
            logits = model(x)
            c = criterion(logits, y)

        scaler.scale(c).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        lr = lr_scheduler.get_last_lr()[0]
        if step % config.PRINT_FREQ == 0:
            logging.info(
                f"[step {step}/epoch {epoch}: loss={c:.5f} (lr={lr:.5e})]"
            )

        lr_scheduler.step()
        step += 1

        if (step % steps_update) == 0:
            logging.info(f"[step {step}: loss={c:.5f} (lr={lr:.5e})]")
            val_start = time.time()
            perf = test(
                config, valid_loader, model, criterion,
                final_output_dir, tb_log_dir, writer_dict,
                args.distributed
            )
            logging.info(
                '=> test end, duration: {:.2f}s'
                .format(time.time()-val_start)
            )

            if perf > best_perf:
                best_perf = perf
                best_model = True
            else:
                best_model = False

            if config.SWA.ENABLED:
                logging.info('=> update swa model')
                if config.SWA.DEVICE == 'cpu':
                    swa_model.cuda()
                    swa_model.update_parameters(model)
                    swa_model.cpu()
                else:
                    swa_model.update_parameters(model)

                logging.info('=> test swa model')
                if config.SWA.DEVICE == 'cpu':
                    swa_model.cuda()
                perf_swa = test(
                    config, valid_loader, swa_model, criterion,
                    final_output_dir, tb_log_dir, writer_dict,
                    args.distributed
                )
                if config.SWA.DEVICE == 'cpu':
                    swa_model.cpu()

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
            if perf_swa > best_perf:
                best_perf = perf_swa
                save_model_on_master(
                    swa_model, args.distributed, final_output_dir,
                    'model_best.pth'
                )
            epoch += 1

    if config.SWA.ENABLED:
        model.cpu()
        swa_model.cuda()
        logging.info('=> update bn for swa model')
        update_bn(train_loader, swa_model, device=torch.device('cuda'))
        logging.info('=> test on swa model')
        test(config, valid_loader, swa_model, criterion, final_output_dir,
             tb_log_dir, writer_dict, args.distributed)

    save_model_on_master(
        swa_model, args.distributed, final_output_dir, 'final_state.pth'
    )

    writer_dict['writer'].close()
    logging.info('=> finish training')


if __name__ == '__main__':
    main()
