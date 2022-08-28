"""
Evaluate image caption model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import logging

import numpy as np

from vision_benchmark.common.utils import log_arg_env_config, submit_predictions
from vision_benchmark.utils import comm, create_logger
from vision_benchmark.common.constants import get_dataset_hub
from vision_benchmark.evaluation import image_caption_generator, image_caption_evaluator
from vision_benchmark.config import config, update_config


def add_image_caption_args(parser):
    parser.add_argument('--ds', required=False, help='Evaluation dataset configure file name.', type=str)
    parser.add_argument('--model', required=True, help='Image caption model configure file name', type=str)
    parser.add_argument('--save-result', help='Flag to save image caption result or not', default=False, action='store_true')

    parser.add_argument('--target', help='target of run. local or azureml', choices=['local', 'azureml'], default='local')
    parser.add_argument('--submit-predictions', help='submit predictions and model info to leaderboard.', default=False, action='store_true')
    parser.add_argument('--submit-by', help='Person who submits the results.', type=str)


def _construct_command(args):
    """Build a commandline from the parsed arguments"""

    cmd = ['vb_image_caption_eval', '--ds', args.ds, '--model', args.model, '--target', args.target]

    if args.submit_predictions:
        assert args.submit_by
        cmd.extend(['--submit-predictions', '--submit-by', args.submit_by])

    return cmd


def load_or_generate_caption_result(args, cfg):

    # Load or generate caption result
    result_file = os.path.join(cfg.DATASET.ROOT, 'image_caption_results_' + cfg.MODEL.NAME.replace('/', '') + '.npy')
    if os.path.exists(result_file):
        logging.info('Loading image caption results from existing files.')
        with open(result_file, 'rb') as fread:
            imcap_predictions = np.load(fread, allow_pickle=True).tolist()
            imcap_targets = np.load(fread, allow_pickle=True).tolist()
    else:
        imcap_predictions, imcap_targets = image_caption_generator(cfg)
        if args.save_result:
            logging.info(f'Saving image caption results to file: {result_file}')
            with open(result_file, 'wb') as fwrite:
                np.save(fwrite, imcap_predictions)
                np.save(fwrite, imcap_targets)

    return imcap_predictions, imcap_targets


def main():
    parser = argparse.ArgumentParser(description='Image caption evaluation script.')
    add_image_caption_args(parser)
    args = parser.parse_args()

    args.cfg = args.ds
    update_config(config, args)
    args.cfg = args.model
    update_config(config, args)
    config.defrost()
    config.NAME = ""
    config.freeze()

    if args.submit_predictions:
        assert args.submit_by

    if args.target == 'azureml':
        from vision_benchmark.common.run_aml import run_aml
        setattr(args, 'target', 'local')
        run_aml(args, _construct_command, 'image_caption')
        return

    final_output_dir = create_logger(config, 'image_caption_eval')
    if comm.is_main_process():
        log_arg_env_config(args, config, final_output_dir)

    imcap_predictions, imcap_results = load_or_generate_caption_result(args, config)
    result = image_caption_evaluator(config, imcap_predictions, imcap_results)
    msg = f'=> TEST results: {result} '
    logging.info(msg)

    hub = get_dataset_hub()
    dataset_info = hub.dataset_registry.get_dataset_info(config.DATASET.DATASET)
    if args.submit_predictions and dataset_info:
        submit_predictions([imcap_predictions, imcap_results], args.submit_by, config, 'image_caption', dataset_info.type)


if __name__ == '__main__':
    main()
