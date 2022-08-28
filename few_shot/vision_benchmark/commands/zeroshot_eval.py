"""
Zero shot evaluation.
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
from vision_benchmark.datasets import SimpleTokenizer, HFPTTokenizer
from vision_benchmark.evaluation import extract_features, extract_text_features, clip_zeroshot_evaluator
from vision_benchmark.config import config, update_config


def add_zero_shot_args(parser):
    parser.add_argument('--ds', required=False, help='Evaluation dataset configure file name.', type=str)
    parser.add_argument('--model', required=True, help='Clip model configure file name', type=str)

    parser.add_argument('--target', help='target of run. local or azureml', choices=['local', 'azureml'], default='local')

    parser.add_argument('--submit-predictions', help='submit predictions and model info to leaderboard.', default=False, action='store_true')
    parser.add_argument('--submit-by', help='Person who submits the results.', type=str)
    parser.add_argument('--save-feature', help='save feature or not.', default=False, action='store_true')
    parser.add_argument('--text_feature_only', help='consider text feature or not.', default=False, action='store_true')

    parser.add_argument('--save-predictions', help='save predictions logits for analysis.', default=False, action='store_true')


    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)    

def _construct_command(args):
    """Build a commandline from the parsed arguments"""

    cmd = ['vb_zero_shot_eval', '--ds', args.ds, '--model', args.model, '--target', args.target]

    if args.submit_predictions:
        assert args.submit_by
        cmd.extend(['--submit-predictions', '--submit-by', args.submit_by])

    return cmd


def load_or_extract_features(args, cfg):
    if cfg.MODEL.SPEC.TEXT.TOKENIZER == 'clip':
        tokenizer = SimpleTokenizer()
    elif 'hf_' in cfg.MODEL.SPEC.TEXT.TOKENIZER:
        tokenizer = HFPTTokenizer(pt_name=cfg.MODEL.SPEC.TEXT.TOKENIZER[3:])
    else:
        tokenizer = None

    # Load or extract image features.
    feature_file = os.path.join(cfg.DATASET.ROOT, 'zeroshot_features_' + cfg.MODEL.NAME.replace('/', '') + f'_wiki_{cfg.KNOWLEDGE.WIKITIONARY.USE_DEFINITION}' + f'_gpt3_{cfg.KNOWLEDGE.GPT3.USE_GPT3}' + '.npy')
    logging.info(f'feature_file: {feature_file}')
    if os.path.exists(feature_file):
        logging.info('Loading features from existing files.')
        with open(feature_file, 'rb') as fread:
            image_features = np.load(fread)
            text_features = np.load(fread)
            image_labels = np.load(fread)
    else:
        image_features, image_labels = extract_features(cfg, test_split_only=True)
        text_features = extract_text_features(cfg, tokenizer, args)
        if args.save_feature:
            logging.info('Saving features to a file.')
            with open(feature_file, 'wb') as fwrite:
                np.save(fwrite, image_features)
                np.save(fwrite, text_features)
                np.save(fwrite, image_labels)
    logging.info(f'Test size is {image_features.shape[0]}.')

    return image_features, text_features, image_labels

def load_or_extract_text_features(args, cfg):
    if cfg.MODEL.SPEC.TEXT.TOKENIZER == 'clip':
        tokenizer = SimpleTokenizer()
    elif 'hf_' in cfg.MODEL.SPEC.TEXT.TOKENIZER:
        tokenizer = HFPTTokenizer(pt_name=cfg.MODEL.SPEC.TEXT.TOKENIZER[3:])
    else:
        tokenizer = None

    # Load or extract image features.
    feature_file = os.path.join(cfg.DATASET.ROOT, 'zeroshot_text_features_' + cfg.MODEL.NAME.replace('/', '') + f'_wiki_{cfg.KNOWLEDGE.WIKITIONARY.USE_DEFINITION}' + f'_gpt3_{cfg.KNOWLEDGE.GPT3.USE_GPT3}' + '.npy')
    logging.info(f'feature_file: {feature_file}')
    if os.path.exists(feature_file):
        logging.info('Loading features from existing files.')
        with open(feature_file, 'rb') as fread:
            text_features = np.load(fread)
    else:
        wiki_dict, gpt3_dict = extract_text_features(cfg, tokenizer, args)
    logging.info(f'Test size is {len(wiki_dict)}.')

    return wiki_dict, gpt3_dict



def main():
    parser = argparse.ArgumentParser(description='Zero-shot evaluation script.')
    add_zero_shot_args(parser)
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
        run_aml(args, _construct_command, 'zero_shot')
        return

    exp_name = 'zeroshot_eval_' + f'wiki_{config.KNOWLEDGE.WIKITIONARY.USE_DEFINITION}_wnh_{config.KNOWLEDGE.WORDNET.USE_HIERARCHY}_wnd_{config.KNOWLEDGE.WORDNET.USE_DEFINITION}_gpt3_{config.KNOWLEDGE.GPT3.USE_GPT3}'
    exp_name += f'agg_{config.KNOWLEDGE.AGGREGATION.MEHTOD}_gpt3count_{config.KNOWLEDGE.AGGREGATION.NUM_GPT3_ITEMS}'
    final_output_dir = create_logger(config, exp_name)

    if comm.is_main_process():
        log_arg_env_config(args, config, final_output_dir)

    if args.text_feature_only:
        wiki_dict, gpt3_dict = load_or_extract_text_features(args, config)

    else:
        image_features, text_features, image_labels = load_or_extract_features(args, config)
        result, test_predictions, metric = clip_zeroshot_evaluator(image_features, text_features, image_labels, config)
        msg = f'=> TEST: {metric} {100 * result:.3f}% '
        logging.info(msg)

    if args.save_predictions:
        import json
        max_label_indices = test_predictions.max(-1)[-1]
        
        results_dict = {'result': result, 'test_predictions': test_predictions.cpu().data.numpy().tolist(), 'image_labels': image_labels.tolist(), 'max_label_indices': max_label_indices.cpu().data.numpy().tolist()}
        # import pdb; pdb.set_trace()
        json_string = json.dumps(results_dict)
        
        with open( os.path.join(config.OUTPUT_DIR, f'{config.DATASET.DATASET}_{exp_name}_predictions.tsv' ) , 'w') as outfile:
            outfile.write(json_string)



    hub = get_dataset_hub()
    dataset_info = hub.dataset_registry.get_dataset_info(config.DATASET.DATASET)
    if args.submit_predictions and dataset_info:
        submit_predictions(test_predictions.tolist(), args.submit_by, config, 'zero_shot', dataset_info.type)


if __name__ == '__main__':
    main()
