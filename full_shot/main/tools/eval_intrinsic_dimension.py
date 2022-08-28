"""
Benchmark a model against a list of datasets.
"""

import argparse
import os
import subprocess
import logging
import pprint
from torch.utils.collect_env import get_pretty_env_info
import _init_paths
from utils.comm import comm
from utils.utils import create_logger
from config import config
from config import update_config


cfg_files_dataset = {
    "caltech101": "experiments/eval/dataset/caltech101.yaml",
    "cifar10": "experiments/eval/dataset/cifar10.yaml",
    "cifar100": "experiments/eval/dataset/cifar100.yaml",
    "dtd": "experiments/eval/dataset/dtd.yaml",
    "fer2013": "experiments/eval/dataset/fer2013.yaml",
    "fgvc-aircraft-2013b": "experiments/eval/dataset/fgvc-aircraft-2013b.yaml",
    "flower102": "experiments/eval/dataset/flower102.yaml",
    "food101": "experiments/eval/dataset/food101.yaml",
    "gtsrb": "experiments/eval/dataset/gtsrb.yaml",
    "hatefulmemes": "experiments/eval/dataset/hatefulmemes.yaml",
    "mnist": "experiments/eval/dataset/mnist.yaml",
    "patchcamelyon": "experiments/eval/dataset/patchcamelyon.yaml",
    "pet37": "experiments/eval/dataset/pet37.yaml",
    "stanfordcar": "experiments/eval/dataset/stanfordcar.yaml",
    "stl10": "experiments/eval/dataset/stl10.yaml",
    "sun397": "experiments/eval/dataset/sun397.yaml",
    "ucf101": "experiments/eval/dataset/ucf101.yaml",
    "voc2007classification": "experiments/eval/dataset/voc2007classification.yaml",
    # "imagenet": "experiments/eval/dataset/imagenet.yaml",
    }


def parse_args():
    """Parse arguments: a list of dataset names and a model"""
    parser = argparse.ArgumentParser(
        description='Test classification network')

    parser.add_argument('--ds',
                        help='Evaluation dataset configure file name.',
                        type=str)

    parser.add_argument('--model',
                        required=True,
                        help='Evaluation model configure file name',
                        type=str)

    parser.add_argument('--save-feature',
                        help='Flag to save feature or not',
                        default=False,
                        type=str)

    parser.add_argument('--data_dir',
                        help='Flag to save feature or not',
                        default='.',
                        type=str)

    parser.add_argument('--output_dir',
                        help='Flag to save feature or not',
                        default='.',
                        type=str)

    parser.add_argument('--model_ckpt',
                        help='Flag to save feature or not',
                        default='.',
                        type=str)

    #add finetune pipline
    parser.add_argument('--finetune',
                        help='Finetune or not.',
                        default=False,
                        type=str)

    #add hyperparameters tuning
    parser.add_argument('--no-search',
                        help='tuning or not.',
                        default=False,
                        type=str)

    #add learning rate search range
    parser.add_argument('--lr-range',
                        help='learning rate search range.',
                        default="1e-5",
                        type=str)

    #add tune layer_norm mode
    parser.add_argument('--layernorm',
                        help='only tune layer_norm mode.',
                        default=False,
                        type=str)

    #add tune adapter mode
    parser.add_argument('--adapter',
                        help='only tune adapter mode.',
                        default=False,
                        type=str)

    #add LoRA mode
    parser.add_argument('--LoRA',
                        help='LoRA mode.',
                        default=False,
                        type=str)

    #add LoRA fix one mode
    parser.add_argument('--LoRAFix',
                        help='LoRAFix mode.',
                        default=False,
                        type=str)

    #add LoRA adpater mmode
    parser.add_argument('--ladapter',
                        help='LoRAFix mode.',
                        default=False,
                        type=str)

    #intrinsic dimension
    parser.add_argument('--dintrinsic',
                        help='intrinsic dimension mode.',
                        default=False,
                        type=str)

    #layer type
    parser.add_argument('--layerType',
                        help='layer type in measuring intrinsic dimension.',
                        default= "mlp",
                        type=str)

    parser.add_argument('--layernum',
                        help='layer in measuring intrinsic dimension.',
                        type=int)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)


    args_batch = parser.parse_args()

    return args_batch


def run_jobs():
    """
    Queue a list of evaluation jobs.
    """
    args = parse_args()
    
    if args.ds is None:
        datasets = list(cfg_files_dataset.keys())
    else:
        datasets = args.ds.split(",")
    # Check dataset availability
    for dataset_name in datasets:
        if not os.path.exists(dataset_name) and (not os.path.exists(cfg_files_dataset[dataset_name])):
            raise Exception(f"Dataset {dataset_name} does not exist.")

    for dataset_name in datasets:
        if os.path.exists(dataset_name):
            cfg_file_ds = dataset_name
        else:
            cfg_file_ds = cfg_files_dataset[dataset_name]
        cfg_file_model = args.model

        cfg_data_dir = args.data_dir 
        cgf_output_dir = args.output_dir
        cgf_model_ckpt = args.model_ckpt


        cmd1 = ["python", "tools/intrinsic_dimension.py", "--ds", cfg_file_ds, "--model", cfg_file_model, "--no-tuning", args.no_search, "--lr-range", args.lr_range, "--dintrinsic", args.dintrinsic, "--layerType", args.layerType, "DATASET.ROOT", cfg_data_dir, "OUTPUT_DIR", cgf_output_dir, "TEST.MODEL_FILE", cgf_model_ckpt, "MODEL.NUM_CLASSES", '0']  
        # cmd1 = ["python", "tools/finetune_lora_adapter.py", "--ds", cfg_file_ds, "--model", cfg_file_model, "--no-tuning", args.no_search, "--lr-range", args.lr_range, "DATASET.ROOT", cfg_data_dir, "OUTPUT_DIR", cgf_output_dir, "TEST.MODEL_FILE", cgf_model_ckpt, "MODEL.NUM_CLASSES", '0']  
        # cmd2 = ["python", "tools/finetune_lora_fix_one.py", "--ds", cfg_file_ds, "--model", cfg_file_model, "--no-tuning", args.no_search, "--lr-range", args.lr_range, "DATASET.ROOT", cfg_data_dir, "OUTPUT_DIR", cgf_output_dir, "TEST.MODEL_FILE", cgf_model_ckpt, "MODEL.NUM_CLASSES", '0']   
        # cmd3 = ["python", "tools/finetune_lora.py", "--ds", cfg_file_ds, "--model", cfg_file_model, "--no-tuning", args.no_search, "--lr-range", args.lr_range, "DATASET.ROOT", cfg_data_dir, "OUTPUT_DIR", cgf_output_dir, "TEST.MODEL_FILE", cgf_model_ckpt, "MODEL.NUM_CLASSES", '0']
        # cmd4 = ["python", "tools/finetune_adapter.py", "--ds", cfg_file_ds, "--model", cfg_file_model, "--no-tuning", args.no_search, "--lr-range", args.lr_range, "DATASET.ROOT", cfg_data_dir, "OUTPUT_DIR", cgf_output_dir, "TEST.MODEL_FILE", cgf_model_ckpt, "MODEL.NUM_CLASSES", '0']
        # cmd5 = ["python", "tools/finetune_layernorm.py", "--ds", cfg_file_ds, "--model", cfg_file_model, "--no-tuning", args.no_search, "--lr-range", args.lr_range, "DATASET.ROOT", cfg_data_dir, "OUTPUT_DIR", cgf_output_dir, "TEST.MODEL_FILE", cgf_model_ckpt, "MODEL.NUM_CLASSES", '0']
        # cmd6 = ["python", "tools/finetune_attention.py", "--ds", cfg_file_ds, "--model", cfg_file_model, "--no-tuning", args.no_search, "--lr-range", args.lr_range, "DATASET.ROOT", cfg_data_dir, "OUTPUT_DIR", cgf_output_dir, "TEST.MODEL_FILE", cgf_model_ckpt, "MODEL.NUM_CLASSES", '0']
        # cmd6 = ["python", "tools/finetune_bias.py", "--ds", cfg_file_ds, "--model", cfg_file_model, "--no-tuning", args.no_search, "--lr-range", args.lr_range, "DATASET.ROOT", cfg_data_dir, "OUTPUT_DIR", cgf_output_dir, "TEST.MODEL_FILE", cgf_model_ckpt, "MODEL.NUM_CLASSES", '0']
        # cmd7 = ["python", "tools/finetune_attention_position_bias.py", "--ds", cfg_file_ds, "--model", cfg_file_model, "--no-tuning", args.no_search, "--lr-range", args.lr_range, "DATASET.ROOT", cfg_data_dir, "OUTPUT_DIR", cgf_output_dir, "TEST.MODEL_FILE", cgf_model_ckpt, "MODEL.NUM_CLASSES", '0']
        # cmd8 = ["python", "tools/finetune_cswin.py", "--ds", cfg_file_ds, "--model", cfg_file_model, "--no-tuning", args.no_search, "--lr-range", args.lr_range, "DATASET.ROOT", cfg_data_dir, "OUTPUT_DIR", cgf_output_dir, "TEST.MODEL_FILE", cgf_model_ckpt, "MODEL.NUM_CLASSES", '0']
        # cmd9 = ["python", "tools/finetune_lr_group.py", "--ds", cfg_file_ds, "--model", cfg_file_model, "--no-tuning", args.no_search, "--lr-range", args.lr_range, "DATASET.ROOT", cfg_data_dir, "OUTPUT_DIR", cgf_output_dir, "TEST.MODEL_FILE", cgf_model_ckpt, "MODEL.NUM_CLASSES", '0']
        # cmd10 = ["python", "tools/linear_probe.py", "--ds", cfg_file_ds, "--model", cfg_file_model, "--classifier", 'Transformer', "DATASET.ROOT", cfg_data_dir, "OUTPUT_DIR", cgf_output_dir, "TEST.MODEL_FILE", cgf_model_ckpt, "MODEL.NUM_CLASSES", '0']
        # cmd11 = ["python", "tools/linear_probe.py", "--ds", cfg_file_ds, "--model", cfg_file_model, "--classifier", 'logistic', "DATASET.ROOT", cfg_data_dir, "OUTPUT_DIR", cgf_output_dir, "TEST.MODEL_FILE", cgf_model_ckpt, "MODEL.NUM_CLASSES", '0']

        subprocess.run(cmd1)
        # subprocess.run(cmd2)
        # subprocess.run(cmd3)
        # subprocess.run(cmd4)
        # subprocess.run(cmd6)
        # subprocess.run(cmd6)
        # subprocess.run(cmd7)
        # subprocess.run(cmd8)
        # subprocess.run(cmd9)
        # subprocess.run(cmd10)
        # subprocess.run(cmd11)

      

if __name__ == "__main__":
    run_jobs()

#     - python tools/linear_probe.py --ds experiments/eval/dataset/{dataset}.yaml --model experiments/eval/model/swin_t_ssl.yaml DATASET.ROOT {data_dir} OUTPUT_DIR {output_dir}/{arch}/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/lincls/epoch0300/transfer TEST.MODEL_FILE {output_dir}/{arch}/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/checkpoint.pth MODEL.NUM_CLASSES 0