# Introdution
The project aims at providing a training tool for classification task pre-training using PyTorch. The project is developed using **PyTorch==1.6.0**. We encourage to use the newest PyTorch version.

We provide a [MODEL ZOO](MODEL_ZOO.md), which collects a set of pre-trained models.

## Highlights
- Support multi dataset (ImageNet-1k, ImageNet-2k, coco), and we will support more datasets in the feature.
- Support folder and tsv dataset format.
- Support label smoothing.
- Support mixup data augmentation.
- Support cutmix data augmentation.
- Support BiT finetuning strategy.
- Support directly submitting jobs to AML or philly using [vislib-toolbox](https://dev.azure.com/vislib/vislib-toolbox).
- Support EMA (Exponential Moving Average).
- Support Auto Mixed Precision training.
- All the experiments are controlled by a config file system.

# Quick start
## Installation
Assuming that you have installed PyTroch and TorchVision, if not, please follow the [officiall instruction](https://pytorch.org/) to install them firstly.
Intall the dependencies using cmd:

``` sh
python -m pip install -r requirements.txt --user -q
```

## Data preparation
The datasets are storaged in the blob container of [*pubdatasets*](https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/22a92cb3-4446-4402-957b-227960e64918/resourceGroups/BatchAIGrp/providers/Microsoft.Storage/storageAccounts/pubdatasets/overview).
### On local machine
Please download the target dataset (imagenet, imagenet-tsv, imagenet22k-tsv, ...) under the directory of *DATASET* in your local machine. Your directory tree should like this. Please note that you do not need download all the dataset, and you just need to download your target dataset.
``` sh
OneClassification
|-DATASET
  |-imagenet
  |-imagenet-tsv
  |-imagenet22k-tsv
```
### On philly or AML
Please add the target blob dataset in your vislib-toolbox config.
For example:

``` sh
$vislib-job add-data imagenet --account pubdatasets --secret SECRET --container imagenet
```

## Run
Each experiment is defined by a yaml config file, which is saved under the directory of `experiments`. The directory of `experiments` has a tree structure like this:

``` sh
experiments
|-{DATASET_A}
| |-{ARCH_A}
| |-{ARCH_B}
|-{DATASET_B}
| |-{ARCH_A}
| |-{ARCH_B}
|-{DATASET_C}
| |-{ARCH_A}
| |-{ARCH_B}
|-...
```

We provide a `run.sh` script for running or submitting jobs in local machine, philly or AML.

``` sh
Usage: run.sh [run_options]
Options:
  -g|--gpus <1> - number of gpus to be used
  -t|--job-type <aml> - job type (train|io|bit_finetune|test)
  -p|--port <9000> - master port
  -i|--install-deps - If install dependencies (default: False)
```

### Training on local machine

``` sh
bash run.sh -g 4 -t train --cfg experiments/imagenet/resnet/r50s3a-aug4-w4c300-bnwd0.yaml
```

You can also modify the config paramters by the command line. For example, if you want to change the lr rate to 0.1, you can run the command:
``` sh
bash run.sh -g 4 -t train --cfg experiments/imagenet/resnet/r50s3a-aug4-w4c300-bnwd0.yaml TRAIN.LR 0.1
```

Notes:
- The checkpoint, model, and log files will be saved in OUTPUT/{dataset}/{training config} by default.

### Training on philly or AML 

You can use [vislib-toolbox](https://dev.azure.com/vislib/vislib-toolbox) to submit jobs to philly or AML. If you do not know the usage of [vislib-toolbox](https://dev.azure.com/vislib/vislib-toolbox). Please follow the instruction of [vislib-toolbox](https://dev.azure.com/vislib/vislib-toolbox) to install and configure `vislib-toolbox` crorectly. 
If you have already installed and configured `vislib-toolbox`. You can submit jobs to philly or AML by `vislib-toolbox`. For example, you can submit a 8 GPUs job to ${SERVER_NAME} by runing the cmd:
``` sh
vislib-job submit 'bash -x run.sh -i -t train --cfg experiments/imagenet/resnet/r50s3a-aug4-w5c300-bnwd0.yaml PRINT_FREQ 1000' --server ${SERVER_NAME} -r 1 -d imagenet -g 8
```

### Testing pre-trained models

``` sh
bash run.sh -t test --cfg experiments/imagenet/resnet/r50s3a-aug4-w5c300-bnwd0.yaml TEST.MODEL_FILE ${PRETRAINED_MODLE_FILE}
```

### Finetuning pre-trained models
``` sh
bash run.sh -t bit_finetune --cfg experiments/imagenet/resnet/r50s3a-finetune.yaml MODEL.PRETRAINED ${PRETRAINED_MODLE_FILE}
```

## Auto Mixed Precison (AMP) training
We encourage to use AMP to speed up your training when using V100 GPU or newer GPU architecture.
To enable AMP training, you need set `AMP.ENABLED` to `True`. For example,
``` sh
vislib-job submit 'bash -x run.sh -i -t train --cfg experiments/imagenet/resnet/r50s3a-aug4-w5c300-bnwd0.yaml AMP.ENABLED True' --server ${SERVER_NAME} -r 1 -d imagenet -g 8
```

