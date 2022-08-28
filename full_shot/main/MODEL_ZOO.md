# Introduction
The models are trained on the Philly or AML. 
The training logs, model definition files, configuration files, and model files are storaged on the blob containers under the account of [vislibexp](https://ms.portal.azure.com/#@72f988bf-86f1-41af-91ab-2d7cd011db47/resource/subscriptions/103fe10b-22e1-477d-aa81-b7a916e5087d/resourceGroups/Logging/providers/Microsoft.Storage/storageAccounts/vislibexp/overview). You can get all the files by the `exp id`. 

## Dataset
- **IN-1k**: ImageNet 1k dataset with 1.2M images.
- **IN-22k**: ImageNet 22k dataset with 14M images.

## Names
- **R{depth}-MSRA**: MSRA's original ResNet models pre-trained on IN-1k. For MSRA's original version, the bottleneck architecture use stride 2 in the first 1x1 convolution. Except for the MSRA original version, all other versions in the model zoo use stride 2 in the 3x3 convolution. 
- **R/X{depth}-PT**: These ResNet/ResNeXt models are from [PyTorch's model zoo](https://pytorch.org/docs/stable/torchvision/models.html#classification) pre-trained on IN-1k. **R** is for ResNet, and **X** is for ResNeXt.
- **R/X{depth}-S3**: 
  - We replace the first 7x7 conv and max pooling by two 3x3 conv with stride 2.
- **R/X{depth}-S3A**: 
  - We replace the first 7x7 conv and max pooling by two 3x3 conv with stride 2.
  - We also add an average pooling with stride 2 before the 1x1 conv in the downsample skip connection branch.
- **SE-R/X{depth}-S3A**: 
  - We replace the first 7x7 conv and max pooling by two 3x3 conv with stride 2. 
  - We also add an average pooling with stride 2 before the 1x1 conv in the downsample skip connection branch.
  - Add [Squeeze-and-Excitation Module](https://arxiv.org/abs/1709.01507).
- **BiT-R{depth}x{width factor}**: The model architecture from [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370). In BiT-ResNet, they use Group Normalization (GN) and Weight Standardization (WS), in stead of Batch Normalization (BN), which is useful for training with large batch sizes, and has a significant impact on transfer learning.

## Data augmentation
- **scale aug**: random scale (default: of 0.08 to 1.0) and random aspect ratio (default: 3/4 to 4/3) augmentation from [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842).
- **flip**: random (default: of 0.5) horizontal flip image.
- **color jitter (CJ)**: random change the brightness, contrast, hue and saturation with a probility. 
- **gray scale (GS)**: convert image to grayscale with a probility.
- **cutmix**: CutMix augmentation strategy from [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899). When using **cutmix**, usually a longer training scheduler is needed. 
- **mixup**: mixup augmentation from [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/pdf/1710.09412.pdf). Like **cutmix**, a longer training scheduler is needed when using **mixup**.

### Data augmentation in the config name
- **aug1**: **scale aug** + **flip**
- **aug2**: **scale aug** + **flip** + **mixup**
- **aug3**: **scale aug** + **flip** + **cutmix**
- **aug4**: **scale aug** + **flip** + **cutmix** + **color jitter** + **gray scale**

We use **scale aug** and **flip** as our default data augmentation. 


## Loss
- **label smoothing (LS)**: a regularization technique from [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567).


## Learning rate scheduler
- **multi step (MS{total epochs})**: multi step learning rate scheduler with {total epochs}.
- **cosine (C{total epochs})**: cosine learning rate scheduler with {total epochs}. 
- **warmup cosine (W{warmup epoch}C{total epochs})**: cosine learning rate scheduler with warmup.


## Weight decay
By default, weight decay is applied to all weights and biases. However, it is recommended that only applying weight decay to weights and leaving biases, gamma and beta in BN layers unregularied. This recommended config is named with **-bnwd0-**.


# Models pre-trained on IN-1k

| Config                                                                       | Model            | #params | FLOPs | lr scheduler | WD  | LS  | aug         | Top1 acc | Exp id                | url                                                                                                                                                                                                               |
|------------------------------------------------------------------------------|------------------|---------|-------|--------------|-----|-----|-------------|----------|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                              | R50-MSRA         | 25.5M   | 4.1B  | MS90         |     | 0   | center crop |          |                       |                                                                                                                                                                                                                   |
|                                                                              | R50-PT           | 25.5M   | 4.1B  | MS100        |     | 0   | aug1        | 76.15    |                       |                                                                                                                                                                                                                   |
| [cfg](/experiments/imagenet/resnet/r50s3-aug1-ms100.yaml)                    | R50-S3           | 25.5M   | 4.1B  | MS100        |     | 0   | aug1        | 76.33    | bixiexp20200605042043 |                                                                                                                                                                                                                   |
| [cfg](/experiments/imagenet/resnet/r50s3-aug1-c120.yaml)                     | R50-S3           | 25.5M   | 4.1B  | C120         |     | 0   | aug1        | 77.0     | bixiexp20200626173446 | [download](https://leohsiao.blob.core.windows.net/models/IN-1k/r50stem3avgdown0-224x224-labelsmooth0.0mixup0.0sgd-lr0.2wd0.0001bs64X8-CosineAnnealingeta0.0Ep120/model_best.pth)                                  |
| [cfg](/experiments/imagenet/resnet/r50s3a-aug1-c120.yaml)                    | R50-S3A          | 25.5M   | 4.1B  | C120         |     | 0   | aug1        | 77.9     | bixiexp20200626173406 | [download](https://leohsiao.blob.core.windows.net/models/IN-1k/r50stem3avgdown1-224x224-labelsmooth0.0mixup0.0sgd-lr0.2wd0.0001bs64X8-CosineAnnealingeta0.0Ep120/model_best.pth)                                  |
| [cfg](/experiments/imagenet/resnet/r50s3a-aug2-ls0.1-c200.yaml)              | R50-S3A          | 25.5M   | 4.1B  | C200         |     | 0.1 | aug2        | 79.0     | bixiexp20200628023452 | [download](https://leohsiao.blob.core.windows.net/models/IN-1k/r50stem3avgdown1-224x224-labelsmooth0.1mixup0.2sgd-lr0.2wd0.0001bs64X8-CosineAnnealingeta0.0Ep200/model_best.pth)                                  |
| [cfg](/experiments/imagenet/resnet/r50s3a-aug3-c300.yaml)                    | R50-S3A          | 25.5M   | 4.1B  | C300         |     | 0   | aug3        | 79.59    | bixiexp20200701051002 | [download](https://leohsiao.blob.core.windows.net/models/IN-1k/r50stem3avgdown1-224x224-labelsmooth0.0cutmix1.0sgd-lr0.2wd0.0001bs64X8-CosineAnnealingeta0.0Ep300/model_best.pth)                                 |
| [cfg](/experiments/imagenet/resnet/r50s3a-aug4-c300.yaml)                    | R50-S3A          | 25.5M   | 4.1B  | C300         |     | 0   | aug4        | 79.49    | bixiexp20200707064202 | [download](https://leohsiao.blob.core.windows.net/models/IN-1k/r50stem3avgdown1-224x224-cj0.4_0.4_0.4_0.1_0.8gs0.2gb0.0-labelsmooth0.0mixcut1.0sgd-lr0.2wd0.0001bs64X8-CosineAnnealingeta0.0Ep300/model_best.pth) |
| [cfg](/experiments/imagenet/resnet/r50s3a-aug4-w5c300.yaml)                  | R50-S3A          | 25.5M   | 4.1B  | W5C300       |     | 0   | aug4        | 79.6     | bixiexp20200730060109 | [download](https://leohsiao.blob.core.windows.net/models/IN-1k/r50stem3avgdown1-224x224-cj0.4_0.4_0.4_0.1_0.8gs0.2gb0.0-labelsmooth0.0mixcut1.0sgd-lr0.2wd0.0001bs64X8-CosineAnnealingeta0.0Ep300/model_best.pth) |
| [cfg](/experiments/imagenet/resnet/r50s3a-aug4-w5c300-bnwd0.yaml)            | R50-S3A          | 25.5M   | 4.1B  | W5C300       | bn0 | 0   | aug4        | 79.9     | bixiexp20200730060035 |                                                                                                                                                                                                                   |
| [cfg](/experiments/imagenet/resnext/se-x50-32x4d-s3a-aug4-w5c300-bnwd0.yaml) | SE-X50-32x4d-S3A | 27.6M   | 4.3B  | W5C300       | bn0 | 0   | aug4        | 81.3     | bixiexp20200809215735 | [download](https://leohsiao.blob.core.windows.net/models/IN-1k/se-x50-32x4d-s3a1-224x224-cj0.4_0.4_0.4_0.1_0.8gs0.2-labelsmooth0.0mixcut1.0sgd-lr0.8wd0.0001bs64X32-CosineEp300/model_best.pth)                   |

**Notes:**
- Training input size is 224x224.
- Training optimizer is SGD.
- For testing, we first resize the image to the size of 256x256, then crop the image at the center with an output size of 224x224. 
- Top 1 accuracy is performed on IN-1k val dataset. 

# Models pre-trained on IN-22k

| Config                                                                           | Model             | #params | FLOPs | lr scheduler | WD  | LS | aug  | IN-1k Top1 acc | Exp id | url                                                                                                                                                                                 |
|----------------------------------------------------------------------------------|-------------------|---------|-------|--------------|-----|----|------|----------------|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [cfg](/experiments/imagenet22k/resnet/r50s3-aug1-ms100.yaml)                     | R50-S3            | 68.3M   | 4.2B  | MS100        | /   | 0  | aug1 | 80.07          |        | [download](https://leohsiao.blob.core.windows.net/models/IN-22k/r50stem3-224x224-sgd-lr0.8wd0.0001bs64X32-MultiStep30_60_90factor0.1ep100/model_best.pth)                           |
| [cfg](/experiments/imagenet22k/resnet/r50s3a-aug4-w5c300-bnwd0.yaml)             | R50-S3A           | 68.3M   | 4.2B  | W5C300       | bn0 | 0  | aug4 | 80.6           |        | [download](https://leohsiao.blob.core.windows.net/models/IN-22k/r50-s3a1-224x224-cj0.4_0.4_0.4_0.1_0.8gs0.2mixcut1.0sgd-lr0.8wd0.0001bnwd0bs64X32-WarmupCosine5Ep300/model_best.pt) |
| [cfg](/experiments/imagenet22k/resnext/se-x50-32x4d-s3a-aug4-w5c300-bnwd0.yaml)  | SE-X50-32x4d-S3A  | 70.3M   | 4.3B  | W5C300       | bn0 | 0  | aug4 | 82.0           |        | [download](https://leohsiao.blob.core.windows.net/models/IN-22k/r50-s3a1-224x224-cj0.4_0.4_0.4_0.1_0.8gs0.2mixcut1.0sgd-lr0.8wd0.0001bnwd0bs64X32-WarmupCosine5Ep300/model_best.pt) |
| [cfg](/experiments/imagenet22k/resnext/x101-64x4d-s3-aug1-ms90.yaml)             | X101-64x4d-S3     | 126.2   | 15.6B | MS90         | /   | 0  | aug1 | 83.4           |        | [download](https://leohsiao.blob.core.windows.net/models/IN-22k/x101stem3c64w4-224x224-sgd-lr0.4wd0.0001bs32X32-MultiStep30_60_80factor0.1ep90/model_best.pth)                      |
| [cfg](/experiments/imagenet22k/resnext/se-x101-64x4d-s3a-aug4-w5c300-bnwd0.yaml) | SE-X101-64x4d-S3A | 130.9M  | 15.6B | W5C300       | bn0 | 0  | aug4 |                |        |                                                                                                                                                                                     |

**Notes:**
- Training input size is 224x224.
- Training optimizer is SGD.
- IN-1k top1 accuracy is result finetuning on IN-1k with the input size of 480x480.

