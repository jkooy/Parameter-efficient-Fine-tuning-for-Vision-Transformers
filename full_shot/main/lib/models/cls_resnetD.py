import logging
import os

import torch
import torch.nn as nn

from layers.dropblock import DropBlock
from layers.se_layer import SELayer
from layers.dy_relu import DYReLU2

__all__ = ['ResNetD', 'get_cls_model']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def get_dy_relu(cfg, inp, oup):
    act = None
    if cfg and cfg.ENABLE:
        act = DYReLU2(
            inp=inp,
            oup=oup,
            reduction=cfg.REDUCTION,
            lambda_a=cfg.LAMBDA_A,
            K2=cfg.K2,
            use_bias=cfg.USE_BIAS,
            init_a=cfg.INIT_A,
            init_b=cfg.INIT_B
        )
    else:
        act = nn.ReLU(inplace=True)

    return act


class PreActBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        dilation=1,
        avg_down=False,
        downsample=None,
        dropblock=None,
        with_se=False,
        dy_relu_cfg=None
    ):
        super(PreActBottleneck, self).__init__()

        self.dropblock = dropblock
        self.act0 = get_dy_relu(dy_relu_cfg, in_channels, in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.act1 = get_dy_relu(
            dy_relu_cfg, bottleneck_channels, bottleneck_channels
        )
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1 * dilation,
            groups=num_groups,
            dilation=dilation,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.act2 = get_dy_relu(
            dy_relu_cfg, bottleneck_channels, bottleneck_channels
        )
        self.conv3 = nn.Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

        self.se = SELayer(out_channels) if with_se else None

    def forward(self, x):
        residual = x

        out = self.act0(x)
        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)
        if self.dropblock:
            out = self.dropblock(out)

        out = self.conv2(self.act1(self.bn1(out)))
        if self.dropblock:
            out = self.dropblock(out)

        out = self.conv3(self.act2(self.bn2(out)))
        if self.dropblock:
            out = self.dropblock(out)

        if self.se:
            out = self.se(out)

        if self.dropblock:
            residual = self.dropblock(residual)

        return self.bn3(out + residual)


class ResNetD(nn.Module):

    def __init__(self,
                 stages,
                 res5_out_channels,
                 dropblocks=[None]*4,
                 deep_stem=False,
                 kernel_size_stem=7,
                 num_classes=1000,
                 zero_init_gamma=False,
                 loss='softmax',
                 dropout=0.0,
                 dy_relu=None):
        super(ResNetD, self).__init__()

        self.zero_init_gamma = zero_init_gamma
        # stem layers
        self.deep_stem = deep_stem
        self.kernel_size_stem = kernel_size_stem
        if deep_stem:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2,
                                   padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.act1 = get_dy_relu(dy_relu, 32, 32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(64)
            self.act2 = get_dy_relu(dy_relu, 64, 64)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2,
                                   padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(64)
        else:
            if kernel_size_stem == 7:
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                                       padding=3, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            elif kernel_size_stem == 3:
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2,
                                       padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.act1 = get_dy_relu(dy_relu, 64, 64)
                self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2,
                                       padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(64)
            else:
                raise ValueError('Unknown stem conv kernel size: {}'.format(
                    kernel_size_stem))

        self.dropblocks = nn.ModuleList(dropblocks)
        self.stages_names = []
        for i, blocks in enumerate(stages):
            name = 'layer' + str(i+1)
            stage = nn.Sequential(*blocks)
            self.add_module(name, stage)
            self.stages_names.append((stage, name))

        self.final = nn.Sequential(
            get_dy_relu(dy_relu, res5_out_channels, res5_out_channels)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        if dropout > 0.0:
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(res5_out_channels, num_classes)
            )
        else:
            self.fc = nn.Linear(res5_out_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layer(block_class,
                   num_blocks,
                   first_stride,
                   *,
                   avg_down,
                   in_channels,
                   out_channels,
                   dropblock,
                   **kwargs):
        downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion:
        if first_stride != 1 or in_channels != out_channels:
            if avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(
                        kernel_size=first_stride,
                        stride=first_stride,
                        ceil_mode=True,
                        count_include_pad=False
                    ),
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=first_stride,
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                )

        layers = []
        for i in range(num_blocks):
            layers.append(
                block_class(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=first_stride if i == 0 else 1,
                    downsample=downsample if i == 0 else None,
                    dropblock=dropblock,
                    **kwargs
                )
            )
            in_channels = out_channels

        return layers

    def forward(self, x):
        for dropblock in self.dropblocks:
            if dropblock:
                dropblock.step()

        if self.deep_stem:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act2(x)
            x = self.conv3(x)
            x = self.bn3(x)
        else:
            if self.kernel_size_stem == 7:
                x = self.conv1(x)
                x = self.bn2(x)
                x = self.maxpool(x)
            elif self.kernel_size_stem == 3:
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.act1(x)
                x = self.conv2(x)
                x = self.bn2(x)

        for stage, name in self.stages_names:
            x = stage(x)

        x = self.final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if verbose:
                    logging.info(
                        '=> init {} weights with kaiming normal (fan_out)'
                        .format(n)
                    )
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                if verbose:
                    logging.info('=> init {} gamma to 1'.format(n))
                    logging.info('=> init {} beta to 0'.format(n))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_gamma:
            for n, m in self.named_modules():
                if isinstance(m, PreActBottleneck):
                    if verbose:
                        logging.info(
                            '=> zero init last bn of {}\' gamma'.format(n)
                        )
                    nn.init.constant_(m.bn3.weight, 0)

        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                    k.split('.')[0] in pretrained_layers
                    or pretrained_layers[0] is '*'
                )
                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)


def get_cls_model(config, **kwargs):
    resnet_spec = {
        50: (PreActBottleneck, [3, 4, 6, 3]),
        101: (PreActBottleneck, [3, 4, 23, 3]),
        152: (PreActBottleneck, [3, 8, 36, 3]),
    }

    spec = config.MODEL.SPEC
    num_layers = spec.NUM_LAYERS
    block_class = resnet_spec[num_layers][0]
    num_blocks = resnet_spec[num_layers][1]

    dropblocks = [None] * 4
    if config.AUG.DROPBLOCK_KEEP_PROB < 1.0:
        aug = config.AUG
        dropblock_layers = aug.DROPBLOCK_LAYERS
        keep_prob = aug.DROPBLOCK_KEEP_PROB
        block_size = aug.DROPBLOCK_BLOCK_SIZE
        for i in dropblock_layers:
            if i < 1 or i > 4:
                raise ValueError(
                    'dropblock layer should be between 1 and 4 '
                )
            _keep_prob = 1 - (1 - keep_prob) / 4.0**(4 - i)
            dropblocks[i-1] = DropBlock(
                block_size=block_size,
                keep_prob=_keep_prob
            )

    in_channels = 64
    out_channels = 64 if num_layers in [18, 34] else 256
    num_groups = spec.NUM_GROUPS
    width_per_group = spec.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    avg_down = spec.AVG_DOWN
    dy_relu_cfg = spec.DY_RELU if 'DY_RELU' in spec else None
    with_se = False if 'WITH_SE' not in spec else spec.WITH_SE

    if num_layers in [18, 34]:
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []
    for idx, num_block in enumerate(num_blocks):
        kwargs = {
            'first_stride': 1 if idx == 0 else 2,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'avg_down': avg_down
        }

        if num_layers not in [18, 34]:
            kwargs['bottleneck_channels'] = bottleneck_channels
            kwargs['num_groups'] = num_groups
            kwargs['dropblock'] = dropblocks[idx]
            kwargs['with_se'] = with_se
            kwargs['dy_relu_cfg'] = dy_relu_cfg

        stage = ResNetD.make_layer(
            block_class,
            num_block,
            **kwargs
        )
        stages.append(stage)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

    kernel_size_stem = 7 if 'KERNEL_SIZE_STEM' not in spec \
        else spec.KERNEL_SIZE_STEM
    deep_stem = False if 'DEEP_STEM' not in spec \
        else spec.DEEP_STEM

    zero_init_gamma = False if 'ZERO_INIT_GAMMA' not in config.MODEL.SPEC \
        else config.MODEL.SPEC.ZERO_INIT_GAMMA

    model = ResNetD(
        stages,
        res5_out_channels=in_channels,
        dropblocks=dropblocks,
        deep_stem=deep_stem,
        kernel_size_stem=kernel_size_stem,
        num_classes=config.MODEL.NUM_CLASSES,
        zero_init_gamma=zero_init_gamma,
        loss=config.LOSS.LOSS,
        dropout=0.0 if 'DROPOUT' not in spec else spec['DROPOUT'],
        dy_relu=dy_relu_cfg,
    )

    if config.MODEL.INIT_WEIGHTS:
        model.init_weights(
            config.MODEL.PRETRAINED,
            config.MODEL.PRETRAINED_LAYERS,
            config.VERBOSE
        )

    return model
