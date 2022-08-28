import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.dropblock import DropBlock
from layers.se_layer import SELayer


__all__ = ['ResNetV2', 'get_cls_model']


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, dilation=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     dilation=dilation, padding=dilation, bias=bias,
                     groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)


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
    ):
        super(PreActBottleneck, self).__init__()

        self.dropblock = dropblock

        self.gn0 = nn.GroupNorm(32, in_channels)
        self.act0 = nn.ReLU(inplace=True)

        self.conv1 = conv1x1(in_channels, bottleneck_channels)
        self.gn1 = nn.GroupNorm(32, bottleneck_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(
            bottleneck_channels,
            bottleneck_channels,
            stride=stride,
            dilation=dilation,
            groups=num_groups
        )
        self.gn2 = nn.GroupNorm(32, bottleneck_channels)
        self.act2 = nn.ReLU(inplace=True)

        self.conv3 = conv1x1(bottleneck_channels, out_channels)

        self.downsample = downsample
        self.stride = stride

        self.se = SELayer(out_channels) if with_se else None

    def forward(self, x):
        residual = x

        out = self.act0(self.gn0(x))
        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)
        if self.dropblock:
            out = self.dropblock(out)

        out = self.conv2(self.act1(self.gn1(out)))
        if self.dropblock:
            out = self.dropblock(out)

        out = self.conv3(self.act2(self.gn2(out)))
        if self.dropblock:
            out = self.dropblock(out)

        if self.se:
            out = self.se(out)

        if self.dropblock:
            residual = self.dropblock(residual)

        return out + residual


class ResNetV2(nn.Module):

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
        super(ResNetV2, self).__init__()

        self.zero_init_gamma = zero_init_gamma
        # stem layers
        self.deep_stem = deep_stem
        self.kernel_size_stem = kernel_size_stem
        if deep_stem:
            self.conv1 = conv3x3(3, 32, stride=2, dilation=1)
            self.gn1 = nn.GroupNorm(32, 32)
            self.act1 = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(32, 64)
            self.gn2 = nn.GroupNorm(32, 64)
            self.act2 = nn.ReLU(inplace=True)
            self.conv3 = conv3x3(64, 64, stride=2)
        else:
            if kernel_size_stem == 7:
                self.conv1 = StdConv2d(3, 64, kernel_size=7, stride=2,
                                       padding=3, bias=False)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            elif kernel_size_stem == 3:
                self.conv1 = conv3x3(3, 64, stride=2)
                self.gn1 = nn.GroupNorm(32, 64)
                self.act1 = nn.ReLU(inplace=True)
                self.conv2 = conv3x3(64, 64, stride=2)
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
            nn.GroupNorm(32, res5_out_channels),
            nn.ReLU(inplace=True)
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
            elif isinstance(m, nn.GroupNorm):
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
                    conv1x1(in_channels, out_channels),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(in_channels, out_channels, stride=first_stride)
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
            x = self.gn1(x)
            x = self.act1(x)
            x = self.conv2(x)
            x = self.gn2(x)
            x = self.act2(x)
            x = self.conv3(x)
        else:
            if self.kernel_size_stem == 7:
                x = self.conv1(x)
                x = self.maxpool(x)
            elif self.kernel_size_stem == 3:
                x = self.conv1(x)
                x = self.gn1(x)
                x = self.act1(x)
                x = self.conv2(x)

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
            elif isinstance(m, nn.GroupNorm):
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
    with_se = False if 'WITH_SE' not in spec else spec.WITH_SE

    assert num_layers not in [18, 34], 'ResNetV2 does not support R18/34 '

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

        stage = ResNetV2.make_layer(
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

    model = ResNetV2(
        stages,
        res5_out_channels=in_channels,
        dropblocks=dropblocks,
        deep_stem=deep_stem,
        kernel_size_stem=kernel_size_stem,
        num_classes=config.MODEL.NUM_CLASSES,
        zero_init_gamma=zero_init_gamma,
        loss=config.LOSS.LOSS,
        dropout=0.0 if 'DROPOUT' not in spec else spec['DROPOUT'],
    )

    if config.MODEL.INIT_WEIGHTS:
        model.init_weights(
            config.MODEL.PRETRAINED,
            config.MODEL.PRETRAINED_LAYERS,
            config.VERBOSE
        )

    return model
