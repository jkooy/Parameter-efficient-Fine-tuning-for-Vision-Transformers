from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import (
    Conv2d,
    get_norm,
)
from layers.se_layer import SELayer

__all__ = ['get_cls_model']


BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False, groups=groups
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        with_se=False,
        with_relu=True,
        groups=1
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

        self.se = SELayer(planes, 1) if with_se else None
        self.with_relu = with_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.with_relu:
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        with_se=False,
        with_relu=True,
        groups=1
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.se = SELayer(planes*self.expansion) if with_se else None
        self.with_relu = with_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.with_relu:
            out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, groups, multi_scale_output=True,
                 with_se=False):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels, groups, with_se
        )
        fd, fu = self._make_fuse_layers_v2()
        self.fuse_downsample_layers = fd
        self.fuse_upsample_layers = fu
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logging.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logging.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logging.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(
        self,
        branch_index,
        block,
        num_blocks,
        num_channels,
        groups,
        with_se,
        stride=1
    ):
        downsample = None
        num_channels_in = self.num_inchannels[branch_index]
        num_channels_out = num_channels[branch_index] * block.expansion
        if (
            stride != 1
            or num_channels_in != num_channels_out
        ):
            downsample = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False
                ),
                Conv2d(
                    num_channels_in,
                    num_channels_out,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    norm=get_norm('BN', num_channels_out)
                )
            )

        num_blocks_cur_branch = num_blocks[branch_index]
        layers = [
            block(
                num_channels_in if i == 0 else num_channels_out,
                num_channels[branch_index],
                stride=stride if i == 0 else 1,
                downsample=downsample if i == 0 else None,
                with_se=with_se,
                with_relu=True if i != num_blocks_cur_branch-1 else False,
                groups=groups[branch_index],
            ) for i in range(num_blocks_cur_branch)
        ]
        self.num_inchannels[branch_index] = num_channels_out

        return nn.Sequential(*layers)

    def _make_branches(
        self, num_branches, block, num_blocks, num_channels, groups, with_se
    ):
        branches = [
            self._make_one_branch(
                i, block, num_blocks, num_channels, groups, with_se=with_se
            ) for i in range(num_branches)
        ]

        return nn.ModuleList(branches)

    def _make_fuse_layers_v2(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_channels_in = self.num_inchannels

        fuse_downsample_layers = [
            Conv2d(
                num_channels_in[i],
                num_channels_in[i+1],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                norm=get_norm('BN', num_channels_in[i+1])
            ) for i in range(num_branches-1)
        ]
        fuse_upsample_layers = [
            Conv2d(
                num_channels_in[-1-i],
                num_channels_in[-2-i],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                norm=get_norm('BN', num_channels_in[-2-i])
            ) for i in range(num_branches-1)
        ]

        return (
            nn.ModuleList(fuse_downsample_layers),
            nn.ModuleList(fuse_upsample_layers)
        )

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [F.relu_(self.branches[0](x[0]))]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_downsample = [0]
        x_upsample = [0]
        for i in range(self.num_branches-1):
            x_downsample.append(
                self.fuse_downsample_layers[i](
                    F.relu_(x_downsample[i]+x[i])
                )
            )
            x_upsample.append(
                F.interpolate(
                    self.fuse_upsample_layers[i](
                        F.relu_(x[-1-i]+x_upsample[i])
                    ),
                    scale_factor=2,
                    mode='nearest'
                )
            )

        x_fuse = []
        for i in range(self.num_branches):
            x_fuse.append(F.relu_(x[i] + x_downsample[i] + x_upsample[-1-i]))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(HighResolutionNet, self).__init__()
        spec = cfg.MODEL.SPEC
        with_se = spec.WITH_SE

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._build_layer(Bottleneck, 64, 64, 4, with_se=with_se)

        # build stage
        self.stages_spec = spec.STAGES
        self.num_stages = spec.STAGES.NUM_STAGES
        num_channels_last = [256]
        for i in range(self.num_stages):
            num_channels = self.stages_spec.NUM_CHANNELS[i]
            transition_layer = \
                self._build_transition_layer(num_channels_last, num_channels)
            setattr(self, 'transition{}'.format(i+1), transition_layer)

            stage, num_channels_last = self._build_stage(
                self.stages_spec, i, num_channels, True, with_se
            )
            setattr(self, 'stage{}'.format(i+2), stage)

        # Classification Head
        head_spec = spec.HEAD
        block_type = head_spec.BLOCK
        num_channels = head_spec.NUM_CHANNELS
        num_channels_proj = head_spec.NUM_CHANNELS_PROJ
        output_modules = self._build_head(
            num_channels_last=num_channels_last,
            blocks=block_type,
            num_channels=num_channels,
            num_channels_proj=num_channels_proj,
            with_se=with_se
        )
        self.incre_modules = output_modules[0]
        self.downsample_modules = output_modules[1]
        self.proj_modules = output_modules[2]
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        in_features = num_channels_proj if num_channels_proj > 0 \
            else num_channels[-1] * blocks_dict[block_type[-1]]

        self.classifier = nn.Linear(in_features, cfg.MODEL.NUM_CLASSES)

    def _build_head(
        self,
        num_channels_last,
        blocks,
        num_channels,
        num_channels_proj,
        with_se
    ):
        incre_modules = []
        for i, channels in enumerate(num_channels_last):
            incre_module = self._build_layer(
                block=blocks_dict[blocks[i]],
                inplanes=channels,
                planes=num_channels[i],
                blocks=1,
                stride=1,
                with_se=with_se,
                with_relu=False
            )
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
            
        # downsampling modules
        downsample_modules = []
        for i in range(len(num_channels_last)-1):
            in_channels = num_channels[i] * blocks_dict[blocks[i]].expansion
            out_channels = num_channels[i+1] * blocks_dict[blocks[i]].expansion

            downsample_module = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            )

            downsample_modules.append(downsample_module)
        downsample_modules = nn.ModuleList(downsample_modules)

        num_channels_in = num_channels[-1] * blocks_dict[blocks[-1]].expansion
        proj_modules = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels_in,
                out_channels=num_channels_proj,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(num_channels_proj, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        ) if num_channels_proj > 0 else None

        return incre_modules, downsample_modules, proj_modules

    def _build_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False
                        ),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _build_layer(self, block, inplanes, planes, blocks,
                     stride=1, with_se=False, with_relu=True):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(
            block(inplanes, planes, stride, downsample, with_se=with_se)
        )
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    with_se=with_se,
                    with_relu=with_relu if i == blocks - 1 else True
                )
            )

        return nn.Sequential(*layers)

    def _build_stage(self, stages_spec, stage_index, num_inchannels,
                     multi_scale_output=True, with_se=False):
        num_modules = stages_spec.NUM_MODULES[stage_index]
        num_branches = stages_spec.NUM_BRANCHES[stage_index]
        num_blocks = stages_spec.NUM_BLOCKS[stage_index]
        num_channels = stages_spec.NUM_CHANNELS[stage_index]
        block = blocks_dict[stages_spec['BLOCK'][stage_index]]
        fuse_method = stages_spec.FUSE_METHOD[stage_index]
        groups = [1] * num_branches if 'GROUPS' not in stages_spec \
            else stages_spec['GROUPS'][stage_index]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    groups,
                    reset_multi_scale_output,
                    with_se=with_se
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, 'transition{}'.format(i+1))
            for j in range(self.stages_spec['NUM_BRANCHES'][i]):
                if transition[j]:
                    if j >= len(y_list):
                        x_list.append(transition[j](y_list[-1]))
                    else:
                        x_list.append(transition[j](y_list[j]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, 'stage{}'.format(i+2))(x_list)

        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsample_modules)):
            y = self.incre_modules[i+1](y_list[i+1]) \
                + self.downsample_modules[i](F.relu(y))

        y = F.relu_(y)
        y = self.proj_modules(y) if self.proj_modules else y
        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)

        return y

    def init_weights(self, pretrained='', pretrained_layers=[]):
        logging.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
                    logging.info(f'=> init {k} from {pretrained}')
                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)


def get_cls_model(config, **kwargs):
    model = HighResolutionNet(config, **kwargs)
    if config.MODEL.INIT_WEIGHTS:
        model.init_weights(
            config.MODEL.PRETRAINED,
            config.MODEL.PRETRAINED_LAYERS
        )

    return model
