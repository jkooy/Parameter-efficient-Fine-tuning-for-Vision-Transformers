from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn.functional as F
import torch.nn as nn

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3_bn_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv1x1_bn_relu(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def seperable_conv3x3(inp, outp, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.Conv2d(inp, outp, 1, 1, 0, bias=False),
        nn.BatchNorm2d(outp),
    )


def seperable_conv3x3_relu(inp, outp, stride=1):
    return nn.Sequential(
        seperable_conv3x3(inp, outp, stride),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups, mini_size=4):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    mini_num_per_group = channels_per_group // mini_size

    # reshape
    x = x.view(batchsize, groups,
        mini_num_per_group, mini_size, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def channel_unshuffle(x, groups, mini_size=4):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    mini_num_per_group = channels_per_group // mini_size

    # reshape
    x = x.view(batchsize, mini_num_per_group, groups,
        mini_size, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


# shufflenet v2 style building block
class InvertedResidual(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_out,
                 channel_neck,
                 stride,
                 block_type,
                 mini_size=4,
                 kernel_size=3):
        super(InvertedResidual, self).__init__()
        self.mini_size = mini_size
        self.block_type = block_type

        assert stride in [1, 2]
        self.stride = stride

        assert kernel_size in [3, 5]
        padding = 1 if kernel_size == 3 else 2

        # bottom block
        if (self.block_type==1):
            assert(stride==1)
            self.channel_branch_in = channel_in
            # branch2 forwards input faithfully, and branch1 outputs the rest of output channels
            channel_inc = channel_out - channel_in
            self.branch1 = nn.Sequential(
                # pw, conv1x1, ch_in->ch_neck, bn-relu
                nn.Conv2d(self.channel_branch_in, channel_neck, 1, 1, 0,
                          bias=False),
                nn.BatchNorm2d(channel_neck),
                nn.ReLU(inplace=True),
                # dw, conv3x3, bn-relu
                nn.Conv2d(channel_neck,
                          channel_neck,
                          kernel_size,
                          stride,
                          padding,
                          groups=channel_neck,
                          bias=False),
                nn.BatchNorm2d(channel_neck),
                nn.ReLU(inplace=True),
                # pw, ch_neck->ch_out-ch_in, bn
                nn.Conv2d(channel_neck, channel_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channel_inc),
            )

        # down-sampling block
        elif (self.block_type==2):
            assert(stride in [1, 2])
            self.channel_branch_in = channel_in//2
            channel_inc = channel_out - channel_in
            self.branch1 = nn.Sequential(
                # pw, conv1x1, ch_in/2->ch_neck, bn-relu
                nn.Conv2d(self.channel_branch_in, channel_neck, 1, 1, 0,
                          bias=False),
                nn.BatchNorm2d(channel_neck),
                nn.ReLU(inplace=True),
                # dw, conv3x3, bn-relu
                nn.Conv2d(channel_neck,
                          channel_neck,
                          kernel_size,
                          stride,
                          padding,
                          groups=channel_neck,
                          bias=False),
                nn.BatchNorm2d(channel_neck),
                nn.ReLU(inplace=True),
                # pw, ch_neck->ch_out-ch_in, bn
                nn.Conv2d(channel_neck, channel_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channel_inc),
            )

            self.branch2 = nn.Sequential(
                # pw, conv1x1, ch_in/2->ch_neck, bn-relu
                nn.Conv2d(self.channel_branch_in, channel_neck, 1, 1, 0,
                          bias=False),
                nn.BatchNorm2d(channel_neck),
                nn.ReLU(inplace=True),
                # dw, conv3x3, bn-relu
                nn.Conv2d(channel_neck,
                          channel_neck,
                          kernel_size,
                          stride,
                          padding,
                          groups=channel_neck,
                          bias=False),
                nn.BatchNorm2d(channel_neck),
                nn.ReLU(inplace=True),
                # pw, ch_neck->ch_in, bn
                nn.Conv2d(channel_neck, channel_in, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channel_in),
            )

        # normal block
        elif (self.block_type==3):
            assert(stride==1)
            self.channel_branch_in = channel_in//2
            channel_inc = channel_out - self.channel_branch_in
            self.branch1 = nn.Sequential(
                # pw, conv1x1, ch_in/2->ch_neck, bn-relu
                nn.Conv2d(self.channel_branch_in, channel_neck, 1, 1, 0,
                          bias=False),
                nn.BatchNorm2d(channel_neck),
                nn.ReLU(inplace=True),
                # dw, conv3x3, bn-relu
                nn.Conv2d(channel_neck,
                          channel_neck,
                          kernel_size,
                          stride,
                          padding,
                          groups=channel_neck,
                          bias=False),
                nn.BatchNorm2d(channel_neck),
                nn.ReLU(inplace=True),
                # pw, ch_neck->ch_out-ch_in/2, bn
                nn.Conv2d(channel_neck, channel_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channel_inc),
            )
        else:
            raise TypeError('unsupported block type')

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        # bottom block
        if 1 == self.block_type:
            out = self._concat(self.branch1(x), x)
        # downsampling block
        elif 2 == self.block_type:
            x1 = x[:, :self.channel_branch_in, :, :]
            x2 = x[:, self.channel_branch_in:, :, :]
            out = self._concat(self.branch1(x1), self.branch2(x2))
        # normal block
        elif 3 == self.block_type:
            x1 = x[:, :self.channel_branch_in, :, :]
            x2 = x[:, self.channel_branch_in:, :, :]
            out = self._concat(self.branch1(x1), x2)
        else:
            raise TypeError('unsupported block type')

        return channel_shuffle(out, 2, mini_size=self.mini_size)


blocks_dict = {
    'INVERTED': InvertedResidual,
}


class TTNetV3(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(TTNetV3, self).__init__()

        # stem net
        extra = cfg.MODEL.EXTRA
        self.num_class = cfg.MODEL.NUM_CLASSES

        self.stem, num_channel_last = self._build_stem(extra.STEM)
        self.stages, num_channel_last = self._build_stages(num_channel_last, extra.STAGES)
        self.conv1x1 = conv1x1_bn_relu(num_channel_last, extra.NUM_CHANNEL_FINAL)
        self.fc = nn.Linear(extra.NUM_CHANNEL_FINAL, self.num_class)

    def _build_stem(self, cfg_stem):
        num_channel_kickoff = cfg_stem.NUM_CHANNEL_KICKOFF
        num_channel_stem_start = cfg_stem.NUM_CHANNEL_STEM_START
        expand_stem_start = cfg_stem.EXPAND_STEM_START
        kernel_size = cfg_stem.KERNEL_SIZE

        # building first layer
        stem = [conv3x3_bn_relu(3, num_channel_kickoff, 2)]
        stem.append(
            InvertedResidual(
                channel_in=num_channel_kickoff,
                channel_out=num_channel_stem_start,
                channel_neck=num_channel_kickoff*expand_stem_start,
                stride=1,
                block_type=1,
                mini_size=4,
                kernel_size=kernel_size
            )
        )

        num_channel_last = num_channel_stem_start
        return nn.Sequential(*stem), num_channel_last

    def _build_stages(self, num_channel_last, cfg_stages):
        stages = []
        for (num_channel_output, num_block_repeats, kernel_size) \
            in zip(cfg_stages.NUM_CHANNEL_OUTPUT,
                   cfg_stages.NUM_BLOCK_REPEATS,
                   cfg_stages.KERNEL_SIZE):
            stage, num_channel_last = \
                self._build_one_stage(
                    num_channel_last,
                    num_channel_output,
                    num_block_repeats,
                    kernel_size)
            stages.append(stage)

        return nn.ModuleList(stages), num_channel_last

    def _build_one_stage(self, num_channel_last, num_channel_output, num_block_repeats, kernel_size):
        stage = []
        # building inverted residual blocks
        stage.append(
            InvertedResidual(
                channel_in=num_channel_last,
                channel_out=num_channel_output,
                channel_neck=num_channel_last,
                stride=2,
                block_type=2,
                mini_size=4,
                kernel_size=kernel_size
            )
        )

        for i in range(1, num_block_repeats):
            stage.append(
                InvertedResidual(
                    channel_in=num_channel_output,
                    channel_out=num_channel_output,
                    channel_neck=num_channel_output,
                    stride=1,
                    block_type=3,
                    mini_size=4,
                    kernel_size=kernel_size
                )
            )
        num_channel_last = num_channel_output

        return nn.Sequential(*stage), num_channel_last

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.conv1x1(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.fc(x)

        return x

    def set_training(self):
        '''
        set_training are deprecated, wd are reset in utils.set_wd when calling get_optimzier()
        '''
        without_decay = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.groups==m.in_channels and m.groups==m.out_channels:
                without_decay.append(m.weight)
        with_decay = []
        for p in self.parameters():
            ever_set = False
            for pp in without_decay:
                if (pp is p):
                    ever_set = True
                    break
            if (not ever_set):
                with_decay.append(p)
        assert(len(with_decay)+len(without_decay)==len(list(self.parameters())))
        params = [{'params':with_decay},{'params':without_decay,'weight_decay':0.}]
        return params

    def init_weights(self, pretrained='', verbose=False):
        logger.info('==> init weights from kaiming distribution(fan_out)')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def get_cls_model(cfg, **kwargs):
    model = TTNetV3(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED, False)

    return model
