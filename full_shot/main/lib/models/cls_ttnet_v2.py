import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


def conv_bn_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn_relu(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
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
    def __init__(self, channel_in, channel_out, channel_neck, stride, block_type, mini_size=4):
        super(InvertedResidual, self).__init__()
        self.mini_size = mini_size
        self.block_type = block_type
        self.stride = stride
        assert stride in [1, 2]

        # bottom block
        if (self.block_type==1):
            assert(stride==1)
            self.channel_branch_in = channel_in
            # branch2 forwards input faithfully, and branch1 outputs the rest of output channels
            channel_inc = channel_out-channel_in
            self.branch1 = nn.Sequential(
                # pw, conv1x1, ch_in->ch_neck, bn-relu
                nn.Conv2d(self.channel_branch_in, channel_neck, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channel_neck),
                nn.ReLU(inplace=True),
                # dw, conv3x3, bn-relu
                nn.Conv2d(channel_neck, channel_neck, 5, stride, 2, groups=channel_neck, bias=False),
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
            channel_inc = channel_out-channel_in
            self.branch1 = nn.Sequential(
                # pw, conv1x1, ch_in/2->ch_neck, bn-relu
                nn.Conv2d(self.channel_branch_in, channel_neck, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channel_neck),
                nn.ReLU(inplace=True),
                # dw, conv3x3, bn-relu
                nn.Conv2d(channel_neck, channel_neck, 5, stride, 2, groups=channel_neck, bias=False),
                nn.BatchNorm2d(channel_neck),
                nn.ReLU(inplace=True),
                # pw, ch_neck->ch_out-ch_in, bn
                nn.Conv2d(channel_neck, channel_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channel_inc),
            )                

            self.branch2 = nn.Sequential(
                # pw, conv1x1, ch_in/2->ch_neck, bn-relu
                nn.Conv2d(self.channel_branch_in, channel_neck, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channel_neck),
                nn.ReLU(inplace=True),
                # dw, conv3x3, bn-relu
                nn.Conv2d(channel_neck, channel_neck, 5, stride, 2, groups=channel_neck, bias=False),
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
                nn.Conv2d(self.channel_branch_in, channel_neck, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channel_neck),
                nn.ReLU(inplace=True),
                # dw, conv3x3, bn-relu
                nn.Conv2d(channel_neck, channel_neck, 5, stride, 2, groups=channel_neck, bias=False),
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
        if 1==self.block_type:
            out = self._concat(self.branch1(x), x)
        # downsampling block
        elif 2==self.block_type:
            x1 = x[:, :self.channel_branch_in, :, :]
            x2 = x[:, self.channel_branch_in:, :, :]
            out = self._concat(self.branch1(x1), self.branch2(x2))
        # normal block
        elif 3==self.block_type:
            x1 = x[:, :self.channel_branch_in, :, :]
            x2 = x[:, self.channel_branch_in:, :, :]
            out = self._concat(self.branch1(x1), x2)
        else:
            raise TypeError('unsupported block type')

        return channel_shuffle(out, 2, mini_size=self.mini_size)


class UpsampleBlock(nn.Module):
    def __init__(self, low_channel, high_channel, upscale=2):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
                conv_1x1_bn_relu(high_channel, low_channel),
                nn.Upsample(scale_factor=upscale, mode='bilinear')
                )
        self.output = nn.ReLU(inplace=True)

    def forward(self, features):
        low_feature = features[0]
        high_feature = features[1]
        hf_up = self.upsample(high_feature)
        return self.output(hf_up+low_feature)


class ShuffleBaseNet(nn.Module):
    def __init__(self):
        super(ShuffleBaseNet, self).__init__()
        
        kickoff_channel = 8
        stage_start_channel = 16
        self.stage_output_channel = [32, 64, 128, 1024]
        self.stage_block_repeats = [3, 5, 10, 5]

        # building first layer
        self.conv1 = conv_bn_relu(3, kickoff_channel, 2)    
        self.block1 = InvertedResidual(
                channel_in=kickoff_channel,
                channel_out=stage_start_channel,
                channel_neck=kickoff_channel*2,
                stride=1,
                block_type=1,
                mini_size=4)
        
        last_stage_output_channel = stage_start_channel
        self.stages = []
        # building inverted residual blocks
        for stage_i in range(len(self.stage_output_channel)):
            numrepeat = self.stage_block_repeats[stage_i]
            output_channel = self.stage_output_channel[stage_i]
            feature_stage = []
            for i in range(numrepeat):
                if i == 0:
                    feature_stage.append(InvertedResidual(
                        channel_in=last_stage_output_channel,
                        channel_out=output_channel,
                        channel_neck=last_stage_output_channel,
                        stride=2,
                        block_type=2,
                        mini_size=4))
                else:
                    feature_stage.append(InvertedResidual(
                        channel_in=output_channel,
                        channel_out=output_channel,
                        channel_neck=output_channel,
                        stride=1,
                        block_type=3,
                        mini_size=4))
            feature_stage = nn.Sequential(*feature_stage)
            self.add_module('stage_%d'%(stage_i+1),feature_stage)
            self.stages.append(feature_stage)
            last_stage_output_channel = output_channel

    def forward(self, x):
        output = []
        x = self.conv1(x)
        x = self.block1(x)
        # output.append(x)
        for stage in self.stages:
            x = stage(x)
            # output.append(x)
        return x


class MobileShuffleV2Net(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(MobileShuffleV2Net, self).__init__()
        self.backbone = ShuffleBaseNet()
        self.fc = nn.Linear(1024, 1000)

    def forward(self, data):
        x = self.backbone(data) 
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        y = self.fc(x)
        # y = F.avg_pool2d(x, kernel_size=x.size()
        #                      [2:]).view(x.size(0), -1)
            
        return y

    def set_training(self):
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
        logger.info('=> init weights by kaiming initialization(fan_out)')
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

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    if verbose:
                        logger.info('=> init {} from {}'.format(name, pretrained))
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} does not exist!'.format(pretrained))


def get_cls_model(cfg, **kwargs):
    model = MobileShuffleV2Net(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED, False)

    return model
