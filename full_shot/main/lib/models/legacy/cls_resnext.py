from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import torch
import torch.nn as nn
import math


__all__ = ['ResNeXt', 'get_cls_model']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResNextBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, base_width, cardinality,
                 stride=1, downsample=None):
        super(ResNextBottleneck, self).__init__()

        D = math.floor(planes * base_width / 64)
        C = cardinality
        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C)
        self.conv2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=C)
        self.bn2 = nn.BatchNorm2d(D*C)
        self.conv3 = nn.Conv2d(D*C, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, base_width, cardinality,
                 kernel_size_stem=7, num_classes=1000):
        self.inplanes = 64
        super(ResNeXt, self).__init__()

        # stem layers
        self.kernel_size_stem = kernel_size_stem
        if kernel_size_stem == 7:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif kernel_size_stem == 3:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                                   bias=False)
            self.bn2 = nn.BatchNorm2d(64)
        else:
            raise ValueError('Unknown stem conv kernel size: {}'.format(
                kernel_size_stem))

        self.layer1 = self._make_layer(block, 64, layers[0],
                                       base_width, cardinality)
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       base_width, cardinality, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2],
                                       base_width, cardinality, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       base_width, cardinality, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks,
                    base_width, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, base_width,
                            cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.kernel_size_stem == 7:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        elif self.kernel_size_stem == 3:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

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
    cfg = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }

    num_layers = config.MODEL.EXTRA.NUM_LAYERS
    base_width = config.MODEL.EXTRA.BASE_WIDTH
    cardinality = config.MODEL.EXTRA.CARDINALITY
    kernel_size_stem = config.MODEL.EXTRA.KERNEL_SIZE_STEM
    model = ResNeXt(
        ResNextBottleneck, cfg[num_layers],
        base_width=base_width,
        cardinality=cardinality,
        kernel_size_stem=kernel_size_stem,
        num_classes=config.MODEL.NUM_CLASSES
    )

    if config.MODEL.INIT_WEIGHTS:
        model.init_weights(
            config.MODEL.PRETRAINED,
            config.MODEL.PRETRAINED_LAYERS
        )

    return model
