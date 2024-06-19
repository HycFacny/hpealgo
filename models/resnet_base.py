from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from pathlib import Path

import torch
from torch import nn

from timm.models.layers.weight_init import trunc_normal_

from models.network_modules import conv3x3
from models.network_modules import BasicBlock
from models.network_modules import Bottleneck


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNetBase(nn.Module):

    def __init__(self, block, num_blocks, cfg, **kwargs):
        super().__init__()
        
        self.in_channels = 64
        extra = cfg['MODEL']['EXTRA']
        self.num_joints = cfg['MODEL']['NUM_JOINTS']
        self.deconv_with_bias = extra['DECONV_WITH_BIAS']
        self.num_deconv_layers = extra['NUM_DECONV_LAYERS']
        self.num_deconv_filters = extra['NUM_DECONV_FILTERS']
        self.num_deconv_kernels = extra['NUM_DECONV_KERNALS']

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels, momentum=BN_MOMENTUM)
        self.relu = nn.Relu(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # deconv layers
        self.deconv_layers = self._make_deconv_layer()

        self.head = nn.Conv2d(
            self.num_deconv_filters[-1],
            self.num_joints,
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                )
                nn.BatchNorm2d(out_channels * block.expansionm, momentum=BN_MOMENTUM)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _get_deconv_cfg(self, deconv_kernel):
        paddinig, output_padding = 0, 0
        if deconv_kernel in [3, 4]: padding = 1
        if deconv_kernel == 3: output_padding = 1
        
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self):
        if [self.num_deconv_layers, self.num_deconv_layers] != \
            [len(self.num_deconv_kernels), len(self.num_deconv_filters)]:
            ermsg = f'NUM_DECONV_LAYERS must be the same as size of NUM_DECONV_KERNALS and NUM_DECONV_FILTERS'
            logger.error(ermsg)
            raise ValueError(ermsg)        

        layers = []
        for layer_index in range(self.num_deconv_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(self.num_deconv_kernels[layer_index])

            out_channels = self.num_deconv_filters[layer_index]
            layers.append(nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias
            ))
            layers.append(nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))
            layers.append(nn.Relu(inplace=True))
            self.in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.head(x)
        
        return x
    
    def init_weights(self, pretrained='', print_load_info=False):
        logger.info('=> init ResNet weights forming as normal distribution')
        
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                trunc_normal_(module.weight, std=.02)
            if isinstance(module, nn.BatchNorm2d):
                trunc_normal_(module.weight, std=1.)
                nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.ConvTranspose2d):
                trunc_normal_(module.weight, std=.02)

        if Path(pretrained).is_file():
            pt_dict = torch.load(pretrained)
            logger.info(f'=> loading pre-trained model {pretrained}')

            existing_state_dict = {}
            for name, module in pt_dict.items():
                if name.split('.')[0] in self.pretrained_layers and name in self.state_dict()\
                   or self.pretrained_layers[0] is '*':
                    existing_state_dict[name] = module
                    if print_load_info:
                        print(f':: {name} is loaded from {pretrained}')
            self.load_state_dict(existing_state_dict, strict=False)


def get_pose_net(cfg, is_train, **kwargs):
    extra = cfg['MODEL']['EXTRA']
    num_layers = extra['NUM_LAYERS']
    block, num_blocks = resnet_spec[num_layers]
    
    model = ResNetBase(block, num_blocks, cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])
    
    return model