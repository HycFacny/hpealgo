from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from pathlib import Path

import torch
from torch import nn

from models.network_modules import BasicBlock
from models.network_modules import Bottleneck
from models.resnet_base import ResNetBase
from models.resnet_base import resnet_spec


logger = logging.getLogger(__name__)


class SimpleBaselines(nn.Module):
    def __init__(self, block, num_blocks, cfg, **kwargs):
        super().__init__()
        
        self.net = ResNetBase(block, num_blocks, cfg, kwargs)
    
    def forward(self, x):
        return self.net(x)

    def init_weights(self, pretrained='', print_load_info=False):
        self.net.init_weights(pretrained=pretrained, print_load_info=print_load_info)


def get_pose_net(cfg, is_train, **kwargs):
    extra = cfg['MODEL']['EXTRA']
    num_layers = extra['NUM_LAYERS']
    block, num_blocks = resnet_spec[num_layers]
    
    model = ResNetBase(block, num_blocks, cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])
    
    return model