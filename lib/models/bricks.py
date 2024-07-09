from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.network_modules import BasicBlock
from models.network_modules import Bottleneck


logger = logging.getLogger(__name__)
BN_MOMENTUM = 0.1
block_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == 'relu': return F.relu
    if activation == 'gelu': return F.gelu
    if activation == 'glu': return F.glu
    if activation == 'sigmoid': return F.sigmoid

    raise RuntimeError(f'activation should be relu/gelu, not {activation}')


def _get_activation_layer(activation):
    if activation == 'relu': return nn.ReLU(inplace=True)
    if activation == 'gelu': return nn.GELU()
    if activation == 'glu': return nn.GLU(dim=-1)
    if activation == 'sigmoid': return nn.Sigmoid()
    
    raise RuntimeError(f'activation should be relu/gelu/glu/sigmoid, not {activation}')


def _get_norm_layer(channels, norm='bn'):
    if norm == 'layernorm': return nn.Laynorm(channels)
    if norm == 'bn': return nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
    
    raise RuntimeError(f'normalization should be laynorm/bn, not {norm}')


def channel_shuffle(x, groups):
    """Channel Shuffle operation.
    This function enables cross-group information flow for multiple groups
    convolution layers.
    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.
    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    assert (num_channels % groups == 0), ('num_channels should be '
                                          'divisible by groups')
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, groups * channels_per_group, height, width)

    return x
