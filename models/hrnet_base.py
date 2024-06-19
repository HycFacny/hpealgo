from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from timm.models.layers.weight_init import trunc_normal_

from models.network_modules import conv3x3
from models.network_modules import BasicBlock
from models.network_modules import Bottleneck
from utils.print_functions import print_inter_debug_info


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
block_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}
# cnt = 0
# fuze_cnt = 0

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == 'relu': return F.relu
    if activation == 'gelu': return F.gelu
    if activation == 'glu': return F.glu

    raise RuntimeError(f'activation should be relu/gelu, not {activation}')


# def _make_fuse_layers(num_branches, num_in_channels, multi_scale_output):
#     if num_branches == 1: return None

#     fuse_layers = []
#     # pre post layers k*k matric
#     for post in range(num_branches if multi_scale_output else 1):
#         layer = []
#         for pre in range(num_branches):
#             in_channel = num_in_channels[pre]
#             out_channel = num_in_channels[post]
#             if pre > post:
#                 layer.append(nn.Sequential(
#                     nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
#                     nn.BatchNorm2d(out_channel),
#                     nn.Upsample(scale_factor=2 ** (pre - post), mode='nearest')
#                 ))
#             elif pre < post:
#                 conv3x3s = []
#                 for cur in range(post - pre):
#                     out_channel_inter = out_channel \
#                         if cur == post - pre - 1 else in_channel
#                     conv3x3_ = nn.Sequential(
#                         nn.Conv2d(in_channel, out_channel_inter, kernel_size=3, stride=2, padding=1, bias=False),
#                         nn.BatchNorm2d(out_channel_inter, momentum=BN_MOMENTUM)
#                     )
#                     if cur < post - pre - 1:
#                         conv3x3_.add_module('relu_{}'.format(cur), nn.ReLU(False))
#                     conv3x3s.append(conv3x3_)
#                 layer.append(nn.Sequential(*conv3x3s))
#             else:
#                 layer.append(None)
#         fuse_layers.append(nn.ModuleList(layer))
    
#     return nn.ModuleList(fuse_layers)


def _head(feature_maps, stage, method='highest'):
    output = None
    if method == 'highest':
        output = feature_maps[0]
    elif method == 'concat':
        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        outputs = [feature_maps[0]]
        for _ in range(1, stage):
            outputs.append(feature_maps[_])
            for _x_ in range(_):
                outputs[_] = upsample(outputs[_])
        output = outputs[0]
        for _ in range(1, stage):
            torch.cat((output, outputs[_]), 1)
    else:
        logger.info(f'HRNet head method {method} is not supported in hpe problem, return defalut')
        output = feature_maps[0]
    
    return output


class HighResolutionModule(nn.Module):
    """
    Args: 
        num_branches: number of branches paralleled in this module
        blocks: Bottleneck or BasicBlock, both are class type
        num_blocks: [num_branches], store number of blocks in each branches
        num_in_channels: [num_branches], store number of channels input in each branch
        num_out_channels: [num_branches], store number of channels output of each branch
        fuse_method: fusion method in Fuse Layer
        multi_scale_output: 
    """

    def __init__(self, num_branches, block, num_blocks, num_in_channels,
                 num_out_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, num_blocks, num_in_channels, num_out_channels)

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.num_branches = num_branches
        self.num_blocks = num_blocks
        self.block = block

        self.fuse_method = fuse_method
        self.multi_scale_output = multi_scale_output
        
        self.branches = self._make_branches()
        # fuse[post][pre](feature maps) == nn.ModuleList, get the feature maps
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)
    
    def _check_branches(self, num_branches, num_blocks, num_in_channels, num_out_channels):
        if [num_branches, num_branches, num_branches] != \
            [len(num_blocks), len(num_out_channels), len(num_in_channels)]:
            ermsg = f'NUM_BRANCHES must be the same as size of NUM_BLOCKS and NUM_CHANNELS'
            logger.error(ermsg)
            raise ValueError(ermsg)
    
    # all basicblocks
    def _make_branch(self, branch_index, stride=1):
        downsample = None
        if stride != 1 or self.num_in_channels[branch_index] != self.num_out_channels[branch_index] * self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_in_channels[branch_index],
                    self.num_out_channels[branch_index] * self.block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(
                    self.num_out_channels[branch_index] * self.block.expansion,
                    momentum=BN_MOMENTUM
                )
            )

        branch_layers = [self.block(
            self.num_in_channels[branch_index], self.num_out_channels[branch_index], stride, downsample
        )]
        self.num_in_channels[branch_index] = self.num_out_channels[branch_index] * self.block.expansion
        for _ in range(1, self.num_blocks[branch_index]):
            branch_layers.append(self.block(
                self.num_in_channels[branch_index],
                self.num_out_channels[branch_index]
            ))
        
        return nn.Sequential(*branch_layers)
    
    def _make_branches(self):
        branches = []
        for branch_index in range(self.num_branches):
            branches.append(self._make_branch(branch_index))
        
        return nn.ModuleList(branches)
    
    def _make_fuse_layers(self):
        if self.num_branches == 1: return None

        fuse_layers = []
        # pre post layers k*k matric
        for post in range(self.num_branches if self.multi_scale_output else 1):
            layer = []
            for pre in range(self.num_branches):
                in_channel = self.num_in_channels[pre]
                out_channel = self.num_in_channels[post]
                if pre > post:
                    layer.append(nn.Sequential(
                        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(out_channel),
                        nn.Upsample(scale_factor=2 ** (pre - post), mode='nearest')
                    ))
                    # global cnt
                    # cnt += 1
                elif pre < post:
                    conv3x3s = []
                    for cur in range(post - pre):
                        out_channel_inter = out_channel if cur == post - pre - 1 else in_channel
                        conv3x3_ = nn.Sequential(
                            nn.Conv2d(in_channel, out_channel_inter, kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(out_channel_inter, momentum=BN_MOMENTUM)
                        )
                        if cur < post - pre - 1:
                            conv3x3_.add_module('(2): ', nn.ReLU(inplace=True))
                        conv3x3s.append(conv3x3_)
                    layer.append(nn.Sequential(*conv3x3s))
                else:
                    layer.append(None)
            fuse_layers.append(nn.ModuleList(layer))
        
        # global fuze_cnt
        # fuze_cnt += 1
        # print_inter_debug_info('fuse_layer {}'.format(fuze_cnt), nn.ModuleList(fuse_layers), 'hrnet')
        # print_inter_debug_info('num_branches of {}'.format(fuze_cnt), self.num_branches, 'hrnet')
        
        return nn.ModuleList(fuse_layers)
    
    def get_in_channels(self):
        return self.num_in_channels
    
    def forward(self, x):
        if self.num_branches == 1: return [self.branches[0](x[0])]
        # print(self.num_branches)
        # print(self.branches)
        for branch_index in range(self.num_branches):
            # if (self.num_branches > 2):
            #     print(len(x), branch_index, x[branch_index].shape)
            x[branch_index] = self.branches[branch_index](x[branch_index])
        
        x_fuse = []
        for post in range(len(self.fuse_layers)):
            # post += pre
            y = x[0] if post == 0 else self.fuse_layers[post][0](x[0])
            for pre in range(1, self.num_branches):
                if post == pre: y = y + x[pre]
                else: y = y + self.fuse_layers[post][pre](x[pre])
            x_fuse.append(self.relu(y))
    
        return x_fuse


class HRNetBase(nn.Module):
    global cnt
    
    def __init__(self, cfg, **kwargs):
        super().__init__()

        extra = cfg['MODEL']['EXTRA']
        print_inter_debug_info('hrinput_cfg', cfg, 'hrnet')
        print_inter_debug_info('cfg_extra', extra, 'hrnet')
        self.in_channels = 64
        self.output_method = cfg['MODEL']['OUTPUT_METHOD'] \
            if 'OUTPUT_METHOD' in cfg['MODEL'] else 'highest'

        # stem net
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        # print_inter_debug_info('bn1_inchannels', self.in_channels, 'hrnet')
        self.bn1 = nn.BatchNorm2d(self.in_channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        # print_inter_debug_info('bn2_inchannels', self.in_channels, 'hrnet')

        self.bn2 = nn.BatchNorm2d(self.in_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        # C = 64 -> 256
        self.layer1 = self._make_layer(Bottleneck, self.in_channels, 4)
        # self.in_channels == 256
        self.in_channels = [self.in_channels]

        # stage 2
        self.stage2_cfg = extra['STAGE2']
        block = block_dict[self.stage2_cfg['BLOCK']]

        self.out_channels = self.stage2_cfg['NUM_CHANNELS']
        self.out_channels = [self.out_channels[i] * block.expansion for i in range(len(self.out_channels))]
        # form transition layer to get the input channels and other setting for the next stage
        self.transition1 = self._make_transition_layer()
        
        # adjust in_channels
        self.in_channels = copy.deepcopy(self.out_channels)
        self.stage2 = self._make_stage(self.stage2_cfg)

        # stage 3
        self.stage3_cfg = extra['STAGE3']
        block = block_dict[self.stage3_cfg['BLOCK']]

        self.in_channels = self.out_channels
        self.out_channels = self.stage3_cfg['NUM_CHANNELS']
        self.out_channels = [self.out_channels[i] * block.expansion for i in range(len(self.out_channels))]
        # form transition layer to get the input channels and other setting for the next stage
        self.transition2 = self._make_transition_layer()
    
        # adjust in_channels
        self.in_channels = copy.deepcopy(self.out_channels)
        self.stage3 = self._make_stage(self.stage3_cfg, multi_scale_output=False)

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

    # first layer with no fuse layer
    def _make_layer(self, block, in_channels, num_blocks, stride=1):
        downsample = None
        # self.in_channels = 64, in_channels = 64
        if stride != 1 or self.in_channels != in_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    in_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels * block.expansion, momentum=BN_MOMENTUM)
            )
        
        branch_layers = [block(self.in_channels, in_channels, stride, downsample)]
        self.in_channels = in_channels * block.expansion
        for _ in range(1, num_blocks):
            branch_layers.append(block(self.in_channels, in_channels))
        
        return nn.Sequential(*branch_layers)
    

    def _make_transition_layer(self):
        num_branches_pre = len(self.in_channels)
        num_branches_post = len(self.out_channels)

        transition_layers = []
        for post in range(num_branches_post):
            if post < num_branches_pre:
                if self.in_channels[post] != self.out_channels[post]:
                    transition_layers.append(nn.Sequential(
                        # conv 3, 1, 1, False
                        conv3x3(self.in_channels[post], self.out_channels[post]),
                        nn.BatchNorm2d(self.out_channels[post], momentum=0.1),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                # here we only consider using pre_last downsampling to form post_{last+1}
                # post + 1 - num_branches_pre always be 1
                # we can change it to num_branches_pre and update transition method to multi-scale fusion
                for pre in range(post + 1 - num_branches_pre):
                    in_channels_3x3 = self.in_channels[-1]
                    out_channels_3x3 = self.out_channels[post] \
                        if pre == post - num_branches_pre else in_channels_3x3
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels_3x3,
                            out_channels_3x3,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False),
                        nn.BatchNorm2d(out_channels_3x3, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        
        return nn.ModuleList(transition_layers)
    

    def _make_stage(self, layer_config, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_out_channels = layer_config['NUM_CHANNELS']

        block = block_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for _ in range(num_modules):
            # upon on the last layer, we dont use multi output
            if not multi_scale_output and _ == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            
            modules.append(
                HighResolutionModule(
                    num_branches=num_branches,
                    block=block,
                    num_blocks=num_blocks,
                    num_in_channels=self.in_channels,
                    num_out_channels=num_out_channels,
                    fuse_method=fuse_method,
                    multi_scale_output=reset_multi_scale_output
                )
            )
            self.in_channels = modules[-1].get_in_channels()

        return nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for branch_index in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[branch_index] is not None:
                x_list.append(self.transition1[branch_index](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for branch_index in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[branch_index] is not None:
                if branch_index == self.stage3_cfg['NUM_BRANCHES'] - 1:
                    x_list.append(self.transition2[branch_index](y_list[-1]))
                else:
                    x_list.append(self.transition2[branch_index](y_list[branch_index]))
            else:
                x_list.append(y_list[branch_index])
        
        y_list = self.stage3(x_list)

        # default: output the highest one
        return _head(y_list, self.output_method)
    
    def init_weights(self, pretrained='', print_load_info=False):
        logger.info('=> init HRnet weights forming as normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
        
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
        
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError(f'{pretrained} is not exist!')


class HRNetBase_S4(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        extra = cfg['MODEL']['EXTRA']
        self.in_channels = 64
        self.output_method = cfg['MODEL']['OUTPUT_METHOD'] \
            if 'OUTPUT_METHOD' in cfg['MODEL'] else 'highest'

        # stem net
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.in_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        # C = 64 -> 256
        self.layer1 = self._make_layer(Bottleneck, self.in_channels, 4)
        # self.in_channels == 256
        self.in_channels = [self.in_channels]

        # stage 2
        self.stage2_cfg = extra['STAGE2']
        block = block_dict[self.stage2_cfg['BLOCK']]

        self.out_channels = self.stage2_cfg['NUM_CHANNELS']
        self.out_channels = [self.out_channels[i] * block.expansion for i in range(len(self.out_channels))]
        self.transition1 = self._make_transition_layer()

        self.in_channels = copy.deepcopy(self.out_channels)
        self.stage2 = self._make_stage(self.stage2_cfg)

        # stage 3
        self.stage3_cfg = extra['STAGE3']
        block = block_dict[self.stage3_cfg['BLOCK']]

        self.in_channels = self.out_channels
        self.out_channels = self.stage3_cfg['NUM_CHANNELS']
        self.out_channels = [self.out_channels[i] * block.expansion for i in range(len(self.out_channels))]
        self.transition2 = self._make_transition_layer()

        self.in_channels = copy.deepcopy(self.out_channels)
        self.stage3 = self._make_stage(self.stage3_cfg)

        # stage 4
        self.stage4_cfg = extra['STAGE4']
        block = block_dict[self.stage4_cfg['BLOCK']]

        self.in_channels = self.out_channels
        self.out_channels = self.stage4_cfg['NUM_CHANNELS']
        self.out_channels = [self.out_channels[i] * block.expansion for i in range(len(self.out_channels))]
        self.transition3 = self._make_transition_layer()

        self.in_channels = copy.deepcopy(self.out_channels)
        self.stage4 = self._make_stage(self.stage4_cfg, multi_scale_output=False)

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

    # first layer with no fuse layer
    def _make_layer(self, block, in_channels, num_blocks, stride=1):
        downsample = None
        # self.in_channels = 64, in_channels = 64
        if stride != 1 or self.in_channels != in_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    in_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels * block.expansion, momentum=BN_MOMENTUM)
            )
        
        branch_layers = [block(self.in_channels, in_channels, stride, downsample)]
        self.in_channels = in_channels * block.expansion
        for _ in range(1, num_blocks):
            branch_layers.append(block(self.in_channels, in_channels))
        
        return nn.Sequential(*branch_layers)
    

    def _make_transition_layer(self):
        num_branches_pre = len(self.in_channels)
        num_branches_post = len(self.out_channels)

        transition_layers = []
        for post in range(num_branches_post):
            if post < num_branches_pre:
                if self.in_channels[post] != self.out_channels[post]:
                    transition_layers.append(nn.Sequential(
                        # conv 3, 1, 1, False
                        conv3x3(self.in_channels[post], self.out_channels[post]),
                        nn.BatchNorm2d(self.out_channels[post], momentum=0.1),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                # here we only consider using pre_last downsampling to form post_{last+1}
                # post + 1 - num_branches_pre always be 1
                # we can change it to num_branches_pre and update transition method to multi-scale fusion
                for pre in range(post + 1 - num_branches_pre):
                    in_channels_3x3 = self.in_channels[-1]
                    out_channels_3x3 = self.out_channels[post] \
                        if pre == post - num_branches_pre else in_channels_3x3
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels_3x3,
                            out_channels_3x3,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False),
                        nn.BatchNorm2d(out_channels_3x3, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        
        return nn.ModuleList(transition_layers)
    

    def _make_stage(self, layer_config, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_out_channels = layer_config['NUM_CHANNELS']

        block = block_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for _ in range(num_modules):
            if not multi_scale_output and _ == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            
            modules.append(
                HighResolutionModule(
                    num_branches=num_branches,
                    block=block,
                    num_blocks=num_blocks,
                    num_in_channels=self.in_channels,
                    num_out_channels=num_out_channels,
                    fuse_method=fuse_method,
                    multi_scale_output=reset_multi_scale_output
                )
            )
            self.in_channels = modules[-1].get_in_channels()

        return nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        # stage 2
        x_list = []
        for branch_index in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[branch_index] is not None:
                x_list.append(self.transition1[branch_index](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        # stage 3
        x_list = []
        for branch_index in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[branch_index] is not None:
                if branch_index == self.stage3_cfg['NUM_BRANCHES'] - 1:
                    x_list.append(self.transition2[branch_index](y_list[-1]))
                else:
                    x_list.append(self.transition2[branch_index](y_list[branch_index]))
            else:
                x_list.append(y_list[branch_index])
        
        y_list = self.stage3(x_list)

        # stage 4
        x_list = []
        for branch_index in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[branch_index] is not None:
                if branch_index == self.stage4_cfg['NUM_BRANCHES'] - 1:
                    x_list.append(self.transition3[branch_index](y_list[-1]))
                else:
                    x_list.append(self.transition3[branch_index](y_list[branch_index]))
            else:
                x_list.append(x_list[branch_index])
        
        y_list = self.stage4(x_list)

        # default: output the highest one
        return _head(y_list, self.output_method)
    
    def init_weights(self, pretrained='', print_load_info=False):
        logger.info('=> init HRnet weights forming as normal distribution')
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
                        print(f":: {name} is loaded from {pretrained}")
            self.load_state_dict(existing_state_dict, strict=False)
        
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError(f'{pretrained} is not exist!')


def get_pose_net(cfg, is_train, **kwargs):
    model = HRNetBase(cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, cfg)
    return model