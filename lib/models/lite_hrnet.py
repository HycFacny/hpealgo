from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
from pathlib import Path
from tokenize import String

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from timm.models.layers.weight_init import trunc_normal_

from config.model import MODEL_EXTRAS
from models.network_modules import BasicBlock
from models.network_modules import Bottleneck
from models.bricks import _get_activation_fn
from models.bricks import _get_activation_layer
from models.bricks import _get_norm_layer
from models.bricks import channel_shuffle
from utils.print_functions import print_inter_debug_info


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
block_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


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


class SpatialWeighting(nn.Module):
    def __init__(self, channels, ratio=16, activations=['relu', 'sigmoid']):
        super().__init__()
        if isinstance(activations, str):
            activations = [activations, activations]
        if isinstance(activations, dict) and len(activations) <= 2:
            activations = [activations[0], activations[-1]]
        assert isinstance(activations, list) and len(activations) == 2
        
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            channels,
            channels // ratio,
            kernel_size=1,
            stride=1
        )
        self.activation1 = _get_activation_fn(activations[0])
        self.conv2 = nn.Conv2d(
            channels // ratio,
            channels,
            kernel_size=1,
            stride=1
        )
        self.activation2 = _get_activation_fn(activations[1])

    def forward(self, x):
        out = self.aap(x)
        out = self.conv1(out)
        out = self.activation1(out)
        out = self.conv2(out)
        out = self.activation2(out)
        return x * out


class CrossResolutionWeighting(nn.Module):
    def __init__(self, channels, ratio=16, activations=['relu', 'sigmoid'], norm='bn'):
        super().__init__()
        if isinstance(activations, str):
            activations = [activations, activations]
        if isinstance(activations, dict) and len(activations) <= 2:
            activations = [activations[0], activations[-1]]
        assert isinstance(activations, list) and len(activations) == 2
        
        self.channels = channels
        sum_channels = sum(channels)
        self.conv1 = nn.Conv2d(
            sum_channels,
            sum_channels // ratio,
            kernel_size=1,
            stride=1
        )
        self.norm1 = _get_norm_layer(sum_channels // ratio, norm)
        self.activation1 = _get_activation_fn(activations[0])
        self.conv2 = nn.Conv2d(
            sum_channels // ratio,
            sum_channels,
            kernel_size=1,
            stride=1
        )
        self.norm2 = _get_norm_layer(sum_channels, norm)
        self.activation2 = _get_activation_fn(activations[1])        

    def forward(self, x):
        min_size = x[-1].size()[-2:]
        out = [F.adaptive_avg_pool2d(s, min_size) for s in x[:-1]] + [x[-1]]
        out = torch.cat(out, dim=1)
        
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.activation1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation2(out)

        out = torch.split(out, self.channels, dim=1)
        out = [
            s * F.interpolate(a, size=s.size()[-2:], mode='nearest')
                for s, a in zip(x, out)
        ]
        return out


class ConditionalChannelWeighting(nn.Module):
    def __init__(self, in_channels, stride, ratio, norm='bn'):
        super().__init__()
        assert stride in [1, 2]
        
        self.stride = stride
        branch_channels = [channel // 2 for channel in in_channels]
        
        self.cross_resolution_weighting = CrossResolutionWeighting( branch_channels, ratio=ratio, norm=norm )
       
        self.depthwise_conv = nn.ModuleList([
            nn.Conv2d(
                channel, channel, kernel_size=3, stride=self.stride, padding=1, groups=channel
            ) for channel in branch_channels
        ])
        
        self.spatial_weighting = nn.ModuleList([
            SpatialWeighting(
                channels=channel, ratio=4
            ) for channel in branch_channels
        ])
    
    def forward(self, x):
        x = [s.chunk(2, dim=1) for s in x]
        x1 = [s[0] for s in x]
        x2 = [s[1] for s in x]
        x2 = self.cross_resolution_weighting(x2)
        x2 = [dw(s) for s, dw in zip(x2, self.depthwise_conv)]
        x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]

        out = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]
        out = [channel_shuffle(s, 2) for s in out]
        
        return out


class Stem(nn.Module):
    def __init__(self, in_channels, stem_channels, out_channels, expand_ratio, norm='bn'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1),
            _get_norm_layer(stem_channels, norm),
            nn.ReLU(inplace=True)
        )
        
        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=2, padding=1, groups=branch_channels),
            _get_norm_layer(branch_channels, norm),
            nn.Conv2d(branch_channels, inc_channels, kernel_size=1, stride=1, padding=0),
            _get_norm_layer(branch_channels, norm),
            nn.ReLU(inplace=True)
        )
        
        self.expansion = nn.Sequential(
            nn.Conv2d(branch_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            _get_norm_layer(mid_channels, norm),
            nn.ReLU(inplace=True)
        )
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels),
            _get_norm_layer(mid_channels, norm)
        )
        
        self.linear = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                branch_channels if stem_channels == self.out_channels else stem_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            _get_norm_layer(
                branch_channels if stem_channels == self.out_channels else stem_channels,
                norm
            ),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)
        
        x2 = self.expansion(x2)
        x2 = self.depthwise(x2)
        x2 = self.linear(x2)
        
        out = torch.cat( (self.branch1(x1), x2), dim=1 )
        out = channel_shuffle(out, 2)
        
        return out
        

class IterativeHead(nn.Module):
    def __init__(self, in_channels, norm='bn'):
        super().__init__()
        projects = []
        num_branches = len(in_channels)
        self.in_channels = in_channels[::-1]
        
        for i in range(num_branches):
            if i != num_branches - 1:
                projects.append(nn.Sequential(
                    nn.Conv2d(self.in_channels[i], self.in_channels[i], kernel_size=3, stride=1, padding=1, groups=self.in_channels[i]),
                    _get_norm_layer(self.in_channels[i], norm),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.in_channels[i], self.in_channels[i + 1], kernel_size=1, stride=1, padding=0),
                    _get_norm_layer(self.in_channels[i + 1], norm),
                    nn.ReLU(inplace=True)
                ))
            else:
                projects.append(nn.Sequential(
                    nn.Conv2d(self.in_channels[i], self.in_channels[i], kernel_size=3, stride=1, padding=1, groups=self.in_channels[i]),
                    _get_norm_layer(self.in_channels[i], norm),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.in_channels[i], self.in_channels[i], kernel_size=1, stride=1, padding=0),
                    _get_norm_layer(self.in_channels[i], norm),
                    nn.ReLU(inplace=True)
                ))

        self.projects = nn.ModuleList(projects)
    
    def forward(self, x):
        x = x[::-1]
        y = []
        last_x = None
        
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(last_x, size=s.size()[-2:], mode='bilinear', align_corners=True)
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s
        
        return y[::-1]


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm='bn', activation='relu'):
        super().__init__()
        self.stride = stride
        branch_features = out_channels // 2
        
        if self.stride == 1:
            assert in_channels == branch_features * 2, (
                f'in_channels ({in_channels}) should equal to '
                f'branch_features * 2 ({branch_features * 2}) '
                'when stride is 1'
            )
        
        if in_channels != branch_features * 2:
            assert self.stride != 1, (
                f'stride ({self.stride}) should not equal 1 when '
                f'in_channels != branch_features * 2'
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
                _get_norm_layer(in_channels, norm),
                _get_activation_layer(activation),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
                _get_norm_layer(in_channels, norm),
                _get_activation_layer(activation),
            )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels if stride > 1 else branch_features,
                branch_features,
                kernel_size=1, stride=1, padding=0
            ),
            _get_norm_layer(branch_features, norm),
            _get_activation_layer(activation),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=stride, padding=1, groups=branch_features),
            _get_norm_layer(branch_features, norm),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0),
            _get_norm_layer(branch_features, norm),
            _get_activation_layer(activation),
        )
    
    def forward(self, x):
        if self.stride > 1:
            out = torch.cat( (self.branch1(x), self.branch2(x)), dim=1 )
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat( (x1, self.branch2(x2)), dim=1 )
        
        out = channel_shuffle(out, 2)
        
        return out


class LiteHRModule(nn.Module):
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

    def __init__(self, num_branches, num_blocks, in_channels, reduce_ratio,
                 multi_scale_output=False, with_fuse=True, norm='bn'):
        super(LiteHRModule, self).__init__()
        self._check_branches(num_branches, in_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches
        self.num_blocks = num_blocks
        self.reduce_ratio = reduce_ratio
        self.multi_scale_output = multi_scale_output
        
        self.with_fuse = with_fuse
        self.norm = norm
        # origin hrnet owns several branches, while lite-hrnet compress them together
        self.layers = self._make_weighting_blocks()
        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU(inplace=True)
    
    def _check_branches(self, num_branches, in_channels):
        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)
    
    def _make_weighting_blocks(self, stride=1):
        layers = []
        for i in range(self.num_blocks[0]):
            layers.append(
                ConditionalChannelWeighting(
                    self.in_channels,
                    stride=stride,
                    ratio=self.reduce_ratio,
                    norm=self.norm
                )
            )
        return nn.Sequential(*layers)
    
    def _make_fuse_layers(self):
        if self.num_branches == 1: return None

        fuse_layers = []
        # pre post layers k*k matric
        for post in range(self.num_branches if self.multi_scale_output else 1):
            layer = []
            for pre in range(self.num_branches):
                in_channel = self.in_channels[pre]
                out_channel = self.in_channels[post]
                if pre > post:
                    layer.append(nn.Sequential(
                        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
                        _get_norm_layer(out_channel, self.norm),
                        nn.Upsample(scale_factor=2 ** (pre - post), mode='nearest')
                    ))
                    # global cnt
                    # cnt += 1
                elif pre < post:
                    conv3x3s = []
                    for cur in range(post - pre):
                        if cur == post - pre - 1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1, groups=in_channel, bias=False),
                                _get_norm_layer(in_channel, self.norm),
                                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
                                _get_norm_layer(out_channel, self.norm)
                            ))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1, groups=in_channel, bias=False),
                                _get_norm_layer(in_channel, self.norm),
                                nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0, bias=False),
                                _get_norm_layer(in_channel, self.norm),
                                nn.ReLU(inplace=True)
                            ))
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
        return self.in_channels
    
    def forward(self, x):
        if self.num_branches == 1: return [self.branches[0](x[0])]
        # print(self.num_branches)
        # print(self.branches)
        '''
        origin hrnet:
        for branch_index in range(self.num_branches):
            x[branch_index] = self.branches[branch_index](x[branch_index])
        '''
        x = self.layers(x)
        
        if self.with_fuse:
            x_fuse = []
            for post in range(len(self.fuse_layers)):
                # post += pre
                y = x[0] if post == 0 else self.fuse_layers[post][0](x[0])
                for pre in range(self.num_branches):
                    if post == pre: y = y + x[pre]
                    else:
                        tmp = self.fuse_layers[post][pre](x[pre])
                        y += tmp
                x_fuse.append(self.relu(y))
        elif not self.multi_scale_output:
            x = [x[0]]
        
        return x_fuse


class LiteHRNet(nn.Module):    
    def __init__(self, cfg, **kwargs):
        super().__init__()
        extra = MODEL_EXTRAS['pose_lite_hrnet']
        self.output_method = cfg['MODEL']['OUTPUT_METHOD'] \
            if 'OUTPUT_METHOD' in cfg['MODEL'] else 'highest'
        
        self.num_stage = extra['NUM_STAGES']
        self.reduce_ratios = extra['REDUCE_RATIOS']
        self.stem_channels = extra['STEM_INCHANNELS']
        self.in_channels = extra['STEM_OUTCHANNELS']
        self.zero_init_residual = extra['ZERO_INIT_RESIDUAL']
        self.with_head = extra['WITH_HEAD']
        self.norm = extra['NORM']
        
        self.stem = Stem(
            in_channels=3,
            stem_channels=self.stem_channels,
            out_channels=self.in_channels,
            expand_ratio=extra['EXPAND_RATIO'],
            norm=self.norm
        )
        self.in_channels = [self.in_channels]
        
        # stage 2
        self.stage2_cfg = extra['STAGE2']
        self.out_channels = self.stage2_cfg['NUM_CHANNELS']
        self.transition1 = self._make_transition_layer()
        self.in_channels = copy.deepcopy(self.out_channels)
        self.stage2 = self._make_stage(self.stage2_cfg, self.reduce_ratios[0])
        
        # stage 3    
        self.stage3_cfg = extra['STAGE3']
        self.out_channels = self.stage3_cfg['NUM_CHANNELS']
        self.transition2 = self._make_transition_layer()
        self.in_channels = copy.deepcopy(self.out_channels)
        self.stage3 = self._make_stage(self.stage3_cfg, self.reduce_ratios[1])
        
        # stage 4
        self.stage4_cfg = extra['STAGE4']
        self.out_channels = self.stage4_cfg['NUM_CHANNELS']
        self.transition3 = self._make_transition_layer()
        self.in_channels = copy.deepcopy(self.out_channels)
        self.stage4 = self._make_stage(self.stage4_cfg, self.reduce_ratios[2])

        if self.with_head:
            self.head_layer = IterativeHead(in_channels=self.in_channels)

        # self.pretrained_layers = extra['PRETRAINED_LAYERS']

    def _make_transition_layer(self):
        num_branches_pre = len(self.in_channels)
        num_branches_post = len(self.out_channels)

        transition_layers = []
        for post in range(num_branches_post):
            if post < num_branches_pre:
                if self.in_channels[post] != self.out_channels[post]:
                    transition_layers.append(nn.Sequential(
                        # conv 3, 1, 1, False
                        nn.Conv2d(self.in_channels[post], self.in_channels[post], kernel_size=3, stride=1, padding=1, groups=self.in_channels[post], bias=False),
                        _get_norm_layer(self.in_channels[post]),
                        nn.Conv2d(self.in_channels[post], self.out_channels[post], kernel_size=1, stride=1, padding=0, bias=False),
                        _get_norm_layer(self.out_channels[post]),
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
                        nn.Conv2d(in_channels_3x3, in_channels_3x3, kernel_size=3, stride=2, padding=1, groups=in_channels_3x3, bias=False),
                        _get_norm_layer(in_channels_3x3),
                        nn.Conv2d(in_channels_3x3, out_channels_3x3, kernel_size=1, stride=1, padding=0, bias=False),
                        _get_norm_layer(out_channels_3x3),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        
        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, reduce_ratio, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']

        modules = []
        for _ in range(num_modules):
            # upon on the last layer, we dont use multi output
            if not multi_scale_output and _ == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
                
                # def __init__(self, num_branches, num_blocks, in_channels, reduce_ratio,
                # multi_scale_output=False, with_fuse=True, norm='bn'):
                
            # LiteHRModule(num_branches, num_blocks, in_channels, reduce_ratio, module_type, multi_scale_output=False, with_fuse=True, norm='bn')
            modules.append(LiteHRModule(
                num_branches=num_branches,
                num_blocks=num_blocks,
                in_channels=self.in_channels,
                reduce_ratio=reduce_ratio,
                multi_scale_output=reset_multi_scale_output,
                with_fuse=True,
                norm='bn'
            ))
            self.in_channels = modules[-1].get_in_channels()

        return nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.stem(x)

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
                x_list.append(y_list[branch_index])
        y_list = self.stage4(x_list)
        
        if self.with_head:
            y_list = self.head_layer(y_list)

        return [y_list[0]]
    
    def init_weights(self, pretrained='', print_load_info=False):
        logger.info('=> init HRnet weights forming as normal distribution')
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                trunc_normal_(module.weight, std=.02)
            if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1.)
                nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.ConvTranspose2d):
                trunc_normal_(module.weight, std=.02)
                nn.init.constant_(module.bias, 0)
                
        if self.zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2, 0)
                elif isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3, 0)
        
        if pretrained != '' and Path(pretrained).is_file():
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


def get_pose_net(cfg, is_train, **kwargs):
    model = LiteHRNet(cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, cfg)
    return model