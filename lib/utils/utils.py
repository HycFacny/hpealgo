from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from pathlib import Path
from collections import namedtuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


def get_projroot():
    root = Path.cwd()
    while str(root).split('/')[-1] != 'hpealgo':
        root = root.parent
    return str(root)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
            # weight_decay=cfg.TRAIN.WD
        )
    
    return optimizer


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """ use torch.hook to record whole network param situation """
    
    summary = []        # all module_details, format as below
    module_details = namedtuple(
        'Layer',
        ['name', 'input_size', 'output_size', 'num_parameters', 'multiply_adds']
    )                   # all module infomation

    hooks = []          # hooks of all module
    layer_instances = {}    # { layer_name: num_layer }

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)
            instance_count = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_count
            else:
                instance_count = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_count
            
            layer_name = class_name + '_' + str(instance_count)
            params = 0

            if class_name.find('Conv') != -1 or class_name.find('BatchNorm') != -1 or \
                class_name.find('Linear') != -1:
                for param in module.parameters():
                    params += param.view(-1).size(0)
            
            # calc all parameters in current module
            flops = 'Not Available'
            if class_name.find('Conv') != -1 and hasattr(module, 'weight'):
                flops = (
                    torch.prod(torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(torch.LongTensor(list(output.size())[2:]))
                ).item()
            elif class_name.find('Linear') != -1:
                flops = (torch.prod(torch.LongTensor(list(output.size()))) * input[0].size(1)).item()
            
            if isinstance(input[0], list): input = input[0]
            if isinstance(output, list): output = output[0]
            summary.append(module_details(
                name=layer_name,
                input_size=list(input[0].size()),
                output_size=list(output.size()),
                num_parameters=params,
                multiply_adds=flops
            ))
        
        if not isinstance(module, nn.ModuleList) and not isinstance(module, nn.Sequential) \
            and module != model:
            hooks.append(module.register_forward_hook(hook))
    
    model.eval()
    # apply add_hooks fn to collect and record information
    model.apply(add_hooks)
    # output indent
    space_len = item_length

    model(*input_tensors)
    for hook in hooks: hook.remove()

    details = ''
    if verbose:
        details = 'Model Summary' + \
            os.linesep + \
            'Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}'.format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
            + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += \
                "{}{}{}{}{}{}{}{}{}{}".format(
                    layer.name,
                    ' ' * (space_len - len(layer.name)),
                    layer.input_size,
                    ' ' * (space_len - len(str(layer.input_size))),
                    layer.output_size,
                    ' ' * (space_len - len(str(layer.output_size))),
                    layer.num_parameters,
                    ' ' * (space_len - len(str(layer.num_parameters))),
                    layer.multiply_adds,
                    ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    torch.save(states, str(Path(output_dir) / filename))
    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'], str(Path(output_dir) / 'model_best_perf.pth'))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)