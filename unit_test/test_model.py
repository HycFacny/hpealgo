from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch

import _init_paths
from utils.utils import get_model_summary
from models.lite_hrnet import get_pose_net
from yacs.config import CfgNode as CN
from config import cfg
from config import update_config
from utils.print_functions import print_inter_debug_info


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', default='experiments/coco/lite-hrnet/w48_256x192_adam_lr1e-3.yaml',
                        type=str)
    parser.add_argument('opts', help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    # philly
    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument('--dataDir', help='data directory',  type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')

    args = parser.parse_args()

    return args


def init_cfg():
    args = parse_args()
    update_config(cfg, args)


def test_hrnet():
    model = get_pose_net(cfg, True)
    print_inter_debug_info('lite_hrnet model: ', model, 'entire_network')
    dump_input = torch.rand((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])).cuda()
    model = torch.nn.DataParallel(model).cuda()
    out = model(dump_input)
    

if __name__ == '__main__':
    init_cfg()
    test_hrnet()
