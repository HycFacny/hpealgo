from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

import sys
import logging
import basic_info
from pathlib import Path

from yacs.config import CfgNode as CN
from config import cfg
from config import update_config


yaml_name = 'tokenpose_L_D24_256_192_patch43_dim192_depth24_heads12.yaml'

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', default=None,
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

def get_config_from_yaml():
    args = parse_args()
    yaml_path = Path(basic_info.get_project_root()) / 'experiments' / 'coco' / yaml_name
    args.cfg = yaml_path
    update_config(cfg, args)
    return cfg

# tokenpose_L_D24_256_192_patch43_dim192_depth24_heads12.yaml
def test_load_config():
    get_config_from_yaml()
    print(cfg)


if __name__ == '__main__':
    test_load_config()