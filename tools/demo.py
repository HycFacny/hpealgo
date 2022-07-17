from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
from pathlib import Path


import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from core.function import demo
from core.function import demo_low_resolution
from utils.logger import create_logger
from utils.demo_utils import demo_dataset

import dataset
import models


def main():
    args = parse_args()
    update_config(cfg, args)
    
    logger, final_output_dir, tensorboard_log_dir = create_logger(cfg, args.cfg, 'demo')
    logger.info(pprint.pformat(args))
    logger.info(cfg)
    
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    print(cfg.MODEL.NAME)
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    
    # prepare data
    datatype = args.datatype
    data_source = args.data_source
    data_output = args.data_output

    # load demo dataset
    demo_set = demo_dataset(cfg, datatype, data_source)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # load model params    
    model_state_file = args.pth
    logger.info(f'=> loading model from {model_state_file}')
    model.load_state_dict(torch.load(model_state_file))
    model = torch.nn.DataParallel(model).cuda()
    
    
    demo_low_resolution(cfg, model, demo_set, transform, data_output)
    


def parse_args():
    parser = argparse.ArgumentParser(description='Keypoints network demo')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', default='experiments/coco/tokenpose/tokenpose_L_D24_256_192_patch43_dim192_depth24_heads12.yaml',
                        type=str)
    parser.add_argument('--pth', help='pretrained model file', default='output/coco/tokenpose_l/tokenpose_L_D24_256_192_patch43_dim192_depth24_heads12/model_best_perf.pth',
                        type=str)
    parser.add_argument('--datatype', help='demo data type ( image or video )', default='image',
                        type=str)
    parser.add_argument('--data_source', help='demo data file path', default='data/demo_input',
                        type=str)
    parser.add_argument('--data_output', help='demo data results saving path', default='data/demo_output',
                        type=str)
    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument('--dataDir', help='data directory',  type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')
    parser.add_argument('opts', help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
