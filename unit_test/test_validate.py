from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pprint
from yacs.config import CfgNode as CN

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
import dataset
import models

from config import cfg
from config import update_config
from utils.utils import get_model_summary
from utils.logger import create_logger
from models.hrnet_base import get_pose_net
from core.loss import JointsMSELoss
from core.function import validate



args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', default='experiments/coco/tokenpose/tokenpose_L_D24_256_192_patch43_dim192_depth24_heads12.yaml',
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
    global args
    args = parse_args()
    update_config(cfg, args)


def test_validate():
    global args
    logger, final_output_dir, tensorboard_log_dir = create_logger(cfg, args.cfg, 'valid')
    logger.info(cfg)
    
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * 1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
    
    validate(cfg, valid_loader, valid_dataset, model,
             criterion, final_output_dir, tensorboard_log_dir)
    

if __name__ == '__main__':
    init_cfg()
    test_validate()
  