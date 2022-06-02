from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pprint
import shutil
from pathlib import Path
import os

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.logger import create_logger
from utils.utils import get_projroot
from utils.utils import get_optimizer
from utils.utils import get_model_summary
from utils.utils import save_checkpoint
from utils.print_functions import print_inter_debug_info


import dataset
import models

def main():
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tensorboard_log_dir = create_logger(cfg, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(cfg)
    
    print(final_output_dir, tensorboard_log_dir)

    # cudnn setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # create and initialize model
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=True
    )
    
    # print_inter_debug_info('model', model, 'entire_network')

    # record model file
    shutil.copy2(
        str(Path(get_projroot()) / 'lib' / 'models' / '{}.py'.format(cfg.MODEL.NAME)),
        final_output_dir
    )

    writer_dict = {
        'writer': SummaryWriter(log_dir=tensorboard_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0
    }

    # random generate an input to calc network params ( one batch )
    dump_input = torch.rand((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))
    logger.info(get_model_summary(model, dump_input))
'''
    # model
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    # dataset
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensors(),
            normalize
        ])
    )

    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensors(),
            normalize
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = Path(final_output_dir) / 'checkpoint.pth'

    if cfg.AUTO_RESUME and checkpoint_file.exists():
        logger.info(f"=> loading checkpoint '{checkpoint_file}'")
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"=> loaded checkpoint '{checkpoint_file}' (epoch {checkpoint['epoch']})")
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    ) if cfg.TRAIN.LR_SCHEDULER is 'MultiStepLR' else torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.TRAIN.END_EPOCH, eta_min=cfg.TRAIN.LR_END, last_epoch=last_epoch)

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        logger.info("=> current learning rate is {:.6f}".format(lr_scheduler.get_last_lr()[0]))
        
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tensorboard_log_dir, writer_dict)

        perf_indicator = validate(
            cfg,  valid_loader, valid_dataset, model, criterion,
            final_output_dir, tensorboard_log_dir, writer_dict
        )

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False
        
        logger.info(f'=> saving checkpoint to {final_output_dir}')
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = Path(final_output_dir) / 'final_state.pth'
    logger.info(f'=> saving final model state to {final_model_state_file}')

    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

'''

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


if __name__ == '__main__':
    main()
