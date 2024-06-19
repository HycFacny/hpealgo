from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pprint
import logging
import numpy as np
import time
from pathlib import Path
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
from core.evaluate import get_accuracy
from core.evaluate import get_final_preds
from utils.transforms import flip_back
from utils.visualizer import save_debug_images
from utils.print_functions import print_inter_debug_info
from utils.print_functions import print_inter_name_value


args = None
logger = logging.getLogger(__name__)


class AverageMeter(object):
    """ compute and store the current and average value """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt if self.cnt != 0 else 0


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', default='experiments/crowdpose/tokenpose/tokenpose_L_D24_256_192_patch43_dim192_depth24_heads12.yaml',
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


def validate(cfg, val_loader, val_dataset, model, criterion, output_dir, tensorboard_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, cfg.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))

    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    
    with torch.no_grad():
        begin = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            outputs = model(input)
            if isinstance(outputs, list): output = outputs[-1]
            else: output = outputs

            if cfg.TEST.FLIP_TEST:
                input_flip = np.flip(input.cpu().numpy(), 3).copy()
                input_flip = torch.from_numpy(input_flip).cuda()
                outputs_flip = model(input_flip)

                if isinstance(outputs_flip, list): output_flip = outputs_flip[-1]
                else: output_flip = outputs_flip

                output_flip = flip_back(output_flip.cpu().numpy(), val_dataset.flip_pairs).copy()
                output_flip = torch.from_numpy(output_flip).cuda()

                output = (output + output_flip) * .5
            # if i == 4: break
        
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            loss = criterion(output, target, target_weight)

            num_images = input.size(0)

            _, avg_accuracy, cnt, pred = get_accuracy(output.cpu().numpy(), target.cpu().numpy())
            losses.update(loss.item(), num_images)
            accuracy.update(avg_accuracy, cnt)
            batch_time.update(time.time() - begin)
            
            center = meta['center'].numpy()
            scale = meta['scale'].numpy()
            score = meta['score'].numpy()

            # preds: ndarray(batch_size, num_joints, 2) -> H, W
            # maxvals: ndarray(batch_size, num_joints, 1) -> value of max preds
            # get final preds after tylor correction
            preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), center, scale)
            # print(type(preds), preds.shape)
            # print(type(preds), maxvals.shape)

            all_preds[idx : idx + num_images, :, 0 : 2] = preds[:, :, 0 : 2]
            all_preds[idx : idx + num_images, :, 2 : 3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx : idx + num_images, 0 : 2] = center[:, 0 : 2]
            all_boxes[idx : idx + num_images, 2 : 4] = scale[:, 0 : 2]
            all_boxes[idx : idx + num_images, 4] = np.prod(scale * 200, 1)
            all_boxes[idx : idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % cfg.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}] \t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) \t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f}) \t' \
                        'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time,
                            loss=losses, accuracy=accuracy)
                logger.info(msg)

                prefix = '{}_{}'.format(str(Path(output_dir) / 'val'), i)
                save_debug_images(cfg, input, meta, target, pred * 4, output, prefix)
                
        # print(image_path)
    
        name_values, pref_indicator = val_dataset.evaluate(
            cfg, all_preds, output_dir, all_boxes, image_path, filenames, imgnums
        )

        model_name = cfg.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                print_inter_name_value(name_value, model_name)
        else:
            print_inter_name_value(name_values, model_name)
        
        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_accuracy', accuracy.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid', dict(name_values), global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return pref_indicator


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
    '''
    if cfg.TEST.MODEL_FILE:
        logger.info(f'=> loading model from {cfg.TEST.MODEL_FILE}')
        pretrained_state_dict = torch.load(cfg.TEST.MODEL_FILE)
        existing_state_dict = {}
        for name, m in pretrained_state_dict.items():
            existing_state_dict[name] = m
            print(f'load layer param: {name}')
        model.load_state_dict(existing_state_dict, strict=False)
    
    # for normal test
    # model.load_state_dict(torch.load(os.path.join('../output/coco/pose_resnet/res50_256x192_d256x3_adam_lr1e-3','model_best.pth')), strict=False)
    else:
        model_state_file = str(Path(final_output_dir) / 'model_best_perf.pth')
        logger.info(f'=> loading model from {model_state_file}')
        model.load_state_dict(torch.load(model_state_file))
    '''
   
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
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
    
    validate(cfg, valid_loader, valid_dataset, model,
             criterion, final_output_dir, tensorboard_log_dir)


if __name__ == '__main__':
    init_cfg()
    test_validate()
  