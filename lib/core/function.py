from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from core.evaluate import get_accuracy
from core.evaluate import get_final_preds
from utils.transforms import flip_back
from utils.visualizer import save_debug_images
from utils.print_functions import print_inter_debug_info
from utils.print_functions import print_inter_name_value


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


def train(cfg, train_loader, model, criterion, optimizer, epoch, output_dir, tensorboard_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # switch to train mode
    model.train()

    begin = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # data loading time
        data_time.update(time.time() - begin)
        # print(data_time.val)
        # print_inter_debug_info('batch_input_size', input.shape, 'input')
        # print_inter_debug_info('batch_target_size', target.shape, 'input')

        # compute output
        # torch.autograd.set_detect_anomaly(True)
        print(input.size(), input.device)
        outputs = model(input)
        # print(time.time() - begin - data_time.val)
        # print(1)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            loss = criterion(outputs, target, target_weight)
        
        # compute gradient and update
        optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        loss.backward()
        optimizer.step()

        # print(2)

        # measure accuracy and record loss
        _, avg_accuracy, cnt, pred = get_accuracy(outputs.detach().cpu().numpy(), target.detach().cpu().numpy())
        print_inter_debug_info('loss_item', loss.item(), 'training_phase')
        print_inter_debug_info('input_size_per_sample', input.size(0), 'training_phase')
        losses.update(loss.item(), input.size(0))
        accuracy.update(avg_accuracy, cnt)
        batch_time.update(time.time() - begin)
        
        if i % cfg.PRINT_FREQ == 0:
            messagebox = 'Epoch: [{0}][{1}/{2}]\t' \
                         'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                         'Speed {speed:.1f} samples/s\t' \
                         'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                         'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                         'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                             epoch, i, len(train_loader), batch_time=batch_time,
                             speed=input.size(0) / batch_time.val,
                             data_time=data_time,
                             loss=losses,
                             accuracy=accuracy
                         )
            logger.info(messagebox)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_accuracy', accuracy.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
            
            prefix = '{}_{}'.format(str(Path(output_dir, 'train')), i)
            save_debug_images(cfg, input, meta, target, pred * 4, outputs, prefix)

        begin = time.time()


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

