from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import logging
from pathlib import Path

import cv2
import numpy as np
import torchvision

from core.evaluate import get_max_pred_locations
from utils.print_functions import print_inter_debug_info


def save_debug_images(cfg, input, meta, target, joints_pred, heatmap_pred, prefix):
    """
    Save intermediate images in validating phase

    Args:
        input: [batch_size, channels, height, width]
        target: [batch_size, num_joints, height, width]   // heatmaps
        joints_pred: [batch_size, num_joints, 3 (w, h, 0) ]
        prefix: output_dir + 'epoch_info'
    """

    if not cfg.DEBUG.DEBUG: return

    if cfg.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'], '{}_gt.jpg'.format(prefix)
        )

    if cfg.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'], '{}_pred.jpg'.format(prefix)
        )
    
    if cfg.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hmp_gt.jpg'.format(prefix)
        )
    
    if cfg.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, heatmap_pred, '{}_hmp_pred.jpg'.format(prefix)
        )
    

def save_batch_image_with_joints(batch_images, batch_joints, batch_joints_vis,
                                 image_name, nrow=8, padding=2, normalize=True):
    """
    visualize images of one batch
    
    Args:
        batch_image: [batch_size, channel, height, width]
        batch_joints: [batch_size, num_joints, 3] ( w, h, score )
        batch_joints_vis: [batch_size, num_joints, 1]
        nrow, padding are factors for forming outputs
    """
    grid = torchvision.utils.make_grid(batch_images, nrow, padding, normalize=normalize)
    num_images = batch_images.size(0)
    x_maps = min(nrow, num_images)
    y_maps = int(math.ceil(num_images / x_maps))
    height_each = int(batch_images.size(2) + padding)
    width_each = int(batch_images.size(3) + padding)

    image_ndarray = grid.detach().mul(255).clamp(0, 255).permute(1, 2, 0).cpu()
    image_ndarray = image_ndarray.numpy().copy().astype(np.uint8)
    
    image_cnt = 0
    for y in range(y_maps):
        for x in range(x_maps):
            if image_cnt >= num_images: break
            joints, joints_vis = batch_joints[image_cnt], batch_joints_vis[image_cnt]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = int(x * width_each + padding + joint[0])
                joint[1] = int(y * height_each + padding + joint[1])
                if joint_vis[0] > 0:
                    cv2.circle(image_ndarray, (joint[0], joint[1]), 2, [255, 0, 0], 2)
            image_cnt += 1
    
    cv2.imwrite(image_name, image_ndarray)


def save_batch_heatmaps(batch_images, batch_heatmaps, image_name, normalize=True):
    """
    visualize heatmaps of one batch
    
    Args:
        batch_image (tensor([batch_size, channel, height, width]))
        batch_heatmaps (tensor([batch_size, num_joints, height, width]))
    """
    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    hmp_height = batch_heatmaps.size(2)
    hmp_width  = batch_heatmaps.size(3)
    eps = 1e-6

    if normalize:
        _min = float(batch_images.min())
        _max = float(batch_images.max())
        batch_images = batch_images.clone()
        batch_images.add_(-_min).div_(_max - _min + eps)
    
    grid = np.zeros((
        batch_size * hmp_height, 
        (num_joints + 1) * hmp_width,
        3
    ), dtype=np.uint8)
    
    preds, maxvals = get_max_pred_locations(batch_heatmaps.detach().cpu().numpy())

    for idx in range(batch_size):
        image = batch_images[idx].mul(255).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        image = cv2.resize(image, (int(hmp_width), int(hmp_height)) )
        heatmaps = batch_heatmaps[idx].detach().mul(255).clamp(0, 255). cpu().numpy().astype(np.uint8)
        
        height_begin = hmp_height * idx
        height_end = hmp_height * (idx + 1)
        for j in range(num_joints):
            cv2.circle(image, (int(preds[idx][j][0]), int(preds[idx][j][1])), 1, [0, 0, 255], 1)
            
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            masked_image = colored_heatmap * 0.7 + image * 0.3
            cv2.circle(masked_image, (int(preds[idx][j][0]), int(preds[idx][j][1])), 1, [0, 0, 255], 1)

            width_begin = hmp_width * (j + 1)
            width_end = hmp_width * (j + 2)
            grid[height_begin : height_end, width_begin : width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid[height_begin : height_end, 0 : hmp_width, :] = image

    cv2.imwrite(image_name, grid)
