from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import width

import cv2
import numpy as np


def gaussian_blur(batch_heatmaps, kernel):
    """
    Gaussian Blur

    Args:
        batch_heatmaps (ndarray([batch_size, num_joints, height, width])): Batch heatmaps
        kernel (int): Gaussian kernel
        
    Returns:
        batch_heatmaps (ndarray([batch_size, num_joints, height, width]))
    """
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    height, width = batch_heatmaps.shape[2], batch_heatmaps.shape[3]
    
    border = (kernel - 1) // 2
    bmap = np.zeros( (height + 2 * border, width + 2 * border) )
    
    for i in range(batch_size):
        for j in range(num_joints):
            _max = np.max(batch_heatmaps[i, j])
            bmap = bmap * 0.
            bmap[border : -border, border : -border] = batch_heatmaps[i, j].copy()
            bmap = cv2.GaussianBlur(bmap, (kernel, kernel), 0)
            batch_heatmaps[i, j] = bmap[border : -border, border : -border].copy()
            batch_heatmaps[i, j] *= _max / np.amax(batch_heatmaps[i, j])

    return batch_heatmaps