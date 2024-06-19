from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from audioop import avg

import numpy as np

from utils.blur import gaussian_blur
from utils.transforms import get_affine_transmat
from utils.transforms import joint_affine_transform


def get_accuracy(heatmaps_pred, heatmaps_target, type='gaussian', thresh=0.5):
    """
    Calculate arruracy PCK@thres factors using gt_heatmaps rather than x, y locations
    
    Args:
        heatmaps_pred ( ndarray(batch_size, num_joints, hmp_height, hmp_width) ): Predicted heatmaps
        heatmaps_target ( same as before ): Ground truth heatmaps
        type (str, optional): Blur type. Defaults to 'gaussian'.
        thresh (float, optional): . Defaults to 0.5.
        
    Returns:

    """
    num_joints = heatmaps_pred.shape[1]
    height, width = heatmaps_pred.shape[2], heatmaps_pred.shape[3]
    norm = 1.

    if type == 'gaussian':
        # calculate location of max confidience of pred and gt
        # [batch_size, num_joints, 2], ...[:, :, 0] -> width, ...[:, :, 1] -> height
        heatmaps_pred, _ = get_max_pred_locations(heatmaps_pred)
        heatmaps_target, _ = get_max_pred_locations(heatmaps_target)
        norm = np.ones( (heatmaps_pred.shape[0], 2) ) * np.array([height, width]) / 10 # [[6.4, 4.8]]
    
    dists = get_heatmap_dist(heatmaps_pred, heatmaps_target, norm)
    # print(dists)
    accuracy = np.zeros(num_joints + 1)
    avg_acc = 0.
    cnt = 0
    
    for idx in range(num_joints):
        accuracy[idx + 1] = get_dist_accuracy(dists[idx])
        if accuracy[idx + 1] >= 0.:
            avg_acc += accuracy[idx + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt > 0 else 0
    if cnt > 0: accuracy[0] = avg_acc
    return accuracy, avg_acc, cnt, heatmaps_pred


def get_final_preds(cfg, batch_heatmaps, center, scale):
    """
    Calculate the final preds processed through blur, tylor and transform

    Args:
        batch_heatmaps ( ndarray(batch_size, joints, hmp_h, hmp_w) ): Batch heatmap preds
        center ( ndarray(batch_size, 1) ): Batch center
        scale ( ndarray(batch_size, 1) ): Batch scale
    """
    locations, maxvals = get_max_pred_locations(batch_heatmaps)
    height, width = batch_heatmaps.shape[2], batch_heatmaps.shape[3]
    preds = locations.copy()
    
    # post processing
    if cfg.TEST.POST_PROCESS:
        heatmaps = gaussian_blur(batch_heatmaps, cfg.TEST.BLUR_KERNEL)
        heatmaps = np.maximum(heatmaps, 1e-10)
        heatmaps = np.log(heatmaps)
        
        # calculate neighborbood derivation of locations to ajust them to relative confident values
        for batch_idx in range(preds.shape[0]):
            for joint_idx in range(preds.shape[1]):
                locations[batch_idx, joint_idx] = \
                    tylor(heatmaps[batch_idx, joint_idx], locations[batch_idx, joint_idx])
    
    preds = locations.copy()
    
    for _ in range(preds.shape[0]):
        preds[_] = get_transform_preds(
            locations[_], center[_], scale[_], [width, height]
        )
    
    return preds, maxvals


def get_max_pred_locations(batch_heatmaps):
    """
    Get prediction locations from score maps
    
    Args:
        batch_heatmaps (ndarray([batch_size, num_joints, height, width])): Batch heatmaps

    Returns:
        preds (ndarray([batch_size, num_joints, 2])): Locations of heatmap maxvals ( w, h )
        maxvals (ndarray([batch_size, num_joints, 1])): Maxvals of heatmap
    """
    assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))

    # index of the pixel with max joint pred of each joint in each heatmap
    # idx: [5, 17, 1]
    idx = np.argmax(heatmaps_reshaped, 2).reshape((batch_size, num_joints, 1))
    # maxval of ...
    # maxvals: [5, 17, 1]
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((batch_size, num_joints, 1))

    # preds: [5, 17, 2], preds[:, :, 0] -> height, preds[:, :, 1] -> width
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    
    preds[:, :, 0] = preds[:, :, 0] % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    # pred_mask, with all preds > 0.
    pred_mask = np.tile(np.greater(maxvals, 0.), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_heatmap_dist(pred, gt, norm):
    """
    Calculate similarity matrix between pred locations and gt locations
    
    Args:
        pred (ndarray([batch_size, num_joints, 2])): Locations of heatmap maxvals
        gt (ndarray([batch_size, num_joints, 2])): Locations of gt maxvals
        norm (ndarray([])): 

    Returns:
        dists (ndarray([num_joints, batch_size])): L2 norm between preds and gts
    """
    '''  '''
    batch_size = int(pred.shape[0])
    num_joints = int(pred.shape[1])
    dists = np.zeros((num_joints, batch_size), dtype=np.float32)
    
    for batch_idx in range(batch_size):
        for joint_idx in range(num_joints):
            if gt[batch_idx, joint_idx, 0] > 1 and gt[batch_idx, joint_idx, 1] > 1:
                normed_preds = pred[batch_idx, joint_idx, :] / norm[batch_idx]
                normed_gt = gt[batch_idx, joint_idx, :] / norm[batch_idx]
                dists[joint_idx, batch_idx] = np.linalg.norm(normed_preds - normed_gt)
            else:
                dists[joint_idx, batch_idx] = -1
    
    return dists


def get_dist_accuracy(dists, thresh=.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dists_valid = np.not_equal(dists, -1)
    num_dists_valid = dists_valid.sum()
    if num_dists_valid > 0:
        return np.less( dists[dists_valid], thresh ).sum() / num_dists_valid
    else:
        return -1


def get_transform_preds(locations, center, scale, output_size):
    """
    Transform back preds with transformat
    
    Args:
        locations (ndarray([num_joints, 2]))
        center (list[2])
        scale (list[2])
        output_size (list[2]): [height, width]

    Returns:
        target_locations (ndarray([num_joints, 2]))
    """
    target_locations = np.zeros(locations.shape)
    trans = get_affine_transmat(center, scale, 0, output_size, inv=1)
    
    for joint_idx in range(locations.shape[0]):
        target_locations[joint_idx, 0 : 2] = joint_affine_transform(locations[joint_idx, 0 : 2], trans)
    
    return target_locations


def tylor(heatmap, locations):
    ''' alculate neighborbood derivation of locations to ajust them to relative confident values '''
    height, width = heatmap.shape[0], heatmap.shape[1]
    point_x, point_y = int(locations[1]), int(locations[0])
    
    if 1 < point_x < height - 2 and 1 < point_y < width - 2:
        dx  = (heatmap[point_x, point_y + 1] - heatmap[point_x, point_y - 1]) / 2.
        dy  = (heatmap[point_x + 1, point_y] - heatmap[point_x - 1, point_y]) / 2.
        dxx = (heatmap[point_x, point_y + 2] + heatmap[point_x, point_y - 2] - 2 * heatmap[point_x, point_y]) / 4.
        dyy = (heatmap[point_x + 2, point_y] + heatmap[point_x - 2, point_y] - 2 * heatmap[point_x, point_y]) / 4.
        dxy = (heatmap[point_x + 1, point_y + 1] + heatmap[point_x - 1, point_y - 1] \
            -  heatmap[point_x + 1, point_y - 1] - heatmap[point_x - 1, point_y + 1]) / 4.
        
        hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
        derivative = np.matrix([[dx], [dy]])
        
        if dxx * dyy - dxy * dxy != 0:
            hessian_inv = hessian.I
            offset = (-hessian_inv * derivative)
            offset = np.squeeze(np.array(offset.T), axis=0)
            locations += offset
        
    return locations
