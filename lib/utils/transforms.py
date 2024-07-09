from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def get_affine_transmat(
    center, scale, rotation, output_size,
    shift=np.array([0, 0], dtype=np.float32), inv=0
):
    """
    Methods for JointsDataset
    Center rotate, and shift if asked
    
    Args:
        center (list[2]): Center of current image
        scale (list[2]): Scale of current image ( pixel_std == 200.)
        rotation (int): Degree of rotation
        output_size (list[2]): [width, height]
        shift (int or list, optional): Image shift factor. Defaults to np.array([0, 0], dtype=np.float32).
        inv (int, optional): If is inv operation. Defaults to 0.

    Returns:
        trans (ndarray([2, 3])): Transform matrix with bias
    """

    # print(rotation)

    if not (isinstance(scale, np.ndarray) or isinstance(scale, list)):
        print('scale is {}, type of scale is {}'.format(scale, type(scale)))
        scale = np.array([scale, scale])
    
    scale_after_preprocess = scale * 200.
    src_w = scale_after_preprocess[0]
    dst_w, dst_h = output_size[0], output_size[1]
    
    rotation_rad = np.pi * rotation / 180   # rad representation for the rotation
    src_direction = get_direction([0, (src_w - 1) * -.5], rotation_rad) # for symmetric random, we can either let 2rd point rotate + or -
    dst_direction = np.array([0, (dst_w - 1) * -.5], dtype=np.float32)  # point after rotating in dst
    
    # need 3 points in both to calc affine transform matrix
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    # src may be shifted, but we only consider centered dst map
    src[0, :] = center + scale_after_preprocess * shift                 # src center point as the first
    src[1, :] = center + src_direction + scale_after_preprocess * shift # 
    dst[0, :] = [(dst_w - 1) * .5, (dst_h - 1) * .5]
    dst[1, :] = np.array([ (dst_w - 1) * .5, (dst_h - 1) * .5 ]) + dst_direction

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])        # get 3rd point use same method to ensure symmetrically
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])        # get 3rd point use same method to ensure symmetrically

    # print(src)
    # print(dst)

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    
    # trans (ndarray([2, 3])): transform matrix with bias
    return trans


# methods for joints after affine op
def joint_affine_transform(joint, transmat):
    '''
        joint: [2]
        transmat: [2, 2]
    '''
    new_joint = np.array([ joint[0], joint[1], 1.], dtype=np.float32).T
    new_joint = np.dot(transmat, new_joint)
    
    return new_joint[:2]


# flip joint coordinates
def fliplr_joints(joints, joints_vis, width, joint_pairs):
    # horizontal flip
    joints[:, 0] = width - joints[:, 0] - 1

    for pair in joint_pairs:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :].copy(), joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :].copy(), joints_vis[pair[0], :].copy()
        
    return joints * joints_vis, joints_vis


def flip_back(output_flipped, joint_pairs):
    """
    Flip back flipped outputs

    Args:
        output_flipped (ndarray([batch_size, num_joints, height, width])): Flipped heatmaps outputed by network
        joint_pairs (list[num_pairs, 2]): Symmetric joint pairs

    Returns:
        output_flipped (ndarray([batch_size, num_joints, height, width])): After flipping back
    """
    assert output_flipped.ndim == 4, 'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    # swap
    for pair in joint_pairs:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


#########################################
# get location after rotating
def get_direction(src_point, rotation_rad):
    SiN, CoS = np.sin(rotation_rad), np.cos(rotation_rad)

    return [
        src_point[0] * CoS - src_point[1] * SiN,
        src_point[0] * SiN + src_point[1] * CoS
    ]


# get location 
def get_3rd_point(a, b):
    return np.array([
        b[0] + b[1] - a[1],
        b[1] + a[0] - b[0]
    ], dtype=np.float32)

