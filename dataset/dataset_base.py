from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random
from utils import zipreader


import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transmat
from utils.transforms import joint_affine_transform
from utils.transforms import fliplr_joints


logger = logging.getLogger(__name__)

debug_flag = False

def print_func(item_name, item):
    global debug_flag
    if not debug_flag:
        debug_flag = True
        with open('unit_test/test_output/test_dataset_temporal_output.txt', 'w') as f:
            f.write('....\n')
    
    with open('unit_test/test_output/test_dataset_temporal_output.txt', 'a') as f:
        f.write(item_name + ': >>>>\n'+ str(item) + '\n')


class JointsDataset(Dataset):

    def __init__(self, cfg, root, image_set, is_train, transform=None):
        '''
        root: path of the root of dataset
            e.g. 'home/huangyuchao/projects/datasets/coco2017'
        image_set: which part of the dataset
            e.g. 'train2017'
        is_train: whether in train phase or not
        '''
        
        # basic joints & images info 
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        # others
        self.is_train = is_train
        self.root = root
        self.image_set = image_set
        self.output_path = cfg.OUTPUT_DIR
        
        self.data_format = cfg.DATASET.DATA_FORMAT
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

        self.debug_flag1 = False

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        """
            load input from dataset, then do transform (affine, flip) ops if wanted.
        """
        db_rec = copy.deepcopy(self.db[idx])
        
        # if not self.debug_flag1:
        #     self._debug_transforms(idx)

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        # load image
        image_mat = None # H * W * C, not W * H * C
        if self.data_format == 'zip':
            image_mat = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            image_mat = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        if image_mat is None:
            logger.error(f'=> fail to read {image_file}')
            raise ValueError(f'fail to read {image_file}')
        
        image_mat = cv2.cvtColor(image_mat, cv2.COLOR_BGR2RGB) \
            if self.color_rgb else image_mat

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']
        center = db_rec['center']
        scale = db_rec['scale']
        # if after training, rec will have score key
        score = db_rec['score'] if 'score' in db_rec else 1
        rotation = 0

        if self.is_train:
            # enhance data by half body transforming, zoom, rotating and flipping
            if self.check_half_body_condition(joints):
                center_half, scale_half = self.half_body_transform( joints, joints_vis )
                if center_half is not None and scale_half is not None:
                    center, scale = center_half, scale_half
            
            scale_factor = self.scale_factor            # 0.35
            rotation_factor = self.rotation_factor      # 0.35

            # get a random scale within [1 - scale_factor, 1 + scale_factor]
            scale = scale * np.clip(np.random.randn() * scale_factor + 1, \
                                    1 - scale_factor, 1 + scale_factor)
            
            # get a random rotation with [-2 x rf, 2 x rf], if random seed <= 0.6
            rotation = np.clip(np.random.randn() * rotation_factor, \
                                      - rotation_factor * 2., rotation_factor * 2.) \
                if random.random() <= 0.6 else 0
            
            if self.flip and random.random() <= 0.5:
                image_mat = image_mat[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, image_mat.shape[1], self.flip_pairs
                )
                center[0] = image_mat.shape[1] - center[0] - 1
        
        joints_heatmap = joints.copy()
        
        # rotate
        trans = get_affine_transmat(center, scale, rotation, self.image_size)
        trans_heatmap = get_affine_transmat(center, scale, rotation, self.heatmap_size)

        # print_func('trans', trans)
        # print_func('trans_heatmap', trans_heatmap)

        # wrap affine use transform matrix
        input = cv2.warpAffine(
            image_mat,
            trans,
            ( int(self.image_size[0]), int(self.image_size[1]) ),
            flags=cv2.INTER_LINEAR
        )

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.:
                joints[i, 0:2] = joint_affine_transform(joints[i, 0:2], trans)
                joints_heatmap[i, 0:2] = joint_affine_transform(joints_heatmap[i, 0:2], trans_heatmap)
        
        # get heatmap, target: [numjoints, hmp_sz[1], hmp_sz[0]], target_weight: [num_joints, 1]
        # because maps we get form as [w, h], but we need to visualize, so we should change axis
        target, target_weight = self.generate_target(joints_heatmap, joints_vis)
        target, target_weight = torch.from_numpy(target), torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': center,
            'scale': scale,
            'rotation': rotation,
            'score': score
        }
        # print('**********************************************')

        return input, target, target_weight, meta

    def _get_db(self):
        raise NotImplementedError

    def _load_image_set_index(self):
        raise NotImplementedError

    #################################################

    def _debug_transforms(self, idx):
        db_rec = copy.deepcopy(self.db[idx])
        self._debug_flag1 = True
        print_func('db_rec', db_rec)

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']
        # if after training, rec will have score key
        score = db_rec['score'] if 'score' in db_rec else 1
        
        _center = db_rec['center']
        _scale = db_rec['scale']
        _scale_factor = self.scale_factor
        _rotation_factor = self.rotation_factor

        # test half body transform
        if self.check_half_body_condition(joints, rand=False):
            _center_half, _scale_half = self.half_body_transform( joints, joints_vis )
            if _center_half is not None and _scale_half is not None:
                _center, _scale = _center_half, _scale_half

        # test rotation
        alpha = 0.5
        _scale = _scale * np.clip(alpha * _scale_factor + 1, \
                                    1 - _scale_factor, 1 + _scale_factor)
        
        _rotation = np.clip(alpha * _rotation_factor, \
                                -_rotation_factor * 2., _rotation_factor * 2.)
        _trans_mat = get_affine_transmat(_center, _scale, _rotation, self.image_size)


    def half_body_transform(self, joints, joints_vis):
        upper_joints, lower_joints = [], []
        
        # part joints into lower and upper set
        for joint_idx in range(self.num_joints):
            if joints_vis[joint_idx][0] <= 0: continue
            if joint_idx in self.upper_body_ids:
                upper_joints.append(joints[joint_idx])
            else:
                lower_joints.append(joints[joint_idx])
        
        # equal probability randomly choose upper or lower part joints to precess on
        # if (np.random.randn() < 0.5 and len(upper_joints) > 2) \
        if (len(upper_joints) > 2) \
            or len(lower_joints) <= 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints
        if len(selected_joints) < 2: return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        
        # form bbox points and center point
        center = selected_joints.mean(axis=0)[:2]
        top_left = np.amin(selected_joints, axis=0)
        bottom_right = np.amax(selected_joints, axis=0)

        w = bottom_right[0] - top_left[0] + 1.0
        h = bottom_right[1] - top_left[1] + 1.0

        # zoom h, w into a accepted range
        if w > self.aspect_ratio * h:
            h = w / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        
        scale = np.array([
            w / self.pixel_std,
            h / self.pixel_std
        ], dtype=np.float32)
        
        scale = scale * 1.5

        return center, scale

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
            :param joints_vis: [num_joints, 3]
            :return param target:  [num_joints, heatmap_size[1], heatmap_size[0]]
            :return param target_weight:  [num_joints, 1] ( 0: invisible, 1: visible )
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]
        assert self.target_type == 'gaussian', 'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros(
                (self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32
            )
            temp_size = self.sigma * 3.
            
            for joint_idx in range(self.num_joints):
                target_weight[joint_idx] = \
                    self.adjust_target_weight(joints[joint_idx], target_weight[joint_idx], temp_size)
                if target_weight[joint_idx] == 0: continue
                
                # feat_stride = self.image_size / self.heatmap_size
                # mu_x = int(joint[0] / feat_stride[0] + 0.5)
                # mu_y = int(joint[1] / feat_stride[1] + 0.5)
                mu_x, mu_y = joints[joint_idx][0], joints[joint_idx][1]

                # np.array([0., 1., 2., ..., heatmap_size[0]])  0 ~ heatmap_size with stride = 1
                # 生成过程与hrnet的heatmap size不一样
                x = np.arange(0, self.heatmap_size[0], 1, dtype=np.float32)
                y = np.arange(0, self.heatmap_size[1], 1, dtype=np.float32)[:, np.newaxis]   # insert a new axis

                value = target_weight[joint_idx]
                # gaussian blur
                if value > 0.5:
                    target[joint_idx] = np.exp( -((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * (self.sigma ** 2)) )
        
        # different joint weight
        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        # img = np.transpose(target.copy(),[1,2,0])*255
        # img = img[:,:,0].astype(np.uint8)
        # img = np.expand_dims(img,axis=-1)
        # cv2.imwrite('./test.jpg', img)

        return target, target_weight
    

    # validate the correctness of half body transform
    def check_half_body_condition(self, joints, rand=True):
        zero_cnt, eps = 0, 1e-4
        for idx in range(joints.shape[0]):
            zero_cnt += 1 if joints[idx][0] < eps else 0
        if self.num_joints - zero_cnt <= self.num_joints_half_body \
            or (rand and np.random.rand() < self.prob_half_body):
            return False
        
        return True


    # Check that any part of the gaussian is in-bounds
    def adjust_target_weight(self, joint, target_weight, temp_size):
        # feat_stride = self.image_size / self.heatmap_size
        # mu_x = int(joint[0] / feat_stride[0] + 0.5)
        # mu_y = int(joint[1] / feat_stride[1] + 0.5)
        
        mu_x, mu_y = joint[0], joint[1]

        top_left = [int(mu_x - temp_size), int(mu_y - temp_size)]
        bottom_right = [int(mu_x + temp_size + 1), int(mu_y - temp_size + 1)]
        if top_left[0] >= self.heatmap_size[0] or top_left[1] >= self.heatmap_size[1] \
            or bottom_right[0] < 0 or bottom_right[1] < 0:
            target_weight = 0

        return target_weight


    # use rec with evaluation ratio > given formulation
    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis, joints_x, joints_y = 0, 0., 0.
            for joint, joint_vis in zip(rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0: continue

                num_vis += 1
                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0: continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis #joints center
            area = rec['scale'][0] * rec['scale'][1] * ( self.pixel_std ** 2 )
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
           
            diff_norm2 = np.linalg.norm((joints_center - bbox_center), 2)
            ks = np.exp(-1. * (diff_norm2 ** 2) / (2. * (0.2 ** 2) * (area ** 2)))  # 0.2: thresh
            matric = 0.2 / 16. * num_vis + 0.45 - 0.2 / 16  # ?
            
            if ks > matric:
                db_selected.append(rec)
            
        logger.info(f'=> num db: {len(db)}')
        logger.info(f'=> num selected db: {len(db_selected)}')

        return db_selected

    def evaluate(self, cfg, preds, output_dior, *args, **kwargs):
        raise NotImplementedError
