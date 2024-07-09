from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from pathlib import Path
import collections

import numpy as np
import json_tricks as json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .dataset_base import JointsDataset
from utils.nms.nms import oks_nms
from utils.nms.nms import soft_oks_nms
from utils.print_functions import print_inter_debug_info
from lib.dataset.dataset_base import JointsDataset

class OCHumanDataset(JointsDataset):
    """
        "keypoints": {
            0: 'nose',
            1: 'left_eye',
            2: 'right_eye',
            3: 'left_ear',
            4: 'right_ear',
            5: 'left_shoulder',
            6: 'right_shoulder',
            7: 'left_elbow',
            8: 'right_elbow',
            9: 'left_wrist',
            10: 'right_wrist',
            11: 'left_hip',
            12: 'right_hip',
            13: 'left_knee',
            14: 'right_knee',
            15: 'left_ankle',
            16: 'right_ankle'
        }
        
        "visible_map": {
            0: "missing",
            1: "vis",
            2: "self_occluded",
            3: "others_occluded"
        }
    
        "skeleton": [
            [16, 19], [13, 17], [4, 5], [19, 17], [17, 14], [5, 6], [17, 18], [14, 4], [1, 2], [18, 15], 
            [14, 1], [2, 3], [4, 10], [1, 7], [10, 7], [10, 11], [7, 8], [11, 12], [8, 9], [16, 4], [15, 1]
        ]
    """
    
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        
        self.image_width, self.image_height = cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width / self.image_height        # w / h ratio
        self.pixel_std = 200

        self.coco = COCO(self._get_keypoint_annotation_file())

        # load classes from annotation file
        self._coco_classes = [cat['name'] \
            for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + self._coco_classes
        self.num_classes = len(self.classes)
        logger.info(f'=> classes: {self.classes}, loading successfully')
        
        # print(f'loading classes {self.classes}')

        # bind index between coco.index, self.class_index, and cats name
        # self.classes -> self.num_classes
        self._class_to_index = dict(zip(self.classes, range(self.num_classes)))
        # coco.classes -> coco.index
        self._class_to_coco_index = dict(zip(self._coco_classes, self.coco.getCatIds()))
        # coco.index   -> self.class_index
        self._coco_index_to_class_index = dict([
            (self._class_to_coco_index[cls], self._class_to_index[cls])
                for cls in self._coco_classes
        ])

        # image file related params
        self.image_set_index = self._load_image_set_index() # == self.coco.getImgIds()
        self.num_images = len(self.image_set_index)
        logger.info(f'=> images num: {self.num_images}')
        # print(f'loading image {self.num_images}')

        # joints related params
        self.num_joints = 17
        self.flip_pairs = [ [2 * i + 1, 2 * (i + 1)] for i in range(8) ]    # [1, 2] ... [15, 16]
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)
        self.parent_ids = None

        self.joints_weight = np.array([
            1.,  1.,  1., 1., 1. , 1. , 1. , 1.2, 1.2,
            1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
        ], dtype=np.float32).reshape( (self.num_joints, 1) )    # super params, can change with our interest

        self.db = self._get_db()

        # load bbox from whether gt or detection
        if self.is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info(f'=> load {len(self.db)} samples.')
        print(f'loading database samples {len(self.db)}')        
        
    
    def _get_db(self):
        raise NotImplementedError
    
    def evaluate(self):
        raise NotImplementedError

