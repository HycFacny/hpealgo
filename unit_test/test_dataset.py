from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math
import logging
import basic_info
from pathlib import Path
import cv2

import numpy as np
import json_tricks as json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import utils.transforms as tfm

import test_config
import basic_info
import dataset
from config import cfg
from config import update_config
# import models


train_dataset = None
valid_dataset = None


def test_cocoapi():
    dataset_root = Path('/home/huangyuchao/projects/datasets/coco2017/')
    # print(dataset_root.exists())
    anno_file = dataset_root / 'annotations' / 'person_keypoints_val2017.json'
    coco = COCO(anno_file)
    
    CATS = coco.loadCats(coco.getCatIds())
    # print(CATS, type(CATS))
    # print(coco.getCatIds(), type(coco.getCatIds()))
    cats = [cat['name'] for cat in CATS]
    # print(cats, type(cats))

    image_index = coco.getImgIds()          # e.g. 397133
    image = coco.loadImgs(image_index[0])
    '''
        image basic info annotation, e.g.
        [{
            'license': 4,
            'file_name': '000000397133.jpg',
            'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg',
            'height': 427,
            'width': 640,
            'date_captured': '2013-11-14 17:02:52',
            'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg',
            'id': 397133
        }]
    '''
    # print(image_index[0], '\n', image)

    bbox_ids = coco.getAnnIds(imgIds=image_index, iscrowd=False)
    items = coco.loadAnns(bbox_ids)
    # print(items[0])
    '''
        bbox info annotation, e.g.
        {
            'segmentation': [ useless],
            'num_keypoints': 13,
            'area': 17376.91885,        # area size
            'iscrowd': 0,
            # keypoint: [x, y, v], v = 2 visible, 1 occluded, 0 none
            'keypoints': [433, 94, 2, 434, 90, 2, 0, 0, 0, 443, 98, 2, 0, 0, 0, 420, 128, 2, 474, 133, 2, 396, 162, 2, 489, 173, 2, 0, 0, 0, 0, 0, 0, 419, 214, 2, 458, 215, 2, 411, 274, 2, 458, 273, 2, 402, 333, 2, 465, 334, 2],
            'image_id': 397133,
            'bbox': [388.66, 69.92, 109.41, 277.62],        # x, y, w, h, first orientation, second horizon
            'category_id': 1,   # belong to person category
            'id': 200887    # annotation index ( unique for every bbox & keypoint annotation )
        }
    '''
    bbox_ids = coco.getAnnIds(imgIds=image_index[0], iscrowd=False)
    bboxes = coco.loadAnns(bbox_ids)
    print(bboxes)
    

def test_crowdposeapi():
    dataset_root = Path('/home/huangyuchao/projects/datasets/crowdpose/')
    # print(dataset_root.exists())
    anno_file = dataset_root / 'annotations' / 'crowdpose_test.json'
    coco = COCO(anno_file)
    
    CATS = coco.loadCats(coco.getCatIds())
    print(CATS, type(CATS))
    print(coco.getCatIds(), type(coco.getCatIds()))
    cats = [cat['name'] for cat in CATS]
    print(cats, type(cats))

    image_index = coco.getImgIds()          # e.g. 397133
    image = coco.loadImgs(image_index[0])
    '''
        image basic info annotation, e.g.
        [{
            'supercategory': 'person',
            'id': 1,
            'name': 'person',
            'keypoints': [
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                'head', 'neck'
            ],
            'skeleton': [
                [12, 13], [13, 0], [13, 1], [0, 2], [2, 4], [1, 3], [3, 5], [13, 7],
                [13, 6], [7, 9], [9, 11], [6, 8], [8, 10]
            ]
        }] 
        
        num_images: 8000
        num_person_instances: 34770
        num_joints: 14
    '''
    print(image_index[0], '\n', image)
    print('num_images: {}'.format(len(image_index)))
    bbox_ids = coco.getAnnIds(imgIds=image_index, iscrowd=False)
    bboxes = coco.loadAnns(bbox_ids)
    print('num_person_instances: {}'.format(len(bboxes)))
    '''
        bbox info annotation, e.g.
        {
            'num_keypoints': 5,
            'iscrowd': 0,
            'keypoints': [
                0, 0, 0, 208, 108, 2, 0, 0, 0, 278, 158, 2, 262, 206, 2, 348, 98, 2, 0, 0, 0, 173, 299, 2,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 256, 27, 1, 220, 89, 1
            ],
            'image_id': 106848,
            'bbox': [106.01, 13.43, 273.15, 352.42],
            'category_id': 1,
            'id': 123803
        }
    '''
    bbox_ids = coco.getAnnIds(imgIds=image_index[0], iscrowd=False)
    bboxes = coco.loadAnns(bbox_ids)
    print(bboxes)
    for i, bbox in enumerate(bboxes):
        bbox['area'] = bbox['bbox'][2] * bbox['bbox'][3]
    
    print(bboxes)
    

def test_dataset_base():
    global valid_dataset

    cfg = test_config.get_config_from_yaml()
    normalize = transforms.Normalize(
        mean=[.485, .456, .406], std=[.229, .224, .225]
    )

    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    )
    
    # valid_dataloader = torch.utils.data.DataLoader(
    #     valid_dataset,
    #     batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
    #     shuffle=False,
    #     num_workers=cfg.WORKERS,
    #     pin_memory=cfg.PIN_MEMORY
    # )

    # # print(cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS))

    # for i, (input, target, target_weight, meta) in enumerate(valid_dataloader):
    #     print(input)
    #     print(target)
    #     print(target_weight)
    #     print(meta)
    #     break

def test_transforms():
    p = valid_dataset[0]


if __name__ == '__main__':
    test_cocoapi()
    # test_dataset_base()
    # test_transforms()
    test_crowdposeapi()