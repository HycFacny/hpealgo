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
    test_dataset_base()
    test_transforms()
