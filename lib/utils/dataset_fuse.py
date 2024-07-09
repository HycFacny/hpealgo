from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import json_tricks as json
from pathlib import Path

import numpy as np


from collections import defaultdict

import pycocotools.coco as coco
import crowdposetools as cdpt
import crowdposetools.coco as cdpe
from crowdposetools.cocoeval import COCOeval


from dataset.dataset_base import JointsDataset



def convert_coco_to_crowdpose(root_path):
    raise NotImplementedError


def convert_ochuman_to_crowdpose()    


class COCOWrapper(JointsDataset):
    
    def __init__(self, set_paths, ratios):
        self.set_paths = set_paths
        self.ratios = ratios / sum(ratios)
        
        assert len(self.sets) == len(self.ratios) \
            '=> Fused sets num must be same as ratios num'

        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

        for set_name, set_path in self.set_paths:
            print('loading fused dataset into memory ...')
        
        
            