from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import collections
import numpy as np
import json_tricks as json
from pathlib import Path
from scipy.io import loadmat, savemat

from dataset.dataset_base import JointsDataset


logger = logging.getLogger(__name__)


class MPIIDataset(JointsDataset):
    '''
    "keypoints": {
        0: "right_ankle",
        1: "right_knee",
        2: "right_hip",
        3: "left_hip",
        4: "left_knee",
        5: "left_ankle",
        6: "pelvis",
        7: "thorax",
        8: "upper_neck",
        9: "head_top",
        10: "right_wrist",
        11: "right_elbow",
        12: "right_shoulder",
        13: "left_shoulder",
        14: "left_elbow",
        15: "left_wrist"
    }
    '''
    
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotate_factor = cfg.DATASET.ROT_FACTOR
        self.num_joints = 16
        
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)
        
        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info(f'=> load {len(self.db)} samples')
    
    def _get_db(self):
        # create train/val split
        file_name = Path(self.root) / 'annotations' / (self.image_set + '.json')

        with open(file_name) as annotation_file:
            annotations = json.load(annotation_file)

        gt_db = []
        for a in annotations:
            image_name = a['image']

            center = np.array(a['center'], dtype=np.float)
            scale = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if center[0] != -1:
                center[1] = center[1] + 15 * scale[1]
                scale = scale * (1 + self.scale_factor)
            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            center -= 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
            
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints), self.num_joints)
                
                joints[:, 0 : 2] -= 1
                joints_vis = np.array(a['joints_vis'])
                joints_3d[:, 0 : 2] = joints[:, 0 : 2]
                
                joints_3d_vis[:, 0] = joints_vis[ : ]
                joints_3d_vis[:, 1] = joints_vis[ : ]

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            
            gt_db.append({
                'image': str(Path(self.root) / image_dir / image_name),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })

        return gt_db
    
    def evaluate(self, cfg, preds, output_dir, all_boxes, image_path, *args, **kwarsg):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = Path(output_dir) / 'pred.mat'
            savemat(str(pred_file), mdict={'preds': preds})

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = Path(cfg.DATASET.ROOT) / 'annotations' / 'gt_{}.mat'.format(cfg.DATASET.TEST_SET)
        gt_dict = loadmat(str(gt_file))
        
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        
        scale = np.multiply( headsizes, np.ones( (len(uv_err), 1) ) )
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply( (scaled_uv_err <= threshold), jnt_visible)
        PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5 + 0.01, 0.01)
        pckAll = np.zeros( (len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold, jnt_visible)
            pckAll[r, :] = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6 : 8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6 : 8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = collections.OrderedDict(name_value)

        return name_value, name_value['Mean']
