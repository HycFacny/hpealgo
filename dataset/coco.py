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


logger = logging.getLogger(__name__)


class COCODataset(JointsDataset):
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
    ]
    '''

    def __init__(self, cfg, root, image_set, is_train, transform=None):
        '''
            root = ''
            image_set = 'train2017' or 'test2017'
        '''

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


    #####################################################################
    ''' inner function used for loading and pre-process infos '''
    #####################################################################
    def _get_keypoint_annotation_file(self):
        '''
            anno_root / annotations / person_keypoints_train2017.json for training, 
                or ... / image_info_test2017.json for validating and testing
        '''
        prefix = 'person_keypoints_' \
            if 'test' not in self.image_set else 'image_info_'
        dataset_root = Path(self.root)

        return str(dataset_root / 'annotations' / (prefix + self.image_set + '.json'))

    def _load_image_set_index(self):
        return self.coco.getImgIds()
    
    def _get_db(self):
        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()  # mainly for validation
    
        return gt_db

    def _image_path_from_index(self, index):
        '''
            e.g. dataset_root / train2017 / index.jpg
            return:
                image_path: str
        '''
        file_name = '%012d.jpg' % index
        if '2014' in self.image_set:
            file_name = 'COCO_%s_' % self.image_set + file_name
        
        # setting not to use valid set
        prefix = 'test2017' if 'test' in self.image_set else self.image_set
        prefix = prefix + '.zip' if self.data_format == 'zip' else prefix

        image_path = Path(self.root) / 'images' / prefix / file_name
        return str(image_path)
    
    # for each image_index, getting all bbox annotations with correct format we expected
    def _load_coco_keypoint_annotations(self):
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotations_kernel(index))
    
        return gt_db
    
    def _load_coco_keypoint_annotations_kernel(self, index):
        """
            coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
            iscrowd:
                crowd instances are handled by marking their overlaps with all categories to -1
                and later excluded in training
            bbox:
                [x1, y1, w, h]
            :param index: coco image id
            :return: db entry
        """
        image_annotaion = self.coco.loadImgs(index)[0]
        width, height = image_annotaion['width'], image_annotaion['height']

        # index of bbox ( all bbox from image[index] )
        annotation_ids = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        bbox_objs = self.coco.loadAnns(annotation_ids)

        # adjust bbox information to normal ones
        valid_bboxes = []
        for bbox in bbox_objs:
            x, y, w, h = bbox['bbox']
            # left, top
            x1, y1 = max(0, x), max(0, y)
            # right, bottom
            x2, y2 = min(width - 1, x1 + max(0, w - 1)), min(height - 1, y1 + max(0, h - 1))
            if not (bbox['area'] > 0. and x2 >= x1 and y2 >= y1):
                continue
            bbox['clean_bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
            valid_bboxes.append(bbox)
        bbox_objs = valid_bboxes

        # gather all boxes and form into dict in our own format
        rec = []
        for bbox in bbox_objs:
            # useless annotation for other mission
            if self._coco_index_to_class_index[bbox['category_id']] != 1: continue
            # no keypoint existing
            if bbox['num_keypoints'] <= 0: continue
            
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float32)

            for keypoint_index in range(self.num_joints):
                joints_3d[keypoint_index, 0] = bbox['keypoints'][keypoint_index * 3 + 0]
                joints_3d[keypoint_index, 1] = bbox['keypoints'][keypoint_index * 3 + 1]
                
                # here is a crucial point how we use occluded keypoint info to do something
                # default: use them same as normal keypoint  
                visible = min(bbox['keypoints'][keypoint_index * 3 + 2], 1)

                joints_3d_vis[keypoint_index, :] = [visible, visible, 0]
            
            # change box[4] to center[2] and scale[2] (overlapped by pixel_std * pixel_std)
            center, scale = self._bbox_to_center_and_scale(bbox['clean_bbox'][:4])

            rec.append({
                'image': self._image_path_from_index(index),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0
            })
        
        # global debug_flag
        # if not debug_flag:
        #     print(f'image_index: {index}')
        #     for i in range(len(rec)):
        #         print(rec[i], '\n')
        #     debug_flag = True

        return rec

    # change box to center and scale
    def _bbox_to_center_and_scale(self, box):
        x, y, w, h = box[:4]
        center = np.array([x + (w - 1) * 0.5, y + (h - 1) * 0.5], dtype=np.float32)
            
        # adjust w, h so that w / h = self.aspect_ratio (same as origin)
        # larger one freeze, increase the smaller one
        if w > self.aspect_ratio * h:
            h = w / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        # overlaped by pixel_std * pixel_std block, record number as scale
        scale = np.array([w / self.pixel_std, h / self.pixel_std], dtype=np.float32)
        # enlarge it to ensure the overlap contain whole box
        if center[0] != -1: scale = scale * 1.25

        return center, scale

    #####################################################################
    ''' inner function used for evaluating or testing phase '''
    #####################################################################

    # get all boxes detected with score higher than cfg.TEST.IMAGE_THRE
    def _load_coco_person_detection_results(self):
        '''
            return filtered dict
        '''

        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)
        if not all_boxes:
            logger.error(f'=> Load {self.bbox_file} failed !')
            return None
        
        logger.info(f'=> Total boxes: {len(all_boxes)}')

        keypoint_db = []
        num_boxes = 0
        for n_image in range(len(all_boxes)):
            det_res = all_boxes[n_image]
            if det_res['category_id'] != 1: continue

            # whether image name like '000000096713.jpg' or complete path
            image_name = str(self._image_path_from_index(det_res['image_id']))[ -16 : ]

            bbox = det_res['bbox']
            score = det_res['score']
            if score < self.image_thre: continue

            num_boxes += 1

            center, scale = self._bbox_to_center_and_scale(bbox)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            joints_3d_vis = np.ones((self.num_joints, 3), dtype=np.float32)

            keypoint_db.append({
                'image': image_name,
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis
            })
        
        logger.info(f'=> Total boxes after filter low score@{self.image_thre}: {num_boxes}')
        return keypoint_db
    
    # write detection results to file
    def _write_coco_keypoint_results(self, keypoints, res_file):
        pack = [ {
            'cat_id': self._class_to_coco_index[cls],
            'cls_ind': cls_index,
            'cls': cls,
            'anno_type': 'keypoints',
            'keypoints': keypoints
        } for cls_index, cls in enumerate(self.classes) if cls != '__background__']

        results = self._load_coco_keypoint_results_percat_kernel(pack[0])
        logger.info(f'=> writing results to {res_file}')

        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        
        # already existing result file
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                content.append(line for line in f) 

            content[-1] = ']'
            with open(res_file, 'w') as f:
                f.write(c for c in content)

    # rearrange pack format
    def _load_coco_keypoint_results_percat_kernel(self, pack):
        cat_id, image_keypoint_pack = pack['cat_id'], pack['keypoints']
        cat_result = []

        for image_res_pack in image_keypoint_pack:
            num_bboxes = len(image_res_pack)
            if num_bboxes <= 0: continue        # no bbox in this image

            # print(num_bboxes)
            # load all keypoints of this image to temp db
            _bbox_kpts = np.array([
                image_res_pack[bbox_idx]['keypoints'] for bbox_idx in range(num_bboxes)
            ])

            bbox_kpts = np.zeros( (_bbox_kpts.shape[0], self.num_joints * 3), dtype=np.float32)

            for kpt in range(self.num_joints):
                # zip keypoint, axis 3 is the keypoint score
                bbox_kpts[:, kpt * 3 + 0 : kpt * 3 + 3] = _bbox_kpts[:, kpt, 0 : 3]

            result = [{
                'image_id' : image_res_pack[bbox_idx]['image'],
                'category_id': cat_id,
                'keypoints': list(bbox_kpts[bbox_idx]),
                'score': image_res_pack[bbox_idx]['score'],
                'center': list(image_res_pack[bbox_idx]['center']),
                'scale': list(image_res_pack[bbox_idx]['scale'])
            } for bbox_idx in range(num_bboxes)]

            cat_result.extend(result)

        return cat_result

    # use cocoeval tools to evaluate
    def _do_python_keypoint_eval(self, res_file):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_string = []
        for idx, name in enumerate(stats_names):
            info_string.append( (name, coco_eval.stats[idx]) )
        
        return info_string
    
    #####################################################################
    ''' evaluate func for validating and testing '''
    #####################################################################
    def evaluate(self, cfg, preds, output_dir, all_boxes, image_path, *args, **kwargs):
        """
        Evaluate func for validating and testing
        
        Args:
            cfg ( cfg ): Network and dataset configs
            preds ( ndarray(batch_size, num_joints, 2) )
                Final preds with order (height, width)
            output_dir ( String ): Output directory
            all_boxes ( ndarray(batch_size * num_joints, 6) ):
                All boxes info, 0~1: center, 2~3: scale, 4: area, 5: score
            image_path ( List(batch_size) ): Image path
                e.g. ['/home/huangyuchao/projects/datasets/coco2017/images/val2017/000000397133.jpg', ...]
            
        Returns:
            name_values ():
            perf_indicator ():
        """
        # setting result files
        rank = cfg.RANK
        res_folder = Path(output_dir) / 'results'
        if not res_folder.exists():
            try:
                res_folder.mkdir()
            except Exception:
                logger.error(f'Fail to create {res_folder}')
        res_file = str(res_folder / 'keypoints_{}_results_{}.json'.format(self.image_set, rank))

        # {person box: keypoints} with image_name
        kpt_all = []
        
        for idx, kpt in enumerate(preds):
            kpt_all.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0 : 2],
                'scale': all_boxes[idx][2 : 4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(image_path[idx][ -16 : -4 ])
            })
        
        # for idx, item in enumerate(kpt_all):
        #     print_inter_debug_info('kpt_all[{}]'.format(idx), item, 'evaluate')
        
        
        # print(kpt_all)
        
        # {image_name: {person boxes: keypoints} }
        keypoints = collections.defaultdict(list)
        for kpt in kpt_all: keypoints[kpt['image']].append(kpt)
        
        # rescoring and getting oks nms
        oks_nms_keypoints = []
        for idx, image in enumerate(keypoints.keys()):
            image_keypoints = keypoints[image]
            
            # rescoring
            for person in image_keypoints:
                box_score = person['score']
                kpt_score = 0.
                valid_num = 0

                # select all valid keypoint from each person instance,
                for kpt_idx in range(self.num_joints):
                    kpt_instance_score = person['keypoints'][kpt_idx][2]    # 0, 1 location, 2 confidence
                    if kpt_instance_score > self.in_vis_thre:
                        kpt_score += kpt_instance_score
                        valid_num += 1
                    if valid_num > 0: kpt_score /= float(valid_num)
                
                # update person score with mean of theirs keypoints and own box score
                person['score'] = kpt_score * box_score
                        
            # calculate oks nms
            if self.soft_nms:
                # keep = soft_oks_nms([image_keypoints[i] for i in range(len(image_keypoints))], self.oks_thre)
                keep = soft_oks_nms(image_keypoints, self.oks_thre)
            else:
                # keep = oks_nms([image_keypoints[i] for i in range(len(image_keypoints))], self.oks_thre)
                keep = oks_nms(image_keypoints, self.oks_thre)
            
            # keep all
            if len(keep) == 0:
                oks_nms_keypoints.append(image_keypoints)
            # keep part we expected
            else:
                oks_nms_keypoints.append([image_keypoints[i] for i in keep])
        
        self._write_coco_keypoint_results(oks_nms_keypoints, res_file)

        # training phase
        if 'test' not in self.image_set:
            info_string = self._do_python_keypoint_eval(res_file)
            value = collections.OrderedDict(info_string)
            return value, value['AP']

        # validating or testing phase
        else:
            return {'NULL': 0}, 0

