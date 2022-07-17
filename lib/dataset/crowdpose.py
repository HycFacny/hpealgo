from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import collections
import numpy as np
import json_tricks as json
from pathlib import Path

import crowdposetools as cdpt
import crowdposetools.coco as coco
from crowdposetools.cocoeval import COCOeval

from dataset.dataset_base import JointsDataset
from utils.nms.nms import oks_nms
from utils.nms.nms import soft_oks_nms
from utils.print_functions import print_inter_debug_info


logger = logging.getLogger(__name__)


class CrowdPoseDataset(JointsDataset):
    """
        "keypoints": {
            0: "left_shoulder",
            1: "right_shoulder",
            2: "left_elbow",
            3: "right_elbow",
            4: "left_wrist",
            5: "right_wrist",
            6: "left_hip",
            7: "right_hip",
            8: "left_knee",
            9: "right_knee",
            10: "left_ankle",
            11: "right_ankle",
            12: "head",
            13: "neck"
        }
        
        "skeleton": [
            [12, 13], [13, 0], [13, 1], [0, 2], [2, 4], [1, 3], [3, 5], [13, 7],
            [13, 6], [7, 9], [9, 11], [6, 8], [8, 10]
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
        
        self.cdpe = coco.COCO(self._get_keypoint_annotation_file())
        
        self._cdpe_classes = [cat['name'] \
            for cat in self.cdpe.loadCats(self.cdpe.getCatIds())]
        self.classes = ['__background__'] + self._cdpe_classes
        self.num_classes = len(self.classes)
        
        logger.info(f'=> classes: {self.classes}, loading successfully')
        
        # bind index between cdpe.index, self.class_index, and cats name
        # self.classes -> self.num_classes
        self._class_to_index = dict(zip(self.classes, range(self.num_classes)))
        # cdpe.classes -> cdpe.index
        self._class_to_cdpe_index = dict(zip(self._cdpe_classes, self.cdpe.getCatIds()))
        # cdpe.index   -> self.class_index
        self._cdpe_index_to_class_index = dict([
            (self._class_to_cdpe_index[cls], self._class_to_index[cls])
                for cls in self._cdpe_classes
        ])
        
        self.image_set_index = self._load_image_set_index()
        # print(self.image_set_index)
        self.num_images = len(self.image_set_index)
        logger.info(f'=> images num: {self.num_images}')
        
        self.num_joints = 14
        self.flip_pairs = [ [i, i + 1] for i in range(0, 11, 2) ]
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 12, 13)
        self.lower_body_ids = (6, 7, 8, 9, 10, 11)
        self.parent_ids = None

        self.joints_weight = np.array([
                1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 
                1.2, 1.2, 1.5, 1.5,
                1., 1.
            ], dtype=np.float32
        ).reshape( (self.num_joints, 1) )

        self.db = self._get_db()
        # print(self.db)
        
        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)
        
        logger.info(f'=> load {len(self.db)} samples')
    
    
    #####################################################################
    ''' inner function used for loading and pre-process infos '''
    #####################################################################
    def _get_keypoint_annotation_file(self):
        # 'crowdpose/annotations/*.json'
        annotation_file = 'crowdpose_' + self.image_set + '.json'
        dataset_root = Path(self.root)
        
        return str(dataset_root / 'annotations' / annotation_file)
    
    def _load_image_set_index(self):
        return self.cdpe.getImgIds()
    
    def _get_db(self):
        return self._load_cdpe_keypoint_annotations()
    
    def _image_path_from_index(self, index):
        '''
            e.g. dataset_root / images / index.jpg
            return:
                image_path: str
        '''
        file_name = '%06d.jpg' % index
        image_path = Path(self.root) / 'images' / file_name
        
        return str(image_path)
    
    def _load_cdpe_keypoint_annotations(self):
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_cdpe_keypoint_annotations_kernel(index))
        return gt_db
    
    def _load_cdpe_keypoint_annotations_kernel(self, index):
        """
            coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
            iscrowd:
                crowd instances are handled by marking their overlaps with all categories to -1
                and later excluded in training
            bbox:
                [x1, y1, w, h]
            :param index: coco image id
            :return: db entry
            
            "images": [{
                "file_name": "103492.jpg",
                "id": 103492,
                "height": 429,
                "width": 640,
                "crowdIndex": 0.08
            },
        """
        image_annotation = self.cdpe.loadImgs(index)[0]
        width, height = image_annotation['width'], image_annotation['height']
        
        #index of bbox ( all bbox from image[inex] )
        annotation_ids = self.cdpe.getAnnIds(imgIds=index, iscrowd=False)
        bbox_objs = self.cdpe.loadAnns(annotation_ids)

        # adjust bbox information to normal ones
        valid_bboxes = []
        for bbox in bbox_objs:
            x, y, w, h = bbox['bbox']
            # left, top
            x1, y1 = max(0, x), max(0, y)
            # right, bottom
            x2, y2 = max(width - 1, x1 + max(0, w - 1)), min(height - 1, y1 + max(0, h - 1))
            bbox['area'] = w * h
            if not (bbox['area'] > 0 and x2 >= x1 and y2 >= y1): continue
            
            bbox['clean_bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
            valid_bboxes.append(bbox)
        bbox_objs = valid_bboxes
        
        # gather all boxes and form into dict in our own format
        rec = []
        for bbox in bbox_objs:
            # useless annotation for other mission
            if self._cdpe_index_to_class_index[bbox['category_id']] != 1: continue
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
    def _load_cdpe_person_detection_results(self):
        """
            return filtered dictionary
        """
        
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)
        if not all_boxes:
            logger.error(f'=> load {self.bbox_file} failed !')
            return None

        logger.info(f'=> Total boxes: {len(all_boxes)}')
        
        keypoint_db = []
        num_boxes = 0
        
        for n_image in range(len(all_boxes)):
            det_res = all_boxes[n_image]
            if det_res['category_id'] != 1: continue
            
            # whethre image name like '%06.jpg' or complete path
            image_name = str(self._image_path_from_index(det_res['image_id']))[ -10: ]
            
            bbox =det_res['bbox']
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

    def _write_cdpe_keypoint_results(self, keypoints, res_file):
        pack = [{
            'cat_id': self._class_to_cdpe_index[cls],
            'cls_id': cls_index,
            'cls': cls,
            'anno_type': 'keypoints',
            'keypoints': keypoints
        } for cls_index, cls in enumerate(self.classes) if cls != '__background__']
        
        results = self._load_cdpe_keypoint_results_percat_kernel(pack[0])
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
    def _load_cdpe_keypoint_results_percat_kernel(self, pack):
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

    # use cdpeeval tools to evaluate
    def _do_python_keypoint_eval(self, res_file):
        cdpe_dt = self.cdpe.loadRes(res_file)
        cdpe_eval = COCOeval(self.cdpe, cdpe_dt, 'keypoints')
        cdpe_eval.params.useSegm = None
        cdpe_eval.evaluate()
        cdpe_eval.accumulate()
        cdpe_eval.summarize()
        
        stats_names = ['AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
        info_string = []
        
        for idx, name in enumerate(stats_names):
            info_string.append( (name, cdpe_eval.stats[idx]) )
        
        return info_string


    #####################################################################
    ''' evaluate func for validating and testing '''
    #####################################################################
    ### waiting for change
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
                e.g. ['/home/huangyuchao/projects/datasets/crowdpose/images/000000397133.jpg', ...]
            
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
        
        print('preds shape: {}'.format(preds.shape))
        print('image shape: {}'.format(len(image_path)))
        
        for idx, kpt in enumerate(preds):
            kpt_all.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0 : 2],
                'scale': all_boxes[idx][2 : 4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(image_path[idx][ -10 : -4 ])
            })
        
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
            keep = []
            
            '''
            # calculate oks nms
            if self.soft_nms:
                # keep = soft_oks_nms([image_keypoints[i] for i in range(len(image_keypoints))], self.oks_thre)
                keep = soft_oks_nms(image_keypoints, self.oks_thre)
            else:
                # keep = oks_nms([image_keypoints[i] for i in range(len(image_keypoints))], self.oks_thre)
                keep = oks_nms(image_keypoints, self.oks_thre)
            '''
            
            # keep all
            if len(keep) == 0:
                oks_nms_keypoints.append(image_keypoints)
            # keep part we expected
            else:
                oks_nms_keypoints.append([image_keypoints[i] for i in keep])
        
        self._write_cdpe_keypoint_results(oks_nms_keypoints, res_file)

        # training phase
        if 'test' not in self.image_set:
            info_string = self._do_python_keypoint_eval(res_file)
            value = collections.OrderedDict(info_string)
            return value, value['AP']

        # validating or testing phase
        else:
            return {'NULL': 0}, 0

