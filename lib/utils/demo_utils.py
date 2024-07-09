from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import cv2
import copy
import logging
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset

from core.evaluate import get_max_pred_locations
from core.evaluate import tylor
from utils.blur import gaussian_blur


logger = logging.getLogger(__name__)

suffix = {
    'image': ['jpg', 'png'],
    'video': ['avi', 'mp4', 'mov', 'mpeg']
}


class DemoIMAGEDataset(Dataset):
    def __init__(self, cfg, root, transform=None):
        self.root = root
        self.transform = transform
        self.color_rgb = cfg.DATASET.COLOR_RGB
        
        self.db = self._get_db()
        logger.info(f'=> Load demo images {len(self.db)} successfully.')
    
    def __len__(self):
        return len(self.db)
    
    def _get_db(self):
        file_list = []
        for suf in suffix['image']:
            for item in Path(self.root).rglob('*.{}'.format(suf)):
                file_list.append(str(item).split('/')[-1])
        
        if len(file_list) < 1:
            logger.error('=> For image test, we only support format jpg, png (suffix must be LOWER CASE), please check your input. ')
        print(file_list)
        return file_list

    def __getitem__(self, idx):
        image_name = self.db[idx]
        image_path = Path(self.root) / image_name
        
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if self.color_rgb else image

        meta = {
            'data_type': 'image',
            'image_name': image_name,
            'width': image.shape[1],
            'height': image.shape[0]
        }

        return image, meta


class DemoVIDEODataset(Dataset):    
    def __init__(self, cfg, root, transform=None):
        self.root = root
        self.transform = transform
        self.color_rgb = cfg.DATASET.COLOR_RGB
        
        self.num_videos, self.db = self._get_db()
        self.temporal_video = {
            'video_name': '',
            'video': []
        }
    
    def __len__(self):
        return len(self.db)
    
    def _get_db(self):
        file_list = []
        for suf in suffix['video']:
            for item in Path(self.root).rglob('*.{}'.format(suf)):
                file_list.append(str(item).split('/')[-1])
        
        if len(file_list) == 1:
            logger.error('=> For video test, we only support format jpg, png (suffix must be LOWER CASE), please check your input. ')
        
        print(file_list)
        rec = []
        segments = []
        for file in file_list:
            file = str(file)
            vid = cv2.VideoCapture(str(Path(self.root) / file))
            if not vid.isOpened():
                logger.error(f'=> Load video {file} error.')
                continue
            
            num_frame = int(vid.get(7))
            fps = int(vid.get(5))
            width, height = int(vid.get(3)), int(vid.get(4))
            segments.extend(video_split(vid, file, self.root))
            vid.release()
            print('file {}, frames {}, fps {}'.format(file, num_frame, fps))

            for frame_idx in range(num_frame):
                segment_idx = int((frame_idx + 1) / 1000)
                segment_name = '{}_seg_{}.{}'.format(file.split('.')[-2], segment_idx, file.split('.')[-1])
                inner_idx = frame_idx % 1000
                rec.append({
                    'data_type': 'video',
                    'video_name': segment_name,
                    'num_frame': 1000,
                    'frame_index': inner_idx,
                    'video_width': width,
                    'video_height': height,
                    'fps': fps
                })
        
        return len(segments), rec
    
    def __getitem__(self, idx):
        print('into get_item')
        meta = copy.deepcopy(self.db[idx])
        vid_name = meta['video_name']
        
        if vid_name != self.temporal_video['video_name']:
            self.temporal_video['video_name'] = vid_name
            self.temporal_video['video'] = []
            vid = cv2.VideoCapture(str(Path(self.root) / vid_name))

            if not vid.isOpened():
                logger.error(f'=> Load video {file} error.')
                return None, None
        
            num_frame = int(vid.get(7))
            
            print(vid_name)
            print(num_frame)
            for _ in range(num_frame):
                ret, _frame = vid.read()
                if not ret: break
                self.temporal_video['video'].append(_frame)
            print('before release')
            vid.release()
            print('after release')

        frame = self.temporal_video['video'][meta['frame_index']]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if self.color_rgb else frame
        return frame, meta


def demo_dataset(cfg, data_type, data_path):
    if data_type == 'image':
        return DemoIMAGEDataset(cfg, data_path)
    elif data_type == 'video':
        return DemoVIDEODataset(cfg, data_path)
    else:
        logger.error('=> Current demo only support type image or video, please check your input. ')
    return None


def demo_final_preds(cfg, heatmap):
    """Get predictions.

    Args:
        heatmap (ndarray(num_joints, hmp_H, hmp_W)): Predicted heatmap
    Returns:
        preds (ndarray(num_joints, 2)): Predicted joint locations
    """
    num_joints = heatmap.shape[1]
    preds, _ = get_max_pred_locations(heatmap)
    preds = preds[0]
    # calculate neighborbood derivation of locations to ajust them to relative confident values
    if cfg.TEST.POST_PROCESS:
        heatmaps = gaussian_blur(heatmap, cfg.TEST.BLUR_KERNEL)
        heatmaps = np.maximum(heatmaps, 1e-10)
        heatmaps = np.log(heatmaps)
        for joint_idx in range(num_joints):
            preds[joint_idx] = \
                tylor(heatmaps[0][joint_idx], preds[joint_idx])
    
    return preds


def video_split(vid, vid_name, video_path):
    num_frame = int(vid.get(7))
    fps = int(vid.get(5))
    video_width, video_height = int(vid.get(3)), int(vid.get(4))

    frames = []
    segments = []
    segment_idx = 0
    
    for frame_idx in range(num_frame):
        ret, frame = vid.read()
        if not ret: break
        frames.append(frame)
        if (frame_idx + 1) % 1000 == 0 or frame_idx == num_frame - 1:
            segment_file = Path(video_path) / ('{}_seg_{}.{}'.format(vid_name.split('.')[-2], segment_idx, vid_name.split('.')[-1]))
            segments.append(segment_file)
            
            fourcc_str = 'X264' if re.search(r'()*.mp4', vid_name) is not None else 'XVID'
            video_fcc = cv2.VideoWriter_fourcc(*fourcc_str)
            video_out = cv2.VideoWriter(segment_file, video_fcc, fps, (video_width, video_height))
            
            for frame_idx in range(len(frames)):
                video_out.write(frames[frame_idx])
            video_out.release()
            
            frames = []
            segment_idx += 1
    
    print('=> {} split successfully.')
    return segments


def draw(image, joints, ratio):
    """Draw keypoints in the original image.

    Args:
        image (ndarray(H, W, C)): Original image.
        joints (ndarray(num_joints, 2)): Locations of joints
        ratio (tuple(2)): Reconstructing scale of joints (w, h).

    Returns:
        image (ndarray(H, W, C)): Image after drawing
    """
    for joint in joints:
        joint_width = int( (joint[0] + 1) * ratio[0] - 1 )
        joint_height = int( (joint[1] + 1) * ratio[1] - 1 )
        cv2.circle(image, (joint_width, joint_height), 2, [255, 0, 0], 2)
        
    return image


def save_image(image, meta, demo_output):
    demo_output = str(Path(demo_output) / meta['image_name'])
    cv2.imwrite(demo_output, image)    


def save_video(video, meta, demo_output):
    """
    视频的编码格式参考如下：
    cv2.VideoWriter_fourcc('I','4','2','0'):YUV编码，4:2:0色度子采样。这种编码广泛兼容，但会产生大文件。文件扩展名应为.avi。
    cv2.VideoWriter_fourcc('P','I','M','1'):MPEG-1编码。文件扩展名应为.avi。
    cv2.VideoWriter_fourcc('X','V','I','D'):MPEG-4编码。如果要限制结果视频的大小，这是一个很好的选择。文件扩展名应为.avi。
    cv2.VideoWriter_fourcc('m', 'p', '4', 'v'):较旧的MPEG-4编码。如果要限制结果视频的大小，这是一个很好的选择。文件扩展名应为.m4v。
    cv2.VideoWriter_fourcc('X','2','6','4'):较新的MPEG-4编码。如果你想限制结果视频的大小，这可能是最好的选择。文件扩展名应为.mp4。
    cv2.VideoWriter_fourcc('T','H','E','O'):这个选项是Ogg Vorbis。文件扩展名应为.ogv。
    cv2.VideoWriter_fourcc('F','L','V','1'):此选项为Flash视频。文件扩展名应为.flv。
    
    :param video: 视频流矩阵
    :param meta: 视频基本信息
    :param demo_output: 视频输出目录
    :return:
    """
    demo_output = Path(demo_output) / 'AFTER_DEMO_{}'.format(meta['video_name'])
    
    num_frame = meta['num_frame']
    video_width = meta['video_width']
    video_height = meta['video_height']
    fps = meta['fps']
    
    fourcc_str = 'X264' if re.search(r'()*\.mp4', meta['video_name']) is not None else 'XVID'
    vid_fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    vid_out = cv2.VideoWriter(demo_output, vid_fourcc, fps, (video_width, video_height))
    
    for frame_idx in range(num_frame):
        vid_out.write(video[frame_idx])
    
    vid_out.release()
