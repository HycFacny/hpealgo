from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


# pose_resnet related params
POSE_RESNET = CN()
POSE_RESNET.NUM_LAYERS = 50
POSE_RESNET.DECONV_WITH_BIAS = False
POSE_RESNET.NUM_DECONV_LAYERS = 3
POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
POSE_RESNET.NUM_DECONV_KERNALS = [4, 4, 4]
POSE_RESNET.FINAL_CONV_KERNEL = 1
POSE_RESNET.PRETRAINED_LAYERS = ['*']

# pose_multi_resolution_net related params
POSE_HIGH_RESOLUTION_NET = CN()
POSE_HIGH_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
POSE_HIGH_RESOLUTION_NET.STEM_INCHANNELS = 64
POSE_HIGH_RESOLUTION_NET.FINAL_CONV_KERNEL = 1

POSE_HIGH_RESOLUTION_NET.STAGE2 = CN()
POSE_HIGH_RESOLUTION_NET.STAGE2.NUM_MODULES = 1
POSE_HIGH_RESOLUTION_NET.STAGE2.NUM_BRANCHES = 2
POSE_HIGH_RESOLUTION_NET.STAGE2.NUM_BLOCKS = [4, 4]
POSE_HIGH_RESOLUTION_NET.STAGE2.NUM_CHANNELS = [32, 64]
POSE_HIGH_RESOLUTION_NET.STAGE2.BLOCK = 'BASIC'
POSE_HIGH_RESOLUTION_NET.STAGE2.FUSE_METHOD = 'SUM'

POSE_HIGH_RESOLUTION_NET.STAGE3 = CN()
POSE_HIGH_RESOLUTION_NET.STAGE3.NUM_MODULES = 1
POSE_HIGH_RESOLUTION_NET.STAGE3.NUM_BRANCHES = 3
POSE_HIGH_RESOLUTION_NET.STAGE3.NUM_BLOCKS = [4, 4, 4]
POSE_HIGH_RESOLUTION_NET.STAGE3.NUM_CHANNELS = [32, 64 ,128]
POSE_HIGH_RESOLUTION_NET.STAGE3.BLOCK = 'BASIC'
POSE_HIGH_RESOLUTION_NET.STAGE3.FUSE_METHOD = 'SUM'

POSE_HIGH_RESOLUTION_NET.STAGE4 = CN()
POSE_HIGH_RESOLUTION_NET.STAGE4.NUM_MODULES = 1
POSE_HIGH_RESOLUTION_NET.STAGE4.NUM_BRANCHES = 4
POSE_HIGH_RESOLUTION_NET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
POSE_HIGH_RESOLUTION_NET.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
POSE_HIGH_RESOLUTION_NET.STAGE4.BLOCK = 'BASIC'
POSE_HIGH_RESOLUTION_NET.STAGE4.FUSE_METHOD = 'SUM'

# pose_lite_hrnet related params
POSE_LITE_HRNET = CN()
POSE_LITE_HRNET.NUM_STAGES = 3
POSE_LITE_HRNET.STEM_INCHANNELS = 32
POSE_LITE_HRNET.STEM_OUTCHANNELS = 32
POSE_LITE_HRNET.EXPAND_RATIO = 1
POSE_LITE_HRNET.REDUCE_RATIOS = [8, 8, 8]
POSE_LITE_HRNET.WITH_HEAD = True
POSE_LITE_HRNET.NORM = 'bn'
POSE_LITE_HRNET.ZERO_INIT_RESIDUAL = False

POSE_LITE_HRNET.STAGE2 = CN()
POSE_LITE_HRNET.STAGE2.NUM_MODULES = 3
POSE_LITE_HRNET.STAGE2.NUM_BRANCHES = 2
POSE_LITE_HRNET.STAGE2.NUM_BLOCKS = [2, 2]
POSE_LITE_HRNET.STAGE2.NUM_CHANNELS = [40, 80]
POSE_LITE_HRNET.STAGE2.BLOCK = 'LITE'
POSE_LITE_HRNET.STAGE2.FUSE_METHOD = 'SUM'

POSE_LITE_HRNET.STAGE3 = CN()
POSE_LITE_HRNET.STAGE3.NUM_MODULES = 8
POSE_LITE_HRNET.STAGE3.NUM_BRANCHES = 3
POSE_LITE_HRNET.STAGE3.NUM_BLOCKS = [2, 2, 2]
POSE_LITE_HRNET.STAGE3.NUM_CHANNELS = [40, 80, 160]
POSE_LITE_HRNET.STAGE3.BLOCK = 'LITE'
POSE_LITE_HRNET.STAGE3.FUSE_METHOD = 'SUM'

POSE_LITE_HRNET.STAGE3 = CN()
POSE_LITE_HRNET.STAGE3.NUM_MODULES = 3
POSE_LITE_HRNET.STAGE3.NUM_BRANCHES = 4
POSE_LITE_HRNET.STAGE3.NUM_BLOCKS = [2, 2, 2, 2]
POSE_LITE_HRNET.STAGE3.NUM_CHANNELS = [40, 80, 160, 320]
POSE_LITE_HRNET.STAGE3.BLOCK = 'LITE'
POSE_LITE_HRNET.STAGE3.FUSE_METHOD = 'SUM'


MODEL_EXTRAS = {
    'pose_resnet': POSE_RESNET,
    'pose_high_resolution_net': POSE_HIGH_RESOLUTION_NET,
    'pose_lite_hrnet': POSE_LITE_HRNET,
}