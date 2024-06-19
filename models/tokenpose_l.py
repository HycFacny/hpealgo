from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time

from torch import nn

from models.tokenpose_base import TokenPose_L_Base
from models.hrnet_base import HRNetBase
from utils.print_functions import print_inter_debug_info

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class TokenPose_L(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(TokenPose_L, self).__init__()
        extra = cfg.MODEL.EXTRA

        self.pre_feature = HRNetBase(cfg, **kwargs)
        self.transformer = TokenPose_L_Base(
            feature_size=[cfg.MODEL.IMAGE_SIZE[0] // 4, cfg.MODEL.IMAGE_SIZE[1] // 4],
            patch_size=cfg.MODEL.PATCH_SIZE,
            num_keypoints=cfg.MODEL.NUM_JOINTS,
            dim=cfg.MODEL.DIM,
            channels=cfg.MODEL.BASE_CHANNEL,
            depth=cfg.MODEL.TRANSFORMER_DEPTH,
            heads=cfg.MODEL.TRANSFORMER_HEADS,
            mlp_dim=cfg.MODEL.DIM * cfg.MODEL.TRANSFORMER_MLP_RATIO,
            apply_init=cfg.MODEL.INIT,
            hidden_heatmap_dim=cfg.MODEL.HIDDEN_HEATMAP_DIM,
            heatmap_dim=cfg.MODEL.HEATMAP_SIZE[0] * cfg.MODEL.HEATMAP_SIZE[1],
            heatmap_size=cfg.MODEL.HEATMAP_SIZE,
            pos_embedding_type=cfg.MODEL.POS_EMBEDDING_TYPE
        )
    
    def forward(self, x):
        # print_inter_debug_info('whole_forward', '123', 'entire_network')
        # print('new mission!!\n')
        # begin = time.time()
        x = self.pre_feature(x)
        # print('hrnet forward time {}'.format(time.time() - begin))
        x = self.transformer(x)
        # print('whole network time {}'.format(time.time() - begin))
        return x
    
    def init_weights(self, pretrained='', cfg=None):
        self.pre_feature.init_weights(pretrained)


def get_pose_net(cfg, is_train, **kwargs):
    model = TokenPose_L(cfg, **kwargs)
    # print_inter_debug_info('whole_model', model, 'entire_network')
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, cfg)
    return model
