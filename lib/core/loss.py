from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ntpath import join

import torch
from torch import nn

from utils.print_functions import print_inter_debug_info


class JointsMSELoss(nn.Module):
    """ MSE loss """
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
    
    def forward(self, output, target, target_weight):
        """
            output: torch.Tensor(batch_size, num_joints, H_hmpsz, W_hmpsz)
            target: torch.Tensor(batch_size, num_joints, H_hmpsz, W_hmpsz)
            target_weight: torch.Tensor(batch_size, num_joints, 1)
        """
        print_inter_debug_info('Loss Output Size', output.size(), 'loss')
        
        batch_size = output.size(0)
        num_joints = output.size(1)
        print(batch_size, num_joints)
        # heatmap_pred: tuple with length = num_joints
        # heatmap_pred[0].shape: [batch_size, 1, H_hmpsz * W_hmpsz]
        heatmap_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmap_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        
        loss = 0.
        for joint_index in range(num_joints):
            # [batch_size, H * W]
            _heatmap_pred = heatmap_pred[joint_index].squeeze()
            _heatmap_gt = heatmap_gt[joint_index].squeeze()
            if self.use_target_weight:
                loss += .5 * self.criterion(
                    _heatmap_pred.mul(target_weight[:, joint_index]),
                    _heatmap_gt.mul(target_weight[:, joint_index])
                )
            else:
                loss += .5 * self.criterion(_heatmap_pred, _heatmap_gt)
        
        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    """ Object Heatmap_pred topK Mean MSE loss """
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='None')
        self.use_traget_weight = use_target_weight
        self.topk = topk
    
    def ohkm(self, loss):
        """ loss: torch.Tensor.Size([batch_size, num_joints, H_hmpsz * W_hmpsz]) """
        ohkm_loss = 0.
        for batch in range(loss.size(0)):
            batch_loss = loss[batch]
            # choose topk prediction of all joints of each pixel
            # top_index: torch.Tensor([topk, H_hmpsz * W_hmpsz]), indices of topk pred
            _, top_index = torch.topk(
                batch_loss, k=self.topk, dim=0, sorted=False
            )
            # topk value(no matter which of the joints)
            # batch_loss_topk: torch.Tensor([topk, H_hmpsz * W_hmpsz]), values of topk pred
            batch_loss_topk = torch.gather(batch_loss, 0, top_index)
            ohkm_loss += torch.sum( batch_loss_topk ) / self.topk
        
        return ohkm_loss / loss.size(0)

    def forward(self, output, target, target_weight):
        """
            output: torch.Tensor([batch_size, num_joints, H_hmpsz, W_hmpsz])
            target: torch.Tensor([batch_size, num_joints, H_hmpsz, W_hmpsz])
            target_weight: torch.Tensor([batch_size, num_joints, 1])
        """
        batch_size = output.size(0)
        num_joints = output.size(1)
        # heatmap_pred: tuple with length = num_joints
        # heatmap_pred[0].shape: [batch_size, 1, H_hmpsz * W_hmpsz]
        heatmap_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmap_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for joint_index in range(num_joints):
            heatmap_pred = heatmap_pred[joint_index].squeeze()
            heatmap_gt = heatmap_gt[joint_index].squeeze()
            if self.use_traget_weight:
                loss.append(.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, num_joints]),
                    heatmap_gt.mul(target_weight[:, num_joints])
                ))
            else:
                loss.append(.5 * self.criterion(heatmap_pred, heatmap_gt))
        
        # get mean of the loss of each num_joints
        # loss: [num_joints, torch.Tensor.Size([batch_size, 1, H_hmpsz * W_hmpsz]) ]
        loss = [inter_loss.mean(dim=1).unsqueeze(dim=1) for inter_loss in loss]
        # loss: torch.Tensor.Size([batch_size, num_joints, H_hmpsz * W_hmpsz])
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)