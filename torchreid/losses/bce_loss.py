from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class BCELoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    
    Equation: y = (1 - epsilon) * y + epsilon / K.
    
    Args:
    - num_classes (int): number of classes
    - epsilon (float): weight
    - use_gpu (bool): whether to use gpu devices
    - label_smooth (bool): whether to apply label smoothing, if False, epsilon = 0
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True):
        super(BCELoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (batch_size, num_classes)
        """
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        ###################################
        ##### pytorch bce with logits######
        ###################################

        max_val = (-inputs).clamp(min=0)
        loss = inputs - inputs * targets + max_val + ((-max_val).exp() + (-inputs - max_val).exp()).log()

        # loss = torch.max(inputs, torch.zeros_like(inputs)) - inputs * targets \
        #        + (torch.ones_like(inputs)+(-torch.abs(inputs)).exp()).log()

        return loss.mean(0).sum()