from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    
    Equation: y = (1 - epsilon) * y + epsilon / K.
    
    Args:
    - num_classes (int): number of classes
    - epsilon (float): weight
    - use_gpu (bool): whether to use gpu devices
    - label_smooth (bool): whether to apply label smoothing, if False, epsilon = 0
    - prior: Class*1 vector, prior class distribution
    """
    def __init__(self, num_classes, w, epsilon=0.1, use_gpu=True, label_smooth=True, gamma = 0.5):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.sigmoid = nn.Sigmoid()
        self.gamma = gamma
        self.w = w

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """
        if self.use_gpu:
            targets = targets.cuda()
            w = torch.tensor(self.w).float().cuda()
        else:
            w = torch.tensor(self.w).float().cpu()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        # targets = targets*torch.log(sig_output+1e-5) + (1-targets)*torch.log(1-sig_output+1e-5)
        # loss = (- targets).mean(0).sum()

        # neg_abs = - inputs.abs()
        # loss = (inputs.clamp(min=0) - inputs * targets + (1 + neg_abs.exp()).log()).mean(0).sum()

        # max_val = (-inputs).clamp(min=0)
        # loss = inputs - inputs * targets + max_val + ((-max_val).exp() + (-inputs - max_val).exp()).log()

        inputs_pos = inputs.clamp(min=0)
        inputs_neg = inputs.clamp(max=0)

        # sigmoid(inputs)**self.gamma
        sigmoid_pos = (self.gamma*inputs_neg-self.gamma*(1+(-inputs.abs()).exp()).log()).exp()

        # 1-sigmoid(inputs)**self.gamma
        sigmoid_neg = (-self.gamma*inputs_pos-self.gamma*(1+(-inputs.abs()).exp()).log()).exp()

        first_pos = -sigmoid_pos*inputs_pos*(1-targets)
        first_neg = sigmoid_neg*inputs_neg*targets
        # print('pos: ',inputs_pos.norm())
        # print('neg: ',inputs_neg.norm())

        loss = -(first_pos + first_neg - sigmoid_neg*(1+(-inputs.abs()).exp()
                            ).log()*targets - sigmoid_pos*(1+(-inputs.abs()).exp()).log()*(1-targets))

        return (w*loss.mean(0)).sum()