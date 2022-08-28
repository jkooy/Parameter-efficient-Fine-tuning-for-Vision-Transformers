import torch as th
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



class SoftTargetCrossEntropy(nn.Module):
    def __init__(self, normslized=False):
        super(SoftTargetCrossEntropy, self).__init__()
        self.normalized = normslized

    def forward(self, x, target):
        if self.normalized:
           loss = th.sum(-target * F.log_softmax(x, dim=-1), dim=-1) / th.sum(target, dim=-1)
        else:
           loss = th.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

class HybridContrastive(nn.Module):
    '''
    Hybrid Contrastive loss
    '''
    def __init__(self, loss_type="i2t+t2i"):
        super(HybridContrastive, self).__init__()
        self.loss_func = SoftTargetCrossEntropy(normslized=True)
        self.loss_type = loss_type

    def forward(self, x, targets=None):
        assert targets is not None, "targets must not be None for hybrid contrastive learning"
        labels = targets
        
        if self.loss_type == "i2t+t2i":
            if isinstance(x, tuple) and isinstance(labels, tuple):
                loss_i = self.loss_func(x[0], labels[0])
                loss_t = self.loss_func(x[1], labels[1].t())
            else:
                loss_i = self.loss_func(x, labels)
                loss_t = self.loss_func(x.t(), labels.t())
            return (loss_i + loss_t) / 2.0
        elif self.loss_type == "i2t":
            loss_i = self.loss_func(x, labels)
            return loss_i
        elif self.loss_type == "t2i":
            loss_t = self.loss_func(x.t(), labels.t())
            return loss_t