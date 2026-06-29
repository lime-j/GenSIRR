import torch
import torch.nn as nn
from xreflection.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class KontextLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(KontextLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        return self.loss_weight * torch.mean(torch.abs(pred - target))