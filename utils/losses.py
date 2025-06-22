import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_class_weights_from_yaml(cfg, class_names):
    weights = []
    for name in class_names:
        if name in cfg['loss']['class_weights']:
            weights.append(cfg['loss']['class_weights'][name])
        else:
            weights.append(cfg['loss']['class_weights'].get('other', 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def build_loss_fn(cfg, class_names):
    weights = get_class_weights_from_yaml(cfg, class_names)
    weights = weights.to(cfg['device'])

    if cfg['loss']['type'] == 'ce':
        return nn.CrossEntropyLoss()
    elif cfg['loss']['type'] == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=weights)
    elif cfg['loss']['type'] == 'dice':
        return DiceLoss()
    elif cfg['loss']['type'] == 'dice_ce':
        return DiceCELoss(weights)
    else:
        raise ValueError(f"Unknown loss type: {cfg['loss']['type']}")


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (inputs * targets_one_hot).sum(dims)
        union = inputs.sum(dims) + targets_one_hot.sum(dims)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(self, weights=None):
        super(DiceCELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        return self.ce(inputs, targets) + self.dice(inputs, targets)
