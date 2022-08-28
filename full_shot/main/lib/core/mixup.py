import numpy as np
import torch


def mixup_data(x, y, mixup):
    """Returns mixed inputs, pairs of targets, and lambda"""
    lam = np.random.beta(mixup, mixup) if mixup > 0 else 1
    indices = torch.randperm(x.shape[0]).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[indices]
    y_a, y_b = y, y[indices]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
