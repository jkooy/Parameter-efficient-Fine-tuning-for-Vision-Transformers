from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


@torch.no_grad()
def multi_classes_accuracy(output, target, fast=False):
    num_labels = target.sum(dim=1)
    valid_indices = torch.nonzero(num_labels, as_tuple=False)

    maxk = num_labels.max().int().item()
    if fast:
        maxk = min(maxk, 10)

    maxk = max(1, maxk)
    topk, pred_topk = output.topk(maxk, dim=1, largest=True)

    n = valid_indices.size(0)
    pred = torch.zeros_like(output).cuda()

    if fast:
        pred = pred.scatter(1, pred_topk, 1)
    else:
        for i in range(n):
            sample_index = valid_indices[i].item()
            k = num_labels[sample_index].int().item()
            pred[sample_index, pred_topk[sample_index, :k]] = 1

    pred = pred * target
    correct = pred.sum(dim=1)
    accuracy = correct[valid_indices] * 100. / num_labels[valid_indices]

    accuracy = accuracy.sum(dim=0).item()

    if n > 0:
        accuracy /= n

    return accuracy
