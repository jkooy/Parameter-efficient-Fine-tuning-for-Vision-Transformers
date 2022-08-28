import numpy as np
import torch


def rand_bbox(w, h, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)

    # uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2


def rand_bbox_v2(w, h, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)

    # uniform
    cx = np.random.randint(w-cut_w)
    cy = np.random.randint(h-cut_h)

    bbx1 = np.clip(cx, 0, w)
    bby1 = np.clip(cy, 0, h)
    bbx2 = np.clip(cx + cut_w, 0, w)
    bby2 = np.clip(cy + cut_h, 0, h)

    return bbx1, bby1, bbx2, bby2


def mixcut_data(x, y, beta, rand_box_mode='v1'):
    """Returns mixed inputs, pairs of ys, and lambda"""
    indices = torch.randperm(x.shape[0]).to(x.device)

    y_a = y
    y_b = y[indices]
    h, w = x.shape[2], x.shape[3]
    lam = np.random.beta(beta, beta)
    rand_bbox_fn = rand_bbox
    if rand_box_mode == 'v2':
        rand_bbox_fn = rand_bbox_v2
    bbx1, bby1, bbx2, bby2 = rand_bbox_fn(w, h, lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.shape[-1] * x.shape[-2]))

    return x, y_a, y_b, lam


def mixcut_criterion(criterion, pred, y_a, y_b, l):
    return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)


def test(x, y, beta):
    import torchvision
    import cv2

    x, _, _, lam = mixcut_data(x, y, beta)
    grid = torchvision.utils.make_grid(x, 8, 2, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr[..., ::-1].copy()
    cv2.imwrite('tmp.jpg', ndarr)
    print(f'lam: {lam}')
