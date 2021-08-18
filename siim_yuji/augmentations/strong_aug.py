import torch
import numpy as np
from torchvision.transforms import Resize

from .aug_utils import rand_bbox, rand_region


__all__ = [
    "cutmix",
    "mixup",
    'apply_mixaug_seg',
    'apply_mixaug',
    "resizemix",
    "resizemix_with_mask",
]


import numpy as np
import torch
import torch.nn.functional as F


def apply_mixup(X, alpha=0.4):
    lam = np.random.beta(alpha, alpha, X.size(0))
    lam = np.max((lam, 1-lam), axis=0)
    index = torch.randperm(X.size(0))
    lam = torch.Tensor(lam).to(X.device)
    for dim in range(X.ndim - 1):
        lam = lam.unsqueeze(-1)
    X = lam * X + (1 - lam) * X[index]
    return X, index, lam


def rand_bbox(size, lam, margin=0):
    # lam is a vector
    B = size[0]
    assert B == lam.shape[0]
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_h = (H * cut_rat).astype(np.int)
    cut_w = (W * cut_rat).astype(np.int)
    # uniform
    if margin < 1 and margin > 0:
        h_margin = margin*H
        w_margin = margin*W
    else:
        h_margin = margin
        w_margin = margin
    cx = np.random.randint(0+h_margin, H-h_margin, B)
    cy = np.random.randint(0+w_margin, W-w_margin, B)
    #
    bbx1 = np.clip(cx - cut_h // 2, 0, H)
    bby1 = np.clip(cy - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_w // 2, 0, W)
    return bbx1, bby1, bbx2, bby2


def apply_cutmix(X, alpha=0.4, y=None):
    SEG = not isinstance(y, type(None))
    batch_size = X.size(0)
    lam = np.random.beta(alpha, alpha, batch_size)
    lam = np.max((lam, 1-lam), axis=0)
    x1, y1, x2, y2 = rand_bbox(X.size(), lam)
    index = torch.randperm(batch_size)
    for b in range(batch_size):
        X[b, ..., x1[b]:x2[b], y1[b]:y2[b]] = X[index[b], ..., x1[b]:x2[b], y1[b]:y2[b]]
        if SEG:
            y[b, ..., x1[b]:x2[b], y1[b]:y2[b]] = y[index[b], ..., x1[b]:x2[b], y1[b]:y2[b]]
    lam = 1. - ((x2 - x1) * (y2 - y1) / float((X.size(-1) * X.size(-2))))
    lam = torch.Tensor(lam).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    if SEG:
        return X, y, index, lam
    return X, index, lam


def rand_region(size, patch_size):
    H, W = size
    pH, pW = patch_size
    maxH = H - pH
    maxW = W - pW
    x1 = np.random.randint(0, maxH)
    y1 = np.random.randint(0, maxW)
    x2 = x1 + pH
    y2 = y1 + pW
    return x1, y1, x2, y2


def apply_resizemix(X, alphabeta, y=None):
    alpha, beta = alphabeta
    SEG = not isinstance(y, type(None))
    assert alpha > 0, 'alpha should be larger than 0'
    assert beta < 1, 'beta should be smaller than 1'
    batch_size = X.size(0)
    index = torch.randperm(batch_size)
    tau = np.random.uniform(alpha, beta, batch_size)
    lam = tau ** 2
    H, W = X.size()[2:]
    for b in range(batch_size):
        _tau = tau[b]
        patch_size = (int(H*_tau), int(W*_tau))
        resized_X = F.interpolate(X[index[b]].unsqueeze(0), size=patch_size, mode='bilinear', align_corners=False).squeeze(0)
        x1, y1, x2, y2 = rand_region((H, W), patch_size)
        X[b, ..., x1:x2, y1:y2] = resized_X
        if SEG:
            resized_y = F.interpolate(y[index[b]].unsqueeze(0), size=patch_size, mode='nearest').squeeze(0)
            y[b, ..., x1:x2, y1:y2] = resized_y
    lam = torch.Tensor(lam).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    if SEG:
        return X, y, index, lam
    return X, index, lam


MIX_FN = {'cutmix': apply_cutmix, 'mixup': apply_mixup, 'resizemix': apply_resizemix}
def apply_mixaug(X, y, mix):
    mixer = np.random.choice([*mix])
    X, index, lam = MIX_FN[mixer](X, mix[mixer])
    return X, {
        'y1':  y,
        'y2':  y[index],
        'lam': lam.to(y.device)
    }

def apply_mixaug_seg(X, yseg, ycls, mixup_params):
    mixer = np.random.choice([*mixup_params])
    if mixer in ['cutmix', 'resizemix']:
        X, yseg, index, lam = MIX_FN[mixer](X, mixup_params[mixer], y=yseg)
    else:
        X, index, lam = MIX_FN[mixer](X, mixup_params[mixer])

    return X, {
        'y1_cls':  ycls,
        'y2_cls':  ycls[index],
        'y1_seg':  yseg,
        # If using cutmix, segmentation ground truth has been edited
        # so just return the gt twice
        'y2_seg':  yseg if mixer in ['cutmix', 'resizemix'] else yseg[index],
        'lam': lam.to(X.device)
    }

def cutmix(x, y, alpha):
    assert alpha > 0, 'alpha should be larger than 0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(device)
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index,
                                      :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
               (x.size()[-1] * x.size()[-2]))
    return x, target_a, target_b, lam


def mixup(x, y, alpha):
    assert alpha > 0, 'alpha should be larger than 0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(device)
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam


def resizemix(x, y, alpha=0.1, beta=0.8):
    assert alpha > 0, 'alpha should be larger than 0'
    assert beta < 1, 'beta should be smaller than 1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rand_index = torch.randperm(x.size()[0]).to(device)
    tau = np.random.uniform(alpha, beta)
    lam = tau ** 2

    H, W = x.size()[2:]
    resize_transform = Resize((int(H*tau), int(W*tau)))
    resized_x = resize_transform(x[rand_index])

    target_a = y[rand_index]
    target_b = y
    x1, y1, x2, y2 = rand_region(x.size(), resized_x.size())
    x[:, :, y1:y2, x1:x2] = resized_x
    return x, target_a, target_b, lam

def resizemix_with_mask(x, mask, y, alpha=0.1, beta=0.8):
    assert alpha > 0, 'alpha should be larger than 0'
    assert beta < 1, 'beta should be smaller than 1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rand_index = torch.randperm(x.size()[0]).to(device)
    tau = np.random.uniform(alpha, beta)
    lam = tau ** 2

    H, W = x.size()[2:]
    resize_transform = Resize((int(H*tau), int(W*tau)))
    resized_x = resize_transform(x[rand_index])
    resized_mask = resize_transform(mask[rand_index])

    target_a = y[rand_index]
    target_b = y
    x1, y1, x2, y2 = rand_region(x.size(), resized_x.size())
    x[:, :, y1:y2, x1:x2] = resized_x
    mask[:, :, y1:y2, x1:x2] = resized_mask
    return x, mask, target_a, target_b, lam
