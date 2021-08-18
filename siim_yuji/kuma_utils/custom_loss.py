import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss
from torch.autograd import Variable
from typing import Dict, List, Tuple
from pdb import set_trace as st

class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def forward(self, p, t):
        return F.binary_cross_entropy_with_logits(p.float(), t.float())


class CrossEntropyLoss(nn.CrossEntropyLoss):

    def forward(self, p, t):
        t = t.view(-1)
        if self.weight:
            return F.cross_entropy(p.float(), t.long(), weight=self.weight.float().to(t.device))
        else:
            return F.cross_entropy(p.float(), t.long())


class OneHotCrossEntropy(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


class SmoothCrossEntropy(nn.Module):
    
    # From https://www.kaggle.com/shonenkov/train-inference-gpu-baseline
    def __init__(self, smoothing = 0.05):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = F.one_hot(target.long(), x.size(1))
            target = target.float()
            logprobs = F.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return F.cross_entropy(x, target.long())


class MixBCE(nn.Module):

    def forward_train(self, p, t):
        lam = t['lam']
        loss1 = F.binary_cross_entropy_with_logits(p.float(), t['y1'].float(), reduction='none')
        loss2 = F.binary_cross_entropy_with_logits(p.float(), t['y2'].float(), reduction='none')
        loss = lam*loss1 + (1-lam)*loss2
        return loss.mean()

    def forward(self, p, t):
        if isinstance(t, dict) and 'lam' in t.keys():
            return self.forward_train(p, t)
        else:
            return F.binary_cross_entropy_with_logits(p.float(), t.float())


class MixCrossEntropy(nn.Module):

    def forward_train(self, p, t):
        lam = t['lam']
        loss1 = F.cross_entropy(p.float(), t['y1'].long(), reduction='none')
        loss2 = F.cross_entropy(p.float(), t['y2'].long(), reduction='none')
        loss = lam*loss1 + (1-lam)*loss2
        return loss.mean()

    def forward(self, p, t):
        if isinstance(t, dict) and 'lam' in t.keys():
            return self.forward_train(p, t)
        else:
            return F.cross_entropy(p.float(), t.long())


class DenseCrossEntropy(nn.Module):

    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcFaceLoss(nn.Module):

    def __init__(self, s=30.0, m=0.5):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        labels = F.one_hot(labels.long(), logits.size(1)).float().to(labels.device)
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss


class DiceLossV1(nn.Module):
    
    def __init__(self, 
                 epsilon: float = 1e-12, 
                 per_image: bool = True, 
                 ignore_empty: bool = False):
        super().__init__()
        self.epsilon = epsilon
        self.per_image = per_image
        self.ignore_empty = ignore_empty

    def forward(self, p, t):
        N,C,H,W = p.shape
        p = torch.sigmoid(p) 
        p = p.reshape(N*C, -1)
        t = t.reshape(N*C, -1)
        if self.ignore_empty:
            mask = t.sum(-1) 
            if (mask>0).sum().item() == 0:
                return 0.5
            p = p[mask > 0]
            t = t[mask > 0]
        if self.per_image:
            loss = 1 - (2 * (p*t).sum(dim=-1) + self.epsilon) / ((t ** 2).sum(dim=-1) + (p ** 2).sum(dim=-1) + self.epsilon)
            loss = loss.mean()
        else:
            loss = 1 - (2 * (p*t).sum() + self.epsilon) / ((t ** 2).sum() + (p ** 2).sum() + self.epsilon)
        return loss


class WeightedBCE(nn.Module):
    # From Heng
    def __init__(self, pos_frac: float = 0.25):
        super(WeightedBCE, self).__init__()
        assert 0 < pos_frac < 1, f'`pos_frac` must be between 0 and 1, {pos_frac} is invalid'
        self.pos_frac = pos_frac
        self.neg_frac = 1-pos_frac

    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        N,C,H,W = p.shape
        p = p.transpose(0,1).reshape(C, -1)
        t = t.transpose(0,1).reshape(C, -1)
        loss = F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction='none')
        pos = (t>0.5).float()
        neg = (t<0.5).float()
        pos_weight = pos.sum(1) + 1e-12
        neg_weight = neg.sum(1) + 1e-12
        loss = self.pos_frac*pos*loss/pos_weight.unsqueeze(1) + self.neg_frac*neg*loss/neg_weight.unsqueeze(1)
        return loss.sum()


class HybridClsSegLoss(nn.Module):

    def __init__(self, cls_loss=None, seg_loss=None, seg_weight=0.2, pos_frac=0.25, use_weighted_bce=True):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_loss = F.binary_cross_entropy_with_logits() if cls_loss is None else cls_loss
        self.seg_loss = F.binary_cross_entropy_with_logits() if seg_loss is None else seg_loss
        # self.seg_loss = WeightedBCE(pos_frac=pos_frac) if use_weighted_bce else BCEWithLogitsLoss()

    def forward(self, p, t):
        pcls, pseg = p
        tcls, tseg = t
        cls_loss = self.cls_loss(pcls.float(), tcls.float())

        # 以下どちらか
        seg_loss = self.seg_loss(pseg, tseg)
        # seg_loss = self.seg_loss(pseg.float(), tseg.float(), reduction='none')
        # if len(tcls[0]) == 5:
        #     for i, true in enumerate(tcls):
        #         if true[3] == 1:
        #             seg_loss[i] = 0
        # seg_loss = seg_loss.mean()

        return cls_loss + self.seg_weight * seg_loss


class MixHybridClsSegLoss(nn.Module):

    def __init__(self, cls_loss=None, seg_loss=None, seg_weight=0.2, use_weighted_bce=False):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_loss = F.binary_cross_entropy_with_logits() if cls_loss is None else cls_loss
        self.seg_loss = BCEWithLogitsLoss() if seg_loss is None else seg_loss
        assert not use_weighted_bce

    def forward_train(self, p, t):
        lam = t['lam']
        pcls, pseg = p
        cls_loss1 = self.cls_loss(pcls.float(), t['y1_cls'].float(), reduction='none')
        cls_loss2 = self.cls_loss(pcls.float(), t['y2_cls'].float(), reduction='none')
        cls_loss = lam*cls_loss1 + (1-lam)*cls_loss2
        # seg_loss1 = self.seg_loss(pseg.float(), t['y1_seg'].float(), reduction='none')
        # seg_loss2 = self.seg_loss(pseg.float(), t['y2_seg'].float(), reduction='none')
        seg_loss1 = self.seg_loss(pseg.float(), t['y1_seg'].float(), reduction='none')
        for i, true in enumerate(t['y1_cls']):
            if true[3] == 1:
                seg_loss1[i] = 0
        seg_loss2 = self.seg_loss(pseg.float(), t['y2_seg'].float(), reduction='none')
        for i, true in enumerate(t['y2_cls']):
            if true[3] == 1:
                seg_loss2[i] = 0

        seg_loss = lam*seg_loss1 + (1-lam)*seg_loss2
        loss = cls_loss.mean() + self.seg_weight * seg_loss.mean()
        return loss.mean()

    def forward(self, p, t):
        if isinstance(t, dict) and 'lam' in t.keys():
            return self.forward_train(p, t)
        else:
            pseg, pcls = p
            tcls, tseg = t
            cls_loss = F.binary_cross_entropy_with_logits(pcls.float(), tcls.float())
            seg_loss = F.binary_cross_entropy_with_logits(pseg.float(), tseg.float())
            return cls_loss + self.seg_weight * seg_loss


class DiceLossV2(nn.Module):
    # Based on https://github.com/pudae/kaggle-understanding-clouds/blob/master/kvt/losses/dice_loss.py
    def __init__(self, smooth: float = 1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(p) 
        N,C,H,W = p.shape
        pflat = p.reshape(N*C, -1) 
        tflat = t.reshape(N*C, -1)
        intersection = (pflat * tflat).sum(dim=1)

        loss = 1 - ((2 * intersection + self.smooth) / (pflat.sum(dim=1) + tflat.sum(dim=1) + self.smooth))
        return loss.mean()


class DiceBCELoss(nn.Module):

    def __init__(self, bce_weight: float = 0.2, dice: str = 'v1', **kwargs):
        super().__init__()
        self.bce_weight = bce_weight
        self.dsc_loss = DiceLossV1(**kwargs) if dice == 'v1' else DiceLossV2(**kwargs)
        self.bce_loss = BCEWithLogitsLoss()

    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.dsc_loss(p, t) + self.bce_weight*self.bce_loss(p, t)


class HybridTaskLoss(nn.Module):

    def __init__(self, seg_loss, cls_loss, weights=[1,1]):
        super().__init__()
        self.weights = weights
        name, params = builder.get_name_and_params(seg_loss)
        self.seg_loss = eval(name)(**params)
        name, params = builder.get_name_and_params(cls_loss)
        self.cls_loss = eval(name)(**params)

    def forward(self, 
        p: Tuple[torch.Tensor, torch.Tensor], 
        t: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        pseg, pcls = p
        tseg, tcls = t
        return self.weights[0]*self.seg_loss(pseg, tseg) + self.weights[1]**self.cls_loss(pcls, tcls)


class SymmetricLovaszLoss(nn.Module):

    def __init__(self, ignore=None, weight=None, size_average=True):
        super(SymmetricLovaszLoss, self).__init__()
        self.ignore = ignore

    def forward(self, logits, targets):
        return ((L.lovasz_hinge(logits.squeeze(1), targets.squeeze(1), per_image=True, ignore=self.ignore)) \
                + (L.lovasz_hinge(-logits.squeeze(1), 1-targets.squeeze(1), per_image=True, ignore=self.ignore))) / 2
