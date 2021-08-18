import numpy as np
import torch
import pytorch_lightning as pl

from pytorch_lightning.metrics import functional as FM
from sklearn.metrics import cohen_kappa_score, roc_auc_score, average_precision_score


def _roc_auc_score(t, p):
    return torch.tensor(roc_auc_score(t, p) if len(np.unique(t)) > 1 else 0.5)


def _average_precision_score(t, p):
    return torch.tensor(average_precision_score(t, p) if len(np.unique(t)) > 1 else 0)


class _BaseMetric(pl.metrics.Metric):

    def __init__(self, dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('p', default=[], dist_reduce_fx=None)
        self.add_state('t', default=[], dist_reduce_fx=None)

    def update(self, p, t):
        self.p.append(p)
        self.t.append(t)

    def compute(self):
        raise NotImplementedError


#From https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/mean_ap.py
def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).
    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]
    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


class mAP(pl.metrics.Metric):

    def __init__(self, num_classes, eps=1.0e-7, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.eps = eps
        for i in range(self.num_classes):
            self.add_state(f'class{i}', default=[], dist_reduce_fx=None)
            self.add_state(f'num_gts{i}', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, results, num_gts):
        # results and num_gts are both lists of length=num_classes
        # Within results is a torch.tensor of size (n_bboxes, 5)
        # Within num_gts is a torch.tensor scalar
        for i in range(len(results)):
            getattr(self, f'class{i}').append(results[i])
            cls_num_gts = getattr(self, f'num_gts{i}')
            setattr(self, f'num_gts{i}', cls_num_gts+num_gts[i])

    def compute(self):
        map_list = []
        for i in range(self.num_classes):
            cls_results = torch.cat(getattr(self, f'class{i}')).cpu().numpy()
            sort_inds = np.argsort(cls_results[:,-1])[::-1]
            tp = np.cumsum(cls_results[sort_inds,0])
            fp = np.cumsum(cls_results[sort_inds,1])
            recalls = tp / np.maximum(getattr(self, f'num_gts{i}').cpu().numpy(), self.eps)
            precisions = tp / np.maximum(tp+fp, self.eps)
            map_list += [average_precision(recalls, precisions)]
        return dict(map=torch.tensor(np.mean(map_list)))


class AUROC(_BaseMetric):
    """For simple binary classification
    """
    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().numpy() #(N,C)
        t = torch.cat(self.t, dim=0).cpu().numpy() #(N,C)
        auc_dict = {}
        for c in range(p.shape[1]):
            tmp_gt = t == c if t.ndim == 1 else t[:,c]
            auc_dict[f'auc{c}'] = _roc_auc_score(tmp_gt, p[:,c])
        auc_dict['auc_mean'] = np.mean([v for v in auc_dict.values()])
        return auc_dict


class AVP(_BaseMetric):
    """For simple binary classification
    """
    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().numpy() #(N,C)
        t = torch.cat(self.t, dim=0).cpu().numpy() #(N,C)
        avp_dict = {}
        for c in range(p.shape[1]):
            tmp_gt = t == c if t.ndim == 1 else t[:,c]
            avp_dict[f'avp{c}'] = _average_precision_score(tmp_gt, p[:,c])
        avp_dict['avp_mean'] = np.mean([v for v in avp_dict.values()])
        return avp_dict


class Accuracy(_BaseMetric):

    def compute(self): 
        p = torch.cat(self.p, dim=0)
        t = torch.cat(self.t, dim=0)
        return dict(accuracy=(p.argmax(1) == t).float().mean())


class QWK(_BaseMetric):

    def compute(self): 
        p = torch.cat(self.p, dim=0).argmax(1) 
        t = torch.cat(self.t, dim=0)
        qwk = cohen_kappa_score(t.cpu().numpy(), p.cpu().numpy(), weights='quadratic')
        return dict(qwk=torch.tensor(qwk))


