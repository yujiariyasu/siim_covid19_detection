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

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('p', default=[], dist_reduce_fx=None)
        self.add_state('t', default=[], dist_reduce_fx=None)

    def update(self, p, t):
        self.p.append(p)
        self.t.append(t)

    def compute(self):
        raise NotImplementedError


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