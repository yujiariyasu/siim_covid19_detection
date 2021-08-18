import copy
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from . import builder
from . import optim
from . import metrics as pl_metrics
from .data.mixaug import apply_mixaug
from .task_utils import *


class BaseTask(pl.LightningModule): 

    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model 

        self.val_loss = []
        
        self.save_hyperparameters(cfg)
        
    def set(self, name, attr):
        if name == 'metrics':
            attr = nn.ModuleList(attr) 
        setattr(self, name, attr)
    
    def check_attributes(self):
        for obj in ['optimizer','scheduler','loss_fn','metrics','valid_metric']:
            assert hasattr(self, obj)

    def on_train_start(self): 
        self.check_attributes()

        self.total_training_steps = self.trainer.num_training_batches * self.trainer.max_epochs 
        self.current_training_step = 0
        
        if isinstance(self.scheduler, optim.CosineAnnealingLR):
            self.scheduler.T_max = self.total_training_steps 

        if isinstance(self.scheduler, optim.CustomOneCycleLR):
            self.scheduler.total_steps = self.total_training_steps
            self.scheduler.step_size_up = float(self.scheduler.pct_start * self.total_training_steps) - 1
            self.scheduler.step_size_down = float(self.total_training_steps - self.scheduler.step_size_up) - 1

    def training_step(self, batch, batch_idx):             
        X, y = batch
        if hasattr(self, 'mixaug') and isinstance(self.mixaug, dict):
            X, y = apply_mixaug(X, y, self.mixaug)
        p = self.model(X) 
        loss = self.loss_fn(p, y)
        self.log('loss', loss) 
        self.current_training_step += 1
        return loss

    def validation_step(self, batch, batch_idx): 
        X, y = batch
        p = self.model(X) 
        loss = self.loss_fn(p, y)
        self.val_loss += [loss]
        for m in self.metrics: m.update(p, y)
        return loss
        
    def validation_epoch_end(self, *args, **kwargs):
        metrics = {}
        for m in self.metrics:
            metrics.update(m.compute())
        metrics['val_loss'] = torch.stack(self.val_loss).mean() ; self.val_loss = []
        max_strlen = max([len(k) for k in metrics.keys()])

        if isinstance(self.valid_metric, list):
            metrics['vm'] = torch.sum(torch.stack([metrics[_vm.lower()].cpu() for _vm in self.valid_metric]))
        else:
            metrics['vm'] = metrics[self.valid_metric.lower()]

        self.log_dict(metrics)
        for m in self.metrics: m.reset()

        if self.global_rank == 0:
            print('\n========')
            for k,v in metrics.items(): 
                print(f'{k.ljust(max_strlen)} | {v.item():.4f}')

    def configure_optimizers(self):
        lr_scheduler = {
            'scheduler': self.scheduler,
            'interval': 'step' if self.scheduler.update_frequency == 'on_batch' else 'epoch'
        }
        if isinstance(self.scheduler, optim.ReduceLROnPlateau): 
            lr_scheduler['monitor'] = self.valid_metric
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': lr_scheduler
            }

    def train_dataloader(self):
        return builder.build_dataloader(self.cfg, self.train_dataset, 'train')

    def val_dataloader(self):
        return builder.build_dataloader(self.cfg, self.valid_dataset, 'valid')


class DetectionTask(BaseTask):

    def __init__(self, *args, **kwargs):
        self.eval_iou_thr = kwargs.pop('eval_iou_thr', 0.5)
        self.mixup = kwargs.pop('mixup', False)
        super().__init__(*args, **kwargs)

    def check_attributes(self):
        for obj in ['optimizer','scheduler','metrics','valid_metric']:
            assert hasattr(self, obj)

    @staticmethod
    def random_derangement(n):
        while True:
            v = [i for i in range(n)]
            for j in range(n - 1, -1, -1):
                p = random.randint(0, j)
                if v[p] == j:
                    break
                else:
                    v[j], v[p] = v[p], v[j]
            else:
                if v[0] != 0:
                    return list(v)

    def apply_mixup(self, X, y, y_cls=None): 
        ori_indices = list(range(X.size(0)))
        mix_indices = self.random_derangement(len(ori_indices))
        lam = np.clip(np.random.beta(1.0,1.0, X.size(0)), 0.35, 0.65)
        lam = torch.from_numpy(lam).float().to(X.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        X = lam * X + (1 - lam) * X[mix_indices]
        y_mix = copy.deepcopy(y)
        if not isinstance(y_cls, type(None)):
            y_cls = y_cls.float()
            lam = lam.squeeze(-1).squeeze(-1)
            y_cls = lam * y_cls + (1 - lam) * y_cls[mix_indices]
        for oi, mi in zip(ori_indices, mix_indices):
            filler = y['bbox'][oi][-1].unsqueeze(0)
            y1 = y['bbox'][oi]
            y1 = y1[y1[:,0] != -1]
            y2 = y['bbox'][mi]
            y2 = y2[y2[:,0] != -1]
            new_y = torch.cat([y1, y2])
            diff = 100 - new_y.size(0)
            y_mix['bbox'][oi] = torch.cat([new_y] + [filler] * diff)
            y1 = y['cls'][oi]
            y1 = y1[y1 != -1]
            y2 = y['cls'][mi]
            y2 = y2[y2 != -1]
            new_y = torch.cat([y1, y2])
            y_mix['cls'][oi] = torch.cat([new_y] + [torch.tensor([-1]).to(X.device)] * diff)
        if not isinstance(y_cls, type(None)):
            return X, y_mix, y_cls
        return X, y_mix

    def training_step(self, batch, batch_idx):             
        X, y = batch
        if self.mixup:
            X, y = self.apply_mixup(X, y)
        output = self.model(X, y) 
        loss = output['loss']
        self.log('loss', loss) 
        self.current_training_step += 1
        return loss

    def get_tpfp(self, dets, gt_bboxes, gt_labels):
        nc = self.model.num_classes
        tpfp_list, num_gts = [], []
        for each_cls in range(1, nc+1): 
            classes = dets[...,-1]
            cls_dets = dets[classes == each_cls]
            cls_gt = gt_labels[gt_labels == each_cls]
            cls_bboxes = gt_bboxes[gt_labels == each_cls]
            tp, fp = tpfp_default(cls_dets[...,:-1], cls_bboxes, iou_thr=self.eval_iou_thr)
            scores = dets[...,-2]
            results = torch.cat([tp, fp, scores.unsqueeze(0)])
            results = results.transpose(1,0)
            tpfp_list.append(results)
            num_gts.append(len(cls_gt))
        return tpfp_list, num_gts

    def validation_step(self, batch, batch_idx): 
        X, y = batch
        output = self.model(X, y) 
        loss = output['loss']
        self.val_loss += [loss]
        dets = output['detections']
        dets = dets[:,:,[1,0,3,2,4,5]]
        results = [self.get_tpfp(dets[i], y['bbox'][i], y['cls'][i]) for i in range(X.size(0))]
        # Reformat results so that it is a list of length=num_classes
        # Then turn each element of this list into a torch.Tensor
        results_reformat, num_gts = [], []
        for each_cls in range(self.model.num_classes):
            class_list = [r[0][each_cls] for r in results]
            cls_num_gts = sum([r[1][each_cls] for r in results])
            results_reformat.append(torch.cat(class_list))
            num_gts.append(cls_num_gts)
        for m in self.metrics: m.update(results_reformat, num_gts)
        return loss


class HybridTask(DetectionTask):

    def __init__(self, *args, **kwargs):
        self.eval_iou_thr = kwargs.pop('eval_iou_thr', 0.5)
        self.cls_loss_weight = kwargs.pop('cls_loss_weight', 1.0)
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):             
        X, y_det, y_cls = batch
        if self.mixup:
            X, y_det, y_cls = self.apply_mixup(X, y_det, y_cls)
        output = self.model(X, y_det) 
        loss = output['loss']
        cls_loss = F.binary_cross_entropy_with_logits(output['img_cls'].float(), y_cls.float())
        loss = loss + self.cls_loss_weight * cls_loss
        loss = loss / (self.cls_loss_weight + 1.0)
        self.log('loss', loss) 
        self.current_training_step += 1
        return loss

    def validation_step(self, batch, batch_idx): 
        X, y_det, y_cls = batch
        output = self.model(X, y_det) 
        loss = output['loss']
        cls_loss = F.binary_cross_entropy_with_logits(output['img_cls'].float(), y_cls.float())
        loss = loss + self.cls_loss_weight * cls_loss
        self.val_loss += [loss]
        for m in self.metrics: m.update(output['img_cls'], y_cls)
        return loss


