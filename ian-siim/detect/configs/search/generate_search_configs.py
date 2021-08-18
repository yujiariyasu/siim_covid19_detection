import itertools
import numpy as np
import os, os.path as osp

from omegaconf import OmegaConf


BASECONFIG = 'base.yaml'

wd = list(10. ** np.arange(-5, 0, 1))
bs = [2, 4, 6, 8]
sz = [256, 384, 512, 640, 768]
nn = [f'tf_efficientdet_d{i}' for i in range(6)]
lr = [1.0e-5, 5.0e-5, 1.0e-4, 5.0e-4]

grid = list(itertools.product(wd,lr,bs,sz,nn))

cfg = OmegaConf.load(BASECONFIG)

scripts = []
for idx, g in enumerate(grid): 
    w, l, b, s, n = g
    cfg.optimizer.params.weight_decay = float(w) 
    cfg.optimizer.params.lr = float(l / 100.)
    cfg.scheduler.params.max_lr = float(l)
    cfg.train.batch_size = int(b)
    cfg.transform.resize.params.imsize = [s, s]
    cfg.model.params.image_size = [s, s]
    cfg.model.params.base = n
    cfg.model.params.backbone = n.replace('efficientdet_d', 'efficientnet_b')
    savedir = osp.dirname(BASECONFIG)
    filename = osp.join(savedir, f'search{idx:04d}.yaml')
    with open(filename, 'w') as f:
        OmegaConf.save(config=cfg, f=f.name)
