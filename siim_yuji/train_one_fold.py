import os
import sys
import time
import datetime
import argparse
from pathlib import Path
from tqdm import tqdm
import copy
from pprint import pprint
import random
from scipy.special import softmax
import json

import numpy as np
import pandas as pd

from kuma_utils.wandb_utils import set_wandb_params
from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error, average_precision_score, label_ranking_average_precision_score
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, StepLR, CosineAnnealingLR
try:
    from apex import amp
    USE_APEX = True
except:
    USE_APEX = False
from sklearn import preprocessing
try:
    from kuma_utils.nn.logger import Logger, DummyLogger
    logger = True
except:
    class DummyLogger:
        def __init__(self, log_dir):
            pass

        def scalar_summary(self, tag, value, step):
            pass

        def list_of_scalars_summary(self, tag_value_pairs, step):
            pass
    logger = False

from kuma_utils.nn.snapshot import *
from kuma_utils.metrics import *

from loader import get_loaders

from configs import *

from scheduler import MyScheduler
from pdb import set_trace as st
try:
    from mcs_kfold import MCSKFold
except:
    pass
import warnings
warnings.simplefilter('ignore')

import json
settings_json = json.load(open('SETTINGS.json', 'r'))

os.makedirs('results', exist_ok=True)
RESULTS_PATH_BASE = settings_json['RESULT_DIR']

os.environ['WANDB_MODE'] = 'offline'


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False

def load_model(model, pretrained_path, num_classes=1, skip_attn=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(pretrained_path, map_location=device)
    print('load pretrained model from', pretrained_path)

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    if 'SwinTransformer' in str(type(model)):
        model.head = nn.Linear(in_features=1536, out_features=num_classes, bias=True)
        print('change num_classes to', num_classes)
    elif 'Cait' in str(type(model)):
        model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        print('change num_classes to', num_classes)

if __name__ == "__main__":

    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, default='Test',
                        help="config name in configs.py")
    parser.add_argument("--n_cpu", type=int, default=40,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--limit_fold", type=int, default=-1,
                        help="train only one fold")
    parser.add_argument("--debug", action='store_true',
                        help="debug")
    parser.add_argument("--log", action='store_false',
                        help="write tensorboard log")
    parser.add_argument("--fp16", action='store_true',
                        help="train on fp16")
    parser.add_argument("--fold", type=int, default=0,
                        help="fold num")
    opt = parser.parse_args()

    pprint(opt)

    cfg = eval(opt.config)()
    if cfg.aux_criterion is None:
        from kuma_utils.nn.training import TorchTrainer
    else:
        from kuma_utils.nn.mask_training import TorchTrainer

    config_str = print_config(cfg)

    logger = DummyLogger('')

    seed_everything(cfg.seed)

    loader_map = get_loaders(cfg, opt)
    train_loader = loader_map['train']
    valid_loader = loader_map['valid']

    print('train / val:', len(train_loader), len(valid_loader))

    model = cfg.model
    if cfg.pretrained_path is not None:
        load_model(model, cfg.pretrained_path, num_classes=cfg.num_classes, skip_attn=cfg.skip_attn)

    snapshot_path = Path(f'{RESULTS_PATH_BASE}/{opt.config}/fold{opt.fold}.pt')
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.lr/1.7)
    if cfg.scheduler == 'StepLR':
        scheduler = StepLR(optimizer, step_size=10000000, last_epoch=-1)
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer, 'min', factor=0.5, patience=2, cooldown=1, verbose=True, min_lr=5e-7)
    elif cfg.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=100000, eta_min=5e-7)
    elif cfg.scheduler == 'fixCosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, 3, eta_min=5e-7)
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, 50, eta_min=0.00001)
    wandb = None

    NN_FIT_PARAMS = {
        'loader': train_loader,
        'fine_tune_loader': train_loader,
        'loader_valid': valid_loader,
        'loader_valid_tta': valid_loader,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'logger': logger,
        'snapshot_path': snapshot_path,
        'info_format': '[epoch] time data loss metric logmetrics earlystopping',
        'info_train': True,
        'info_interval': 1,
        'wandb': wandb,
        'cfg': cfg
    }

    trainer = TorchTrainer(
        model, serial=f'fold{opt.fold}', fp16=cfg.fp16)
    trainer.apex_opt_level = 'O1'

    trainer.fit(**NN_FIT_PARAMS)
