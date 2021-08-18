from pathlib import Path
from pprint import pprint
import timm049
import albumentations as A
from albumentations.augmentations.transforms import MotionBlur
from albumentations.pytorch import ToTensor, ToTensorV2
import cv2
from kuma_utils.nn.training import EarlyStopping, NoEarlyStoppingNEpochs
from kuma_utils.metrics import AUC, MultiAUC, MultiAP, Accuracy, AUC_Anno, SeUnderSp, F1
from kuma_utils.custom_loss import *
from kuma_utils.loss import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from noisy_loss import *
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import copy

import os
import torch.nn as nn
import pandas as pd
import numpy as np
from pdb import set_trace as st

from models.backbones import *
from models.group_norm import convert_groupnorm
from models.batch_renorm import convert_batchrenorm
from models.multi_instance import MultiInstanceModel, MetaMIL, AttentionMILModel, MultiInstanceModelWithWataruAttention
from models.resnet import resnet18, resnet34, resnet101, resnet152
from models.model_4channels import get_attention, get_resnet34, get_attention_inceptionv3
from models.vae import VAE, ResNet_VAE
from models.model_with_arcface import EnetBertArcface

from augmentations.strong_aug import *
from augmentations.augmentation import *
from augmentations.policy_transform import policy_transform

import json
settings_json = json.load(open('SETTINGS.json', 'r'))

def print_config(cfg):
    info = ''
    items = [
        # general
        'batch_size', 'lr', 'seed', 'image_size',
        'criterion', 'metric', 'log_metrics', 'event', 'grad_accumulations', 'fp16',
    ]
    print(f'\n----- Config -----')
    for key in items:
        try:
            value = eval(f'cfg.{key}')
            print(f'{key}: {value}')
            info += f'{key}: {value}, '
        except:
            print('{key}: ERROR')
    print(f'----- Config -----\n')
    return info

class Baseline:
    def __init__(self):
        self.batch_size = 64
        self.lr = 0.0001
        self.CV = 5
        self.epochs = 300
        self.resume = False
        self.seed = 2023
        self.tta = 1
        self.predict_valid = False
        self.predict_test = False

        self.train_df = pd.read_csv(settings_json['TRAIN_FOR_CLASSIFICATION_PATH'])
        self.train_df['path'] = 'input/images/train/' + self.train_df['image_id'].values + '.jpg'
        self.train_df['mask_path'] = 'input/mask_train/' + self.train_df['image_id'].values + '.jpg'

        self.detection_pred_df = pd.read_csv(settings_json['YOLO_OOF_PATH']) # created by detection models & wbf
        self.test_df = copy.deepcopy(self.train_df) # Test inference will be done by kaggle kernel, so set a dummy

        self.model_name = 'swin_large_patch4_window12_384_in22k'
        self.num_classes = 5

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.metric = MultiAUC().torch

        self.image_size = 512
        self.log_metrics = []
        self.event = NoEarlyStoppingNEpochs(0)
        self.stopper = EarlyStopping(patience=5, maximize=True)

        self.transform = base_aug_v2(self.image_size)
        self.ian_mixup_aug = None
        self.mixup_aug = None
        self.cutmix_aug = None
        self.fmix_aug = None
        self.resizemix_aug = None
        self.grad_accumulations = 1
        self.inference_only = False
        self.fp16 = True
        self.fine_tune_epochs = 1
        self.image_level = False
        self.crop_pad = 100
        self.ch4 = False
        self.optimizer = 'adam'
        self.scheduler = 'CosineAnnealingWarmRestarts'
        self.train_by_all_data = False
        self.mse = False
        self.val_origin = False
        self.train_skip_eval = False
        self.crop_by_mask = True
        self.major_feat_pseudo_th = None
        self.minor_feat_pseudo_th = None
        self.box_df = None
        self.pad_to_square = False
        self.test_split_fold = True
        self.upsample = False
        self.freeze = False
        self.use_box_mask = False
        self.aux_criterion = None
        self.pretrained_path = None
        self.mixup_params = None
        self.mask_x_resize = 48
        self.mixup_aug_with_mask = None
        self.box_mask_v1 = True
        self.label_to_0 = False
        self.skip_attn = False
        self.unfreeze = False

class model0changelr(Baseline):
    def __init__(self):
        super().__init__()
        self.label_features = ['Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance', 'Negative for Pneumonia', 'none']
        self.metric = MultiAP(label_features=self.label_features, use_label_features=self.label_features).torch
        self.batch_size = 8
        self.fp16 = False
        self.grad_accumulations = 4
        self.image_size = 384
        self.transform = with_mask_aug_v2()
        self.crop_by_mask = True
        self.pad_to_square = False
        self.aux_criterion = F.binary_cross_entropy_with_logits
        self.cls_criterion = self.criterion
        self.criterion = HybridClsSegLoss(cls_loss=self.cls_criterion, seg_loss=self.aux_criterion, seg_weight=1.0)
        self.tta = 2
        self.model = timm049.create_model('swin_large_patch4_window12_384_in22k', pretrained=True, num_classes=14)
        self.pretrained_path = settings_json['CHEXPERT_PRETRAINED_PATH']
        self.num_classes = 5
        self.box_mask_v1 = False
        self.box_mask_v2 = False
        self.box_mask_v3 = True
        self.box_mask_v4 = False
        self.conf_th = 0.3
        self.ellipse = False

class swinmixupchangelr(Baseline):
    def __init__(self):
        super().__init__()
        self.label_features = ['Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance', 'Negative for Pneumonia', 'none']
        self.metric = MultiAP(label_features=self.label_features, use_label_features=self.label_features).torch
        self.batch_size = 8
        self.fp16 = False
        self.grad_accumulations = 4
        self.image_size = 384
        self.transform = with_mask_aug_v2()
        self.crop_by_mask = True
        self.pad_to_square = False

        self.aux_criterion = F.binary_cross_entropy_with_logits
        self.cls_criterion = copy.deepcopy(self.criterion)
        self.mixup_aug_with_mask = apply_mixaug_seg
        self.mixup_params = {'mixup': 0.4}
        self.criterion = MixHybridClsSegLoss(cls_loss=F.binary_cross_entropy_with_logits, seg_loss=self.aux_criterion, seg_weight=1.0)

        self.tta = 2
        self.model = timm049.create_model('swin_large_patch4_window12_384_in22k', pretrained=True, num_classes=14)
        self.pretrained_path = settings_json['CHEXPERT_PRETRAINED_PATH']
        self.num_classes = 5
        self.box_mask_v1 = False
        self.box_mask_v2 = False
        self.box_mask_v3 = True
        self.box_mask_v4 = False
        self.conf_th = 0.3
        self.ellipse = False

class chexpert(Baseline):
    def __init__(self):
        super().__init__()
        self.train_df = pd.read_csv('input/chexpert.csv')
        self.model = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=True, num_classes=3)
        self.label_features = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        for label in self.label_features:
            self.train_df = self.train_df[self.train_df[label]>=0]
        self.metric = MultiAUC(label_features=self.label_features, use_label_features=self.label_features).torch
        self.batch_size = 16
        self.fp16 = False
        self.grad_accumulations = 4
        self.image_size = 384
        self.transform = image_clasification_medical_v2_resize_first(self.image_size)
        self.crop_by_mask = True
        self.pad_to_square = True
        self.aux_criterion = None
        self.tta = 1
