import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os

import json
settings_json = json.load(open('SETTINGS.json', 'r'))

class Baseline:
    def __init__(self):
        self.batch_size = 8
        self.lr = 1e-4
        self.epochs = 30
        self.seed = 2021

        # Dataset
        self.train_df_path = settings_json['TRAIN_FOR_DETECTION_PATH']
        self.train_size_df = pd.read_csv(settings_json['TRAIN_SIZE_DF_PATH'])
        self.train_df = pd.read_csv(self.train_df_path)
        self.image_size = 512
        self.num_classes = 3
        self.negative_class_id = 3
        self.fp16 = False
        self.predict_valid = True
        self.predict_test = True

        self.images_and_labels_dir = 'train'
        self.model_name= 'yolov5m'
        self.yaml = 'data/hyp.scratch.yaml'
        self.inference_only = False
        self.evolve = False

################
class mixup05_l6(Baseline):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.train_df.loc[:, 'class_name'] = 'opacity'
        self.train_df.loc[:, 'class_id'] = 0
        self.train_df = self.train_df[self.train_df.none==0]
        self.num_classes = 1
        self.yaml = settings_json['YOLO_YAML_PATH']
        self.model_name= settings_json['YOLO_RSNA_PRETRAINED_PATH']
        self.predict_test = False

class mixup05_l(Baseline):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.train_df.loc[:, 'class_name'] = 'opacity'
        self.train_df.loc[:, 'class_id'] = 0
        self.train_df = self.train_df[self.train_df.none==0]
        self.num_classes = 1
        self.yaml = settings_json['YOLO_YAML_PATH']
        self.model_name= settings_json['YOLO_RSNA_PRETRAINED_PATH']
        self.predict_test = False

class rsna_mixup05_l6(Baseline):
    def __init__(self):
        super().__init__()
        self.train_df_path = 'input/rsna18_train_bbox_annotations_effdet.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df.loc[:, 'class_name'] = 'opacity'
        self.train_df.loc[:, 'class_id'] = 0
        self.num_classes = 1
        self.model_name= 'yolov5l6'
        self.yaml = settings_json['YOLO_YAML_PATH']
        self.images_and_labels_dir = 'input/prev_rsna/train'
        self.predict_test = False
        self.predict_valid = False

class rsna_mixup05_l(Baseline):
    def __init__(self):
        super().__init__()
        self.train_df_path = 'input/rsna18_train_bbox_annotations_effdet.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df.loc[:, 'class_name'] = 'opacity'
        self.train_df.loc[:, 'class_id'] = 0
        self.num_classes = 1
        self.yaml = settings_json['YOLO_YAML_PATH']
        self.model_name= 'yolov5l'
        self.images_and_labels_dir = 'input/prev_rsna/train'
        self.predict_test = False
        self.predict_valid = False
