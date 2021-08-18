import os
import sys
import time
import datetime
import argparse
from pathlib import Path
from tqdm import tqdm
from copy import copy, deepcopy
from pprint import pprint
import random
from scipy.special import softmax
import json
from glob import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
import torch.utils.data as D
from torchvision import transforms as T
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
try:
    from apex import amp
    USE_APEX = True
except:
    USE_APEX = False

from kuma_utils.nn.logger import Logger, DummyLogger
from kuma_utils.nn.snapshot import *
from kuma_utils.metrics import *
# from kuma_utils.google_spread_sheet_editor import GoogleSpreadSheetEditor
from kuma_utils.nn.training import TorchTrainer

from yolo_configs import *
from pdb import set_trace as st
import warnings
warnings.simplefilter('ignore')

import json
settings_json = json.load(open('SETTINGS.json', 'r'))
input_dir = settings_json['INPUT_DIR']

def seed_everything(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def yolo2voc(height, width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]

    """
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int

    bboxes[..., [0, 2]] = bboxes[..., [0, 2]]* width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]]* height

    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]

    return bboxes


if __name__ == "__main__":
    start = time.time()
    seed_everything()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, default='Test',
                        help="config name in configs.py")
    parser.add_argument("--fold", type=int, default=0,
                        help="fold num")
    parser.add_argument("--use_row", type=int, default=2,
                        help="google spread sheet row")
    parser.add_argument("--make_labels", action='store_true',
                        help="make_labels")

    opt = parser.parse_args()
    pprint(opt)

    cfg = eval(opt.config)()
    fold = opt.fold
    os.makedirs('results', exist_ok=True)
    train_image_path = f'{input_dir}/images/{cfg.images_and_labels_dir}'

    train_df = cfg.train_df

    train_df['image_path'] = f'{train_image_path}/'+train_df.image_id+'.jpg'
    oof = train_df[train_df.fold == fold]

    train_val = train_df[train_df.class_id!=cfg.negative_class_id].reset_index(drop = True)
    train = train_val[train_val.fold != fold]
    val = train_val[train_val.fold == fold]

    class_ids, class_names = list(zip(*set(zip(train_val.class_id, train_val.class_name))))
    classes = list(np.array(class_names)[np.argsort(class_ids)])
    classes = list(map(lambda x: str(x), classes))

    train_files = list(train.image_path.unique())
    val_files = list(val.image_path.unique())
    oof_files = list(oof.image_path.unique())
    print('train/val/oof image nunique:', len(train_files), len(val_files), len(oof_files))

    if (opt.fold == 0) & (not cfg.inference_only):
        print('make labels start.')
        cfg.train_size_df = cfg.train_size_df.rename(columns={'dim0': 'height', 'dim1': 'width'})
        train_df = train_df.merge(cfg.train_size_df, on='image_id')

        train_df['scaled_x_min'] = train_df['x_min'] / train_df['width']
        train_df['scaled_x_max'] = train_df['x_max'] / train_df['width']
        train_df['scaled_y_min'] = train_df['y_min'] / train_df['height']
        train_df['scaled_y_max'] = train_df['y_max'] / train_df['height']

        train_df['x_center_scaled'] = (train_df.scaled_x_min + train_df.scaled_x_max) / 2
        train_df['y_center_scaled'] = (train_df.scaled_y_min + train_df.scaled_y_max) / 2
        train_df['width_scaled'] = (train_df.scaled_x_max-train_df.scaled_x_min)
        train_df['height_scaled'] = (train_df.scaled_y_max-train_df.scaled_y_min)
        train_df['yolo_text'] = train_df['class_id'].astype(str) + ' ' + train_df['x_center_scaled'].astype(str) + ' ' + train_df['y_center_scaled'].astype(str) + ' ' + train_df['width_scaled'].astype(str) + ' ' + train_df['height_scaled'].astype(str)
        if os.path.exists(f'{input_dir}/labels'):
            os.system(f'rm -r {input_dir}/labels')
        os.makedirs(f'{input_dir}/labels/{cfg.images_and_labels_dir}', exist_ok=True)
        for id_, df in train_df[train_df.class_id!=cfg.negative_class_id].groupby('image_id'):
            filename = f'{input_dir}/labels/{cfg.images_and_labels_dir}/{id_}.txt'
            f = open(filename, 'w')
            for l in df.yolo_text.values:
                f.writelines(l+"\n")
            f.close()
        print('make labels end.')

    os.chdir('yolov5')

    if not cfg.inference_only:

        os.makedirs(f'results/{opt.config}/fold{fold}', exist_ok=True)
        f = open(f'results/{opt.config}/fold{fold}/train.txt', 'w')
        for file in train_files:
            f.writelines(file+"\n")
        f.close()

        f = open(f'results/{opt.config}/fold{fold}/val.txt', 'w')
        for file in val_files:
            f.writelines(file+"\n")
        f.close()

        from os import listdir
        from os.path import isfile, join
        import yaml

        data = dict(
            train =  join(f'results/{opt.config}/fold{fold}/train.txt') ,
            val   =  join(f'results/{opt.config}/fold{fold}/val.txt'),
            nc    = cfg.num_classes,
            names = classes
            )

        with open(join(f'results/{opt.config}/fold{fold}/covid.yaml'), 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
        print('train start.')

        weight = f'{cfg.model_name}.pt'
        RESULTS_PATH = '../results'
        os.system(f'WANDB_MODE="dryrun" python train.py --project {RESULTS_PATH}/{opt.config} --name fold_{opt.fold} --hyp {cfg.yaml} --img {cfg.image_size} --batch {cfg.batch_size} --epochs {cfg.epochs} --data results/{opt.config}/fold{fold}/covid.yaml --weights {weight} --cache --exist-ok')
        print('train end.')

    oof_image_dir = f'{input_dir}/images/{opt.config}/fold{fold}'
    if os.path.exists(oof_image_dir):
        os.system(f'rm -r {oof_image_dir}')

    os.system(f'mkdir -p {oof_image_dir}')
    for file in oof_files:
        os.system(f'cp {file} {oof_image_dir}/')

    # oof pred
    if cfg.predict_valid:
        if os.path.exists(f'{RESULTS_PATH}/{opt.config}/oof_{fold}/labels'):
            os.system(f'rm -r {RESULTS_PATH}/{opt.config}/oof_{fold}/labels')

        os.system(f"python detect.py --project {RESULTS_PATH}/{opt.config} --name oof_{opt.fold} --weights {RESULTS_PATH}/{opt.config}/fold_{opt.fold}/weights/best.pt --img {int(cfg.image_size*1.0)} --conf 0.00001 --iou 0.2 --source {oof_image_dir} --save-txt --save-conf --exist-ok --max-det 100")

    # test pred
    if cfg.predict_test:
        if os.path.exists(f'{RESULTS_PATH}/{opt.config}/pred_{fold}/labels'):
            os.system(f'rm -r {RESULTS_PATH}/{opt.config}/pred_{fold}/labels')
        os.system(f"python detect.py --project {RESULTS_PATH}/{opt.config} --name pred_{opt.fold} --weights {RESULTS_PATH}/{opt.config}/fold_{opt.fold}/weights/best.pt --img {int(cfg.image_size*1.0)} --conf 0.00001 --iou 0.2 --source {cfg.test_image_path} --save-txt --save-conf --exist-ok --max-det 100")

    val_df = train_df
    val_df = val_df[val_df.image_id.isin(oof.image_id.unique())]

    for df, mode in zip([val_df], ['oof']):
        if ((mode == 'oof') & (not cfg.predict_valid) or ((mode == 'pred') & (not cfg.predict_test))):
            continue
        image_ids = []
        pred_array_list = []

        for i, file_path in enumerate(glob(f'{RESULTS_PATH}/{opt.config}/{mode}_{fold}/labels/*txt')):
            image_id = file_path.split('/')[-1].replace('.txt', '')
            image_id_df = df.loc[df.image_id==image_id,['width', 'height']]
            if len(image_id_df)==0:
                continue
            w, h = image_id_df.values[0]
            f = open(file_path, 'r')
            f_read = f.read()
            if f_read == '':
                continue

            data = np.array(f_read.replace('\n', ' ').strip().split(' ')).astype(np.float32).reshape(-1, 6)
            data = data[:, [0, 5, 1, 2, 3, 4]]
            bboxes = np.concatenate((data[:, :2], np.round(yolo2voc(h, w, data[:, 2:]))), axis =1)
            pred_array = np.append([[image_id]]*len(bboxes), bboxes, axis=1)
            pred_array_list.append(pred_array)
        pred = pd.DataFrame(np.vstack(pred_array_list), columns=['image_id', 'class_id', 'conf', 'x_min', 'y_min', 'x_max', 'y_max'])
        print(mode)
        print(pred)
        pred.to_csv(f'{RESULTS_PATH}/{opt.config}/{mode}_fold{fold}.csv',index = False)
    os.system(f'rm -r {oof_image_dir}')
