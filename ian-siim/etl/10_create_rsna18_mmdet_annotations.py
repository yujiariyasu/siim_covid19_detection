import ast
import glob
import numpy as np
import os, os.path as osp
import pandas as pd
import pickle

from PIL import Image
from tqdm import tqdm


def get_bboxes(box_list):
    box_list = ast.literal_eval(box_list)
    boxes = []
    for box in box_list:
        boxes.append(np.expand_dims(np.array([box['x'], box['y'], box['width'], box['height']]), axis=0))
    boxes = np.concatenate(boxes)
    boxes[:,2] += boxes[:,0] ; boxes[:,3] += boxes[:,1]
    return boxes


DATADIR = '../data/rsna18/'
DCMSDIR = osp.join(DATADIR, 'stage_2_train_images/')

df = pd.read_csv(osp.join(DATADIR, 'train_kfold_for_covid.csv'))
bbox_df = pd.read_csv(osp.join(DATADIR, 'stage_2_train_labels.csv'))
bbox_df = bbox_df.merge(df, on='patientId')
# Exclude negatives

annotations = []
for pid, _df in bbox_df.groupby('filename'):
    ann = dict(filename=pid)
    ann['width'], ann['height'] = 1024, 1024
    ann['ann'] = {}
    if _df.x.isna().sum() > 0: 
        ann['ann']['bboxes'] = np.zeros((0,4)).astype('float')
        ann['ann']['labels'] = np.zeros((0,)).astype('int')
    else:
        bboxes = _df[['x','y','width', 'height']].values
        bboxes[:,2] += bboxes[:,0]
        bboxes[:,3] += bboxes[:,1]
        bboxes[:,[0,2]] = np.clip(bboxes[:,[0,2]], 0, ann['width'])
        bboxes[:,[1,3]] = np.clip(bboxes[:,[1,3]], 0, ann['height'])
        ann['ann']['bboxes'] = bboxes.astype('float')
        ann['ann']['labels'] = np.zeros((bboxes.shape[0],)).astype('int')
    assert ann['ann']['labels'].shape[0] == ann['ann']['bboxes'].shape[0]
    ann['folds'] = dict(outer=_df.outer.iloc[0])
    for column in _df.columns:
        if 'inner' in column:
            ann['folds'].update({column : _df[column].iloc[0]})
    annotations += [ann]


with open(osp.join(DATADIR, 'train_bbox_annotations_mmdet.pkl'), 'wb') as f:
    pickle.dump(annotations, f)


