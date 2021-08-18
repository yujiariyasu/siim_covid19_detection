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
    ann['img_width'], ann['img_height'] = 1024, 1024
    if _df.x.isna().sum() > 0: 
        ann['bbox'] = np.zeros((0,4)).astype('float')
        ann['cls'] = np.zeros((0,)).astype('int')
    else:
        bboxes = _df[['x','y','width', 'height']].values
        bboxes[:,2] += bboxes[:,0]
        bboxes[:,3] += bboxes[:,1]
        bboxes[:,[0,2]] = np.clip(bboxes[:,[0,2]], 0, ann['img_width'])
        bboxes[:,[1,3]] = np.clip(bboxes[:,[1,3]], 0, ann['img_height'])
        ann['bbox'] = bboxes
        ann['cls'] = np.ones((bboxes.shape[0],)).astype('int')
    assert ann['cls'].shape[0] == ann['bbox'].shape[0]
    ann['folds'] = dict(outer=_df.outer.iloc[0])
    for column in _df.columns:
        if 'inner' in column:
            ann['folds'].update({column : _df[column].iloc[0]})
    annotations += [ann]


with open(osp.join(DATADIR, 'train_bbox_annotations_effdet.pkl'), 'wb') as f:
    pickle.dump(annotations, f)


