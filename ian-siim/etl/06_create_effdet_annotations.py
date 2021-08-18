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


DATADIR = '../data/covid/'
PNGSDIR = osp.join(DATADIR, 'train_pngs/')

bbox_df = pd.read_csv(osp.join(DATADIR, 'train_kfold_cleaned.csv'))

annotations = []
for i in tqdm(range(len(bbox_df)), total=len(bbox_df)):
    ann = dict(filename=bbox_df.filename.iloc[i])
    img = Image.open(osp.join(PNGSDIR, ann['filename']))
    ann['img_width'], ann['img_height'] = img.size
    box_list = bbox_df.boxes.iloc[i]
    if not isinstance(box_list, str) and np.isnan(box_list):
        ann['bbox'] = np.zeros((0,4)).astype('float')
        ann['cls'] = np.zeros((0,)).astype('int')
    else:
        bboxes = get_bboxes(box_list)
        bboxes[:,[0,2]] = np.clip(bboxes[:,[0,2]], 0, ann['img_width'])
        bboxes[:,[1,3]] = np.clip(bboxes[:,[1,3]], 0, ann['img_height'])
        ann['bbox'] = bboxes
        ann['cls'] = np.ones((bboxes.shape[0],)).astype('int')
    ann['class_label'] = bbox_df[['negative','atypical','indeterminate','typical','none']].iloc[i].values
    assert ann['cls'].shape[0] == ann['bbox'].shape[0]
    ann['folds'] = dict(outer=bbox_df.outer.iloc[i])
    for column in bbox_df.columns:
        if 'inner' in column:
            ann['folds'].update({column : bbox_df[column].iloc[i]})
    annotations += [ann]


with open(osp.join(DATADIR, 'train_bbox_annotations_effdet.pkl'), 'wb') as f:
    pickle.dump(annotations, f)


