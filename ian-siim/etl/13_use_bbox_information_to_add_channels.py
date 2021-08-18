import cv2
import glob
import numpy as np
import os, os.path as osp
import pandas as pd
import pickle

from utils import create_directory
from tqdm import tqdm


BASEDIR = '../data/covid/train_pngs/'
SAVEDIR = '../data/covid/train_bbox_channel/'


predictions = glob.glob('../predictions/swin004*pkl')
df = pd.read_csv('../data/covid/train_kfold_cleaned_w_bboxes_yuji.csv')
df['image_name'] = df.filename.apply(lambda x: x.split('/')[-1].replace('.png','') + '_image')

with open(predictions[0], 'rb') as f:
    pred_bboxes = pickle.load(f)


for p in predictions[1:]:
    with open(p, 'rb') as f:
        tmp_pred = pickle.load(f)
        pred_bboxes.update(tmp_pred)

for rownum, image in tqdm(enumerate(df.image_name)):
    fp = df.filename.iloc[rownum]
    img = cv2.imread(osp.join(BASEDIR, fp), 0)
    create_directory(osp.join(SAVEDIR, osp.dirname(fp)))
    bbox = pred_bboxes[image]
    bbox = bbox[bbox[:,-1] > 0.3]
    if len(bbox) == 0:
        img = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)
        img[...,1] = 0
        img[...,2] = 0
    else:
        base_img = np.zeros_like(img)
        box_img = np.zeros_like(img)
        for box in bbox[::-1]: # reverse box order
            x1 = int(np.min(box[0]))
            y1 = int(np.min(box[1]))
            x2 = int(np.max(box[2]))
            y2 = int(np.max(box[3]))
            base_img[y1:y2,x1:x2] = img[y1:y2,x1:x2]
            box_img[y1:y2,x1:x2] = int(box[4] * 255)
        img = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)
        img[...,1] = base_img
        img[...,2] = box_img
    _ = cv2.imwrite(osp.join(SAVEDIR, osp.dirname(fp), image.split('_')[0]+'.png'), img)

