import cv2
import glob
import numpy as np
import os, os.path as osp
import pandas as pd
import pickle

from utils import create_directory
from tqdm import tqdm


BASEDIR = '../data/covid/train_pngs/'
SAVEDIR = '../data/covid/train_cropped/'


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
    # Take top 2 bboxes
    bbox = bbox[:2]
    xmin = int(np.min(bbox[:,0]))
    ymin = int(np.min(bbox[:,1]))
    xmax = int(np.max(bbox[:,2]))
    ymax = int(np.max(bbox[:,3]))
    cropped_img = img[ymin:ymax+1, xmin:xmax+1]
    _ = cv2.imwrite(osp.join(SAVEDIR, osp.dirname(fp), image.split('_')[0]+'.png'), cropped_img)

