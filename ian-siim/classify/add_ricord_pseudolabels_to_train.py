import cv2
import glob
import numpy as np
import pandas as pd
import os, os.path as osp

from tqdm import tqdm


# First, load segmentation predictions from each fold
# Average them and resave
SAVE_DIR = '../pseudosegs/ricord_avg'
if not osp.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

pseudosegs = glob.glob('../pseudosegs/ricord_0/*')
for ps in tqdm(pseudosegs):
    seglist = []
    for fold in range(5):
        seglist.append(cv2.imread(ps.replace('ricord_0', f'ricord_{fold}'), 0).astype('float32'))
    seglist = np.asarray(seglist)
    seglist = np.mean(seglist, axis=0)
    seglist = seglist.astype('uint8')
    new_fp = ps.replace('ricord_0', 'ricord_avg')
    status = cv2.imwrite(new_fp, seglist)


# Load in classification predictions for each fold and average
pseudocls = glob.glob('../predictions/ricord*csv')
pseudocls = [pd.read_csv(pc) for pc in pseudocls]

assert np.sum(pseudocls[0].imgfile.values == pseudocls[1].imgfile.values) == len(pseudocls[0])
pred_array = np.asarray([pc[['p0','p1','p2','p3','p4']].values for pc in pseudocls])
pred_array = np.mean(pred_array, axis=0)
pseudocls = pseudocls[0]
pseudocls[['p0','p1','p2','p3','p4']] = pred_array

df = pd.read_csv('../data/covid/train_kfold_cleaned_w_bboxes_yuji.csv')
# Load in duplicate RICORD images in training set
with open('../data/ricord_in_kaggle_train.txt') as f:
    dupes = [l.strip() for l in f.readlines()]

pseudocls = pseudocls[~pseudocls.imgfile.isin(dupes)]
pseudocls.columns = ['imgfile', 'negative', 'atypical', 'indeterminate', 'typical', 'none']
pseudocls['filename'] = pseudocls.imgfile.apply(lambda x: x.replace('../data/', ''))
pseudocls['outer'] = -1
pseudocls['boxes'] = pseudocls.filename.apply(lambda x: x.replace('ricord-kaggle/images/MIDRC-RICORD/', '../pseudosegs/ricord_avg/'))

df['filename'] = df.filename.apply(lambda x: osp.join('covid/train_pngs', x))
df = pd.concat([df, pseudocls])
df.to_csv('../data/covid/train_kfold_cleaned_w_bboxes_plus_ricord_pseudo.csv', index=False)

