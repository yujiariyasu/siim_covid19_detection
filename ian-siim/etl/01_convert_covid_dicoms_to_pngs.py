import cv2
import glob
import numpy as np
import os, os.path as osp

from tqdm import tqdm
from utils import load_dicom, create_directory


def get_uids_from_fp(fp):
    split_fp = fp.split('/')
    return split_fp[-3], split_fp[-2], split_fp[-1].replace('.dcm', '')


DATADIR = '../data/covid/'

train_dicoms = glob.glob(osp.join(DATADIR, 'train/*/*/*.dcm'))
test_dicoms = glob.glob(osp.join(DATADIR, 'test/*/*/*.dcm'))

for t in tqdm(train_dicoms, total=len(train_dicoms)):
    dcm = load_dicom(t)
    study, series, sop = get_uids_from_fp(t)
    new_fp = osp.join(DATADIR, 'train_pngs', study, series)
    create_directory(new_fp)
    new_fp = osp.join(new_fp, f'{sop}.png')
    status = cv2.imwrite(new_fp, dcm)

for t in tqdm(test_dicoms, total=len(test_dicoms)):
    dcm = load_dicom(t)
    study, series, sop = get_uids_from_fp(t)
    new_fp = osp.join(DATADIR, 'test_pngs', study, series)
    create_directory(new_fp)
    new_fp = osp.join(new_fp, f'{sop}.png')
    status = cv2.imwrite(new_fp, dcm)




