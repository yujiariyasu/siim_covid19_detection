"""Six classification labels:
[negative, typical, indeterminate, atypical, opacity, none]
"""
import glob
import numpy as np
import os, os.path as osp
import pandas as pd
import pydicom

from tqdm import tqdm
from utils import create_double_cv


def get_patient_and_study_ids_from_dicom(dcm):
    dcm = pydicom.dcmread(dcm, stop_before_pixels=True)
    return {dcm.StudyInstanceUID: dcm.PatientID}


DATADIR = '../data/covid/'

study_df = pd.read_csv(osp.join(DATADIR, 'train_study_level.csv'))
study_df.columns = ['pid', 'negative', 'typical', 'indeterminate', 'atypical']
study_df = create_double_cv(study_df, 'pid', 5, 5, stratified=None, seed=88)

files = glob.glob(osp.join(DATADIR, 'train_pngs/*/*/*.png'))
files_df = pd.DataFrame({'filename': files})
files_df['filename'] = files_df.filename.apply(lambda x: '/'.join(x.split('/')[-3:]))
files_df['pid'] = files_df.filename.apply(lambda x: x.split('/')[-3]+'_study')

image_df = files_df.merge(study_df, on='pid')
image_df.to_csv('../data/covid/train_kfold_not_cleaned_split_by_study.csv', index=False)


