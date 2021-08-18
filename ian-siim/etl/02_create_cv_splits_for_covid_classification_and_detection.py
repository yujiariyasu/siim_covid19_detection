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

dicoms = glob.glob(osp.join(DATADIR, 'train/*/*/*.dcm'))
study_to_pid = list(map(lambda dcm: get_patient_and_study_ids_from_dicom(dcm), tqdm(dicoms)))
study_to_pid = {k:v for d in study_to_pid for k,v in d.items()}

image_df = pd.read_csv(osp.join(DATADIR, 'train_image_level.csv'))
study_df = pd.read_csv(osp.join(DATADIR, 'train_study_level.csv'))
study_df['StudyInstanceUID'] = study_df.id.apply(lambda x: x.split('_')[0])
study_df['pid'] = study_df.StudyInstanceUID.map(study_to_pid)
study_df = create_double_cv(study_df, 'pid', 5, 5, stratified=None, seed=88)

files = glob.glob(osp.join(DATADIR, 'train_pngs/*/*/*.png'))
files_df = pd.DataFrame({'filename': files})
files_df['StudyInstanceUID'] = files_df.filename.apply(lambda x: x.split('/')[-3])
files_df['id'] = files_df.filename.apply(lambda x: x.split('/')[-1].replace('.png', '_image'))
image_df = image_df.merge(files_df, on=['id', 'StudyInstanceUID'])
image_df['_label'] = image_df.label.apply(lambda x: x.split()[0])
image_df['SeriesInstanceUID'] = image_df.filename.apply(lambda x: x.split('/')[-2])
image_df['SOPInstanceUID'] = image_df.id.apply(lambda x: x.split('_')[0])

# Find studies where some but not all images contain bounding box
study_list = []
for pid, study in image_df.groupby('StudyInstanceUID'):
    if len(study._label.unique()) > 1: study_list += [study]

print(f'Found {len(study_list)} studies !')

# I reviewed these images, and pretty much all images w/o bbox are
# duplicates or near-duplicates of the image w/ bbox, so this must
# be data error 

# Thus, I will exclude these images (more important for opacity/none
# classification)
discordant_studies = [s.StudyInstanceUID.unique()[0] for s in study_list]
image_df = image_df[~image_df.StudyInstanceUID.isin(discordant_studies)]

discordant_studies = pd.concat([s[s._label == 'opacity'] for s in study_list])

image_df = pd.concat([image_df, discordant_studies])
del image_df['id'] ; del study_df['id']
image_df = image_df.merge(study_df, on='StudyInstanceUID')
image_df['opacity'] = (image_df._label == 'opacity').astype('int')
colnames = list(image_df.columns)
colnames[7:11] = ['negative', 'typical', 'indeterminate', 'atypical']
image_df.columns = colnames
image_df['filename'] = image_df.filename.apply(lambda x: '/'.join(x.split('/')[-3:]))
for c in ['negative','typical','indeterminate','atypical','opacity']:
    print(f'{c}: n={image_df[c].sum()}')

image_df['none'] = 1 - image_df.opacity

image_df.to_csv('../data/covid/train_kfold_cleaned.csv', index=False)


