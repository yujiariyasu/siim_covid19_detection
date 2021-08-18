import numpy as np
import os, os.path as osp
import pydicom

from pydicom.pixel_data_handlers.util import apply_voi_lut
from sklearn.model_selection import KFold, StratifiedKFold


def create_directory(d):
    if not osp.exists(d): os.makedirs(d)


def create_double_cv(df, id_col, outer_splits, inner_splits, stratified=None, seed=88):
    df['outer'] = 888
    splitter = KFold if stratified is None else StratifiedKFold
    outer_spl = splitter(n_splits=outer_splits, shuffle=True, random_state=seed)
    outer_counter = 0
    for outer_train, outer_test in outer_spl.split(df) if stratified is None else outer_spl.split(df, df[stratified]):
        df.loc[outer_test, 'outer'] = outer_counter
        inner_spl = splitter(n_splits=inner_splits, shuffle=True, random_state=seed)
        inner_counter = 0
        df['inner{}'.format(outer_counter)] = 888
        inner_df = df[df['outer'] != outer_counter].reset_index(drop=True)
        # Determine which IDs should be assigned to inner train
        for inner_train, inner_valid in inner_spl.split(inner_df) if stratified is None else inner_spl.split(inner_df, inner_df[stratified]):
            inner_train_ids = inner_df.loc[inner_valid, id_col]
            df.loc[df[id_col].isin(inner_train_ids), 'inner{}'.format(outer_counter)] = inner_counter
            inner_counter += 1
        outer_counter += 1
    return df


def load_dicom(fp, fake_rgb=False):
    dcm = pydicom.dcmread(fp)
    try:
        arr = apply_voi_lut(dcm.pixel_array, dcm)
    except Exception as e:
        print(e)
        arr = dcm.pixel_array.astype('float32')
    arr = arr - np.min(arr)
    arr = arr / np.max(arr)
    arr = arr * 255.0
    arr = arr.astype('uint8')
    if dcm.PhotometricInterpretation != 'MONOCHROME2':
        arr = np.invert(arr)
    if fake_rgb: arr = np.repeat(np.expand_dims(arr, axis=-1), 3, axis=-1)
    return arr