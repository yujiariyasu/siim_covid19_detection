import numpy as np
import pandas as pd
import os, os.path as osp

from ..builder import build_dataset, build_dataloader


def get_train_val_test_splits(cfg, df): 
    if 'split' in df.columns and cfg.data.use_fixed_splits:
        train_df = df[df.split == 'train']
        valid_df = df[df.split == 'valid']
        test_df  = df[df.split == 'test']
        return train_df, valid_df, test_df

    i, o = cfg.data.inner_fold, cfg.data.outer_fold
    if isinstance(i, (int,float)):
        print(f'<inner fold> : {i}')
        print(f'<outer fold> : {o}')
        test_df = df[df.outer == o]
        df = df[df.outer != o]
        train_df = df[df[f'inner{o}'] != i]
        valid_df = df[df[f'inner{o}'] == i]
        valid_df = valid_df.drop_duplicates().reset_index(drop=True)
        test_df = test_df.drop_duplicates().reset_index(drop=True)
    else:
        print('No inner fold specified ...')
        print(f'<outer fold> : {o}')
        train_df = df[df.outer != o]
        valid_df = df[df.outer == o]
        valid_df = valid_df.drop_duplicates().reset_index(drop=True)
        test_df = valid_df
    return train_df, valid_df, test_df


def prepend_filepath(lst, prefix): 
    return np.asarray([osp.join(prefix, item) for item in lst])


def get_train_val_datasets(cfg):
    INPUT_COL = cfg.data.input
    LABEL_COL = cfg.data.target

    df = pd.read_csv(cfg.data.annotations) 
        
    train_df, valid_df, _ = get_train_val_test_splits(cfg, df)
    data_dir = cfg.data.data_dir
    train_inputs = prepend_filepath(train_df[INPUT_COL], data_dir)
    valid_inputs = prepend_filepath(valid_df[INPUT_COL], data_dir)

    train_labels = train_df[LABEL_COL].values
    valid_labels = valid_df[LABEL_COL].values

    if cfg.loss.name == 'CrossEntropyLoss' and 'typical' in LABEL_COL:
        train_labels = np.argmax(train_labels, axis=1)
        valid_labels = np.argmax(valid_labels, axis=1)

    train_data_info = dict(inputs=train_inputs, labels=train_labels)
    valid_data_info = dict(inputs=valid_inputs, labels=valid_labels)

    if cfg.data.dataset.name == 'SegmentClassify':
        train_data_info.update(dict(bboxes=train_df.boxes.values))
        valid_data_info.update(dict(bboxes=valid_df.boxes.values))

    train_dataset = build_dataset(cfg, 
        data_info=train_data_info,
        mode='train')
    valid_dataset = build_dataset(cfg, 
        data_info=valid_data_info,
        mode='valid')

    print(f'TRAIN : n={len(train_dataset)}')
    print(f'VALID : n={len(valid_dataset)}')

    return train_dataset, valid_dataset


