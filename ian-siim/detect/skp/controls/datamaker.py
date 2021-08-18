import numpy as np
import pandas as pd
import pickle
import os, os.path as osp

from ..builder import build_dataset, build_dataloader


def unique_list_of_dicts(Ld):
    return list({v['filename']: v for v in Ld}.values())


def get_train_val_test_splits(cfg, ann): 
    
    i, o = cfg.data.inner_fold, cfg.data.outer_fold
    if isinstance(i, (int,float)):
        print(f'<inner fold> : {i}')
        print(f'<outer fold> : {o}')
        test_ann = [a for a in ann if a['folds']['outer'] == o]
        ann = [a for a in ann if a['folds']['outer'] != o]
        train_ann = [a for a in ann if a['folds'][f'inner{o}'] != i]
        valid_ann = [a for a in ann if a['folds'][f'inner{o}'] == i]
        valid_ann, test_ann = unique_list_of_dicts(valid_ann), unique_list_of_dicts(test_ann)
    else:
        print('No inner fold specified ...')
        print(f'<outer fold> : {o}')
        valid_ann = [a for a in ann if a['folds']['outer'] == o]
        train_ann = [a for a in ann if a['folds']['outer'] != o]
        valid_ann = unique_list_of_dicts(valid_ann)
        test_ann = valid_ann
    return train_ann, valid_ann, test_ann

    
def prepend_filepath(ann, prefix): 
    for a in ann:
        a['filename'] = osp.join(prefix, a['filename'])
    return ann


def get_train_val_datasets(cfg):

    with open(cfg.data.annotations, 'rb') as f:
        ann = pickle.load(f)

    train_ann, valid_ann, _ = get_train_val_test_splits(cfg, ann)
    
    if cfg.data.exclude_negative_train:
        train_ann = [a for a in train_ann if len(a['bbox']) > 0]
    if cfg.data.exclude_negative_valid:
        valid_ann = [a for a in valid_ann if len(a['bbox']) > 0]

    data_dir = cfg.data.data_dir
    train_ann = prepend_filepath(train_ann, data_dir)
    valid_ann = prepend_filepath(valid_ann, data_dir)

    train_dataset = build_dataset(cfg, 
        data_info=dict(annotations=train_ann),
        mode='train')
    valid_dataset = build_dataset(cfg, 
        data_info=dict(annotations=valid_ann),
        mode='valid')

    print(f'TRAIN : n={len(train_dataset)}')
    print(f'VALID : n={len(valid_dataset)}')

    return train_dataset, valid_dataset
