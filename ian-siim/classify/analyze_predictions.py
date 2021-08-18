import numpy as np
import pandas as pd

from scipy.stats import pearsonr, rankdata
from sklearn.metrics import roc_auc_score


LABELS = ['ett_abnormal', 'ett_borderline', 'ett_normal', 'ngt_abnormal',
          'ngt_borderline', 'ngt_incomplete', 'ngt_normal', 'cvc_abnormal',
          'cvc_borderline', 'cvc_normal', 'swan_ganz']


def rank_transform(mat):
    return np.concatenate([np.expand_dims(rankdata(mat[:,i]), axis=1) for i in range(mat.shape[1])], axis=1)


def get_single_auc(exp='inf000', fold=0, subset=None):
    global LABELS
    x = pd.read_csv(f'../predictions/{exp}/fold{fold}.csv')
    t = pd.read_csv('../data/train_folds_kaggle_usersin.csv')
    del t['imgfile']
    if 'StudyInstanceUID' not in x.columns:
        x['StudyInstanceUID'] = x.imgfile.apply(lambda x: x.split('/')[-1].replace('.jpg', ''))
    x = x.merge(t, on='StudyInstanceUID')
    indices = range(len(LABELS))
    _labels = LABELS
    if subset == 'ett':
        x = x[x.ett_present == 1] ; _labels = LABELS[:3]
        indices = range(3)
    elif subset == 'ngt':
        x = x[x.ngt_present == 1] ; _labels = LABELS[3:7]
        indices = range(3, 7)
    elif subset == 'cvc':
        x = x[x.cvc_present == 1] ; _labels = LABELS[7:]
        indices = range(7, 11)
    auc = roc_auc_score(x[_labels], x[[f'p{i}' for i in indices]].values)
    print(f'AUC   : {auc:.4f}')
    for i in indices:
        print(f'AUC {i} : {roc_auc_score(x[_labels[i]], x[f"p{i}"]):.4f}')


def get_auc(fold=0, weights=[1,1], print_cor=False):
    global LABELS
    x = pd.read_csv(f'../predictions/mk035/fold{fold}.csv')
    y = pd.read_csv(f'../predictions/mk019/fold{fold}.csv')
    t = pd.read_csv('../data/train_folds_kaggle_usersin.csv')
    del t['imgfile']
    x['StudyInstanceUID'] = x.imgfile.apply(lambda x: x.split('/')[-1].replace('.jpg', ''))
    x = x.merge(t, on='StudyInstanceUID')
    auc = roc_auc_score(x[LABELS], x[[f'p{i}' for i in range(11)]].values)
    print(f'AUC_x   : {auc:.4f}')
    y['StudyInstanceUID'] = y.imgfile.apply(lambda y: y.split('/')[-1].replace('.jpg', ''))
    y = y.merge(t, on='StudyInstanceUID')
    auc = roc_auc_score(y[LABELS], y[[f'p{i}' for i in range(11)]].values)
    print(f'AUC_y   : {auc:.4f}')
    auc = roc_auc_score(x[LABELS], weights[0]*x[[f'p{i}' for i in range(11)]].values ** 0.5 + \
        weights[1]*y[[f'p{i}' for i in range(11)]].values ** 0.5)
    print(f'AUC_x+y : {auc:.4f}')
    if print_cor:
        print('=======')
        for i in range(11):
            cor = pearsonr(x[f'p{i}'], y[f'p{i}'])
            print(f'COR_{i:02d}  : {cor[0]:.4f}')


def get_individual_auc(fold=0, weights=[1,1], print_cor=False):
    global LABELS
    x = pd.read_csv(f'../predictions/mk016/fold{fold}.csv')
    y = pd.read_csv(f'../predictions/mk019/fold{fold}.csv')
    t = pd.read_csv('../data/train_folds_kaggle_usersin.csv')
    del t['imgfile']
    x['StudyInstanceUID'] = x.imgfile.apply(lambda x: x.split('/')[-1].replace('.jpg', ''))
    x = x.merge(t, on='StudyInstanceUID')
    y['StudyInstanceUID'] = y.imgfile.apply(lambda y: y.split('/')[-1].replace('.jpg', ''))
    y = y.merge(t, on='StudyInstanceUID')
    for i in range(11):
        if i < 3:
            w = [1.5, 1]
        if i > 6:
            w = [1, 1.5]
        auc_x = f'AUC_x{i} : {roc_auc_score(x[LABELS[i]], x[f"p{i}"]):.4f}'
        auc_y = f'AUC_y{i} : {roc_auc_score(y[LABELS[i]], y[f"p{i}"]):.4f}'
        auc_z = f'AUC_z{i} : {roc_auc_score(x[LABELS[i]], w[0]*x[f"p{i}"]+w[1]*y[f"p{i}"]):.4f}'
        print(f'{auc_x} / {auc_y} / {auc_z}')
#     auc = roc_auc_score(x[LABELS], weights[0]*x[[f'p{i}' for i in range(11)]].values + \
#         weights[1]*y[[f'p{i}' for i in range(11)]].values)
#     print(f'AUC_x+y : {auc:.4f}')
#     if print_cor:
#         print('=======')
#         for i in range(11):
#             cor = pearsonr(x[f'p{i}'], y[f'p{i}'])
#             print(f'COR_{i:02d}  : {cor[0]:.4f}')


get_auc(0)
get_auc(1)
get_auc(2)