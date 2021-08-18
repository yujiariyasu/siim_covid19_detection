import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score


df = pd.read_csv('../../predictions/rsna18-model-on-covid-data/mk001.csv')
gt_df = pd.read_csv('../../data/covid/train_study_level.csv')
gt_df['label'] = np.argmax(gt_df.iloc[:,1:].values, axis=1)
# 0=Negative, 1=Typical, 2=Indeterminate, 3=Atypical
gt_dict = {0: 0, 1: 2, 2: 2, 3: 1}
gt_df['label'] = gt_df.label.map(gt_dict)

df['id'] = df.imgfile.apply(lambda x: x.split('/')[-3] + '_study')
dfm = df.merge(gt_df, on='id')

accuracy = np.mean(np.argmax(dfm[['p0','p1','p2']].values, axis=1) == dfm.label)
print(f'ACCURACY={accuracy*100:0.2f}%')

for i in range(3):
    if np.sum(dfm.label == i) == 0: continue
    print(f'AUC({i})={roc_auc_score(dfm.label == i, dfm[f"p{i}"]):0.4f}')
    print(f'AVP({i})={average_precision_score(dfm.label == i, dfm[f"p{i}"]):0.4f}')

