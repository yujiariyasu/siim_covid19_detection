import numpy as np
import pandas as pd

from collections import defaultdict


def simplify_label(l):
    l = l.lower()
    labels = ['negative', 'indeterminate', 'atypical', 'typical']
    for each_label in labels:
        if each_label in l: return each_label


df = pd.read_csv('../data/covid/train_kfold_cleaned.csv')
with open('../data/ricord_in_kaggle_train.txt') as f:
    train_dupes = [l.strip() for l in f.readlines()]


train_dupes = [_.split('/')[-1] for _ in train_dupes]
ext_df = pd.read_csv('../data/ricord-kaggle/MIDRC-RICORD-meta.csv')
ext_df = ext_df[~ext_df.fname.isin(train_dupes)]

# Parse labels
# 3 annotations, separated by commas
plurality_labels = []
for rownum, row in ext_df.iterrows():
    if str(row.labels) == 'nan': 
        plurality_labels.append(row.labels)
        continue
    labels = row.labels.split(',')
    labels = [simplify_label(l) for l in labels]
    # Plurality vote
    labels, counts = np.unique(labels, return_counts=True)
    indices = np.argsort(counts)[::-1]
    labels, counts = labels[indices], counts[indices]
    if np.max(counts) > 1 or len(labels) == 1:
        plurality_labels.append(labels[0])
    else:
        plurality_labels.append('no_consensus')

plurality_labels = np.asarray(plurality_labels)
np.unique(plurality_labels, return_counts=True)

ext_df['label'] = plurality_labels
ext_df = ext_df[~ext_df.label.isin(['nan','no_consensus'])]
labels = ['negative', 'atypical', 'indeterminate', 'typical']
for lab in labels:
    ext_df[lab] = 0
    ext_df.loc[ext_df.label == lab,lab] = 1

# Need to add some prefixes to filenames
df['filename'] = df.filename.apply(lambda x: f'covid/train_pngs/{x}')
ext_df['filename'] = ext_df.fname.apply(lambda x: f'ricord-kaggle/images/MIDRC-RICORD/{x}')
for c in df.columns:
    if 'inner' in c or 'outer' in c:
        ext_df[c] = -1

df = pd.concat([df, ext_df])
df.to_csv('../data/train_kfold_cleaned_plus_ricord_external_without_dupes.csv', index=False)


