import json
import numpy as np
import pandas as pd

from collections import defaultdict


with open('../data/ricord/annotations.json', 'r') as f:
    mdai_ann = json.load(f)

dset = mdai_ann['datasets'][0]
ann = dset['annotations']

ann_dict = defaultdict(list)
for a in ann:
    ann_dict['StudyInstanceUID'].append(a['StudyInstanceUID'])
    ann_dict['label'].append(a['labelId'])

ann_df = pd.DataFrame(ann_dict)

labelgroups = mdai_ann['labelGroups']
# Keep track of which labels belong to which groups 
# Assuming these are different annotators, and that some studies
# are multi-read
group_identifier_dict = {}
label_dict = {} 
for ind, group in enumerate(labelgroups):
    for label in group['labels']:
        group_identifier_dict[label['id']] = ind
        label_dict[label['id']] = label['name']


ann_df['group'] = ann_df['label'].map(group_identifier_dict)
ann_df['label'] = ann_df['label'].map(label_dict)

number_of_anns = [len(_df.group.unique()) for uid,_df in ann_df.groupby('StudyInstanceUID')]
print(f'Number of annotators/study ranges from {np.min(number_of_anns)} to {np.max(number_of_anns)}.')

# Turn long to wide
new_labelnames = ['negative', 'atypical', 'indeterminate', 'typical', 'invalid', 'mild', 'moderate', 'severe']
dflist = []
for uid,_df in ann_df.groupby('StudyInstanceUID'):
    tmp_dict = {'StudyInstanceUID': uid}
    for each_label in new_labelnames:
        for each_group in range(6):
            tmp_dict[f'{each_label}_{each_group}'] = 0
    for rownum, row in _df.iterrows():
        tmp_dict[f'{row.label.split()[0].lower()}_{row.group}'] = 1
    tmp_df = pd.DataFrame({k: [v] for k,v in tmp_dict.items()})
    tmp_df['num_annotators'] = len(_df.group.unique())
    dflist.append(tmp_df)

wide_df = pd.concat(dflist)


