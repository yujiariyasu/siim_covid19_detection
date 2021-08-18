import pandas as pd


df = pd.read_csv('../data/rsna18/train_kfold_for_covid.csv')
bbox_df = pd.read_csv('../data/rsna18/stage_2_train_labels.csv')

dflist = []
for pid, _df in bbox_df.groupby('patientId'):
    boxlist = []
    for rownum, row in _df.iterrows():
        if str(row.x) == 'nan':
            boxlist = row.x
            break
        else:
            boxlist.append((row.x, row.y, row.width, row.height))
    dflist.append(pd.DataFrame({'patientId': [pid], 'boxes': [boxlist]}))

bbox_df = pd.concat(dflist)
df = df.merge(bbox_df, on='patientId')
df['negative'] = (df.label == 0).astype('int')
df['not_normal'] = (df.label == 1).astype('int')
df['opacity'] = (df.label == 2).astype('int')

df.to_csv('../data/rsna18/train_kfold_for_covid_w_bboxes.csv', index=False)