import pandas as pd

from utils import create_double_cv


df = pd.read_csv('../data/rsna18/stage_2_detailed_class_info.csv')
df = df.drop_duplicates().reset_index(drop=True)
assert len(df.patientId.unique()) == len(df)

label_dict = {
    'Normal': 0,
    'No Lung Opacity / Not Normal': 1,
    'Lung Opacity': 2
}
df['label'] = df['class'].map(label_dict)
df = create_double_cv(df, 'patientId', 10, 10, stratified=None, seed=88)
df['filename'] = df.patientId.apply(lambda x: f'{x}.dcm')
df.to_csv('../data/rsna18/train_kfold_for_covid.csv', index=False)