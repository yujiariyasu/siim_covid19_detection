import glob
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import average_precision_score


def load_pickle(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)


def reformat_predictions(prediction):
    pred_list = prediction['predictions']
    all_pred_list = []
    for element in pred_list:
        # List of N classes
        # Each element in list is (n_bboxes, 5)
        plist = []
        for each_class in element:
            if len(each_class) == 0:
                p = 0
            else:
                p = each_class[0][-1]
            plist.append(p)
        all_pred_list.append(plist)
    df = pd.DataFrame(np.asarray(all_pred_list))
    df.columns = ['negative', 'indeterminate', 'atypical', 'typical']
    df['filename'] = prediction['filename']
    df['filename'] = df.filename.apply(lambda x: x.replace('../data/covid/train/', '').replace('dcm','png'))
    return df


preds = glob.glob('../predictions/swin006*pkl')
df = pd.read_csv('../data/covid/train_kfold_cleaned.csv')

preds = [load_pickle(p) for p in preds]

df_list = [reformat_predictions(p) for p in preds]
df_list = [df.merge(each_df, on='filename') for each_df in df_list]

for each_df in df_list:
    avp = 0.
    for col in ['negative','indeterminate','atypical','typical']:
        avp += average_precision_score(each_df[col+'_x'], each_df[col+'_y'])
    avp /= 4.0
    print(f'AVP: {avp:0.4f}')
