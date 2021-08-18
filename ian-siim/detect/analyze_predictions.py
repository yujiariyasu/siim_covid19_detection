import glob
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score


preds = glob.glob('../predictions/seg019/fold*csv')
preds = pd.concat([pd.read_csv(p) for p in preds])
preds['filename'] = preds.imgfile.apply(lambda x: x.replace('../data/covid/train_pngs/', ''))
df = pd.read_csv('../data/covid/train_kfold_cleaned_w_bboxes_yuji.csv')
df = df.merge(preds, on='filename')

labels = ['negative', 'atypical', 'indeterminate', 'typical', 'none']
avp_dict = {}
for i,l in enumerate(labels):
    tmp_avp = average_precision_score(df[l], df[f'p{i}']) 
    avp_dict[l] = tmp_avp
    print(f'AVP-{l.upper()}: {tmp_avp:0.4f}')

print(np.mean([avp_dict[k] for k in labels[:4]])*2/3)


average_precision_score(df.none, (1-df.p5)+df.p4)

df.to_csv('../oof_seg019_classification.csv', index=False)