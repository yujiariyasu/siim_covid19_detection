import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.optimize import linear_sum_assignment
import sys
import random
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error, average_precision_score, label_ranking_average_precision_score
import copy
import warnings
warnings.simplefilter('ignore')

from tqdm import tqdm

from ensemble_boxes_bk import *

def wbf(preds, iou_thr = 0.5, skip_box_thr = 0.0001, weights=None):
    dfs = []
    for i, pred in enumerate(preds):
        pred['pred_n'] = i
        dfs.append(pred)
    df = pd.concat(dfs)

    results = []
    image_ids = df["image_id"].unique()

    for nnn, image_id in enumerate(tqdm(image_ids, total=len(image_ids))):
#         if image_id != '4cdcc3f97bf7':
#             continue

        # All annotations for the current image.
        data = df[df["image_id"] == image_id]
        data = data.reset_index(drop=True)

        annotations = {}

        # WBF expects the coordinates in 0-1 range.
        max_value = data[["x_min", "y_min", "x_max", "y_max"]].values.max()
        data[["x_min", "y_min", "x_max", "y_max"]] = data[["x_min", "y_min", "x_max", "y_max"]] / max_value

        # Loop through all of the annotations
        for idx, row in data.iterrows():
            pred_n = row["pred_n"]
            if pred_n not in annotations:
                annotations[pred_n] = {
                    "boxes_list": [],
                    "scores_list": [],
                    "labels_list": [],
                }

                # We consider all of the data as equal.

            annotations[pred_n]["boxes_list"].append([row["x_min"], row["y_min"], row["x_max"], row["y_max"]])
            annotations[pred_n]["scores_list"].append(row["conf"])
            annotations[pred_n]["labels_list"].append(row["class_id"])

        boxes_list = []
        scores_list = []
        labels_list = []

        for annotator in annotations.keys():
            boxes_list.append(annotations[annotator]["boxes_list"])
            scores_list.append(annotations[annotator]["scores_list"])
            labels_list.append(annotations[annotator]["labels_list"])

        # Calculate WBF
#         try:
#         print(image_id)
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr
        )
#         except:
#             import pdb;pdb.set_trace()
            
        for idx, box in enumerate(boxes):
            results.append({
                "image_id": image_id,
                "class_id": int(labels[idx]),
                "pred_n": "wbf",
                'conf':scores[idx],
                "x_min": box[0] * max_value,
                "y_min": box[1] * max_value,
                "x_max": box[2] * max_value,
                "y_max": box[3] * max_value,
            })

    results = pd.DataFrame(results)
    return results

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def nms_one_fold(pred, nms_iou_th, all_size_df):
    id_list = []
    class_id_list = []
    score_list = []
    box_list = []
    use_cols = ['x_min', 'y_min', 'x_max', 'y_max']
    for id, one_id_pred in pred.groupby('image_id'):
        size_df = all_size_df[all_size_df.image_id==id]
        scale = np.max([size_df.width.values[0], size_df.height.values[0]])
        for class_id in one_id_pred.class_id.unique():
            pr = one_id_pred.query('class_id == @class_id')
            boxes = pr[use_cols].values/scale
            scores = pr.conf.values
            class_ids = pr.class_id.values
            boxes, scores, class_ids = nms([boxes], [scores], [class_ids], iou_thr=nms_iou_th)
            boxes = (boxes*scale).astype(int)
            box_list += boxes.tolist()
            class_id_list += class_ids.tolist()
            score_list += scores.tolist()
            id_list += [id]*len(scores)

    new_pred = pd.DataFrame(id_list, columns=['image_id'])
    new_pred['class_id'] = class_id_list
    new_pred['conf'] = score_list
    new_pred[['x_min', 'y_min', 'x_max', 'y_max']] = box_list
    return new_pred


yolo_dirs = ['mixup05_l', 'mixup05_l6']

train_with_size = pd.read_csv('input/train_with_size.csv')
train_with_size = train_with_size.rename(columns={'dim0': 'height', 'dim1': 'width'})
oofs = []
for di in yolo_dirs:
    one_dir_oofs = []
    for fold in range(5):
        one_fold_oof = pd.read_csv(f'results/{di}/oof_fold{fold}.csv')
        one_fold_oof = one_fold_oof[one_fold_oof.conf>0.00001]

        one_dir_oofs.append(one_fold_oof)
    oof = nms_one_fold(pd.concat(one_dir_oofs), nms_iou_th=0.45, all_size_df=train_with_size)
    oofs.append(oof)

oof = wbf(oofs, iou_thr = 0.6, skip_box_thr = 0.0, weights = [1]*len(oofs))


oof.to_csv('input/yolo_oof.csv', index=False)
