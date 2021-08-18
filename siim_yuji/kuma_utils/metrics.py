import os
import sys
import time
import datetime
import argparse
import re
from pathlib import Path
from copy import deepcopy, copy
import traceback
import warnings

import numpy as np
from numba import jit
import pandas as pd
from pdb import set_trace as st

from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error, average_precision_score
from .preprocessing import DistTransformer


class MetricTemplate:
    '''
    Custom metric template

    # Usage
    general:    Metric()(target, approx)
    catboost:   eval_metric=Metric()
    lightgbm:   eval_metric=Metric().lgbm
    pytorch:    Metric().torch(output, labels)
    '''

    def __init__(self, maximize=False):
        self.maximize = maximize

    def __repr__(self):
        return f'{type(self).__name__}(maximize={self.maximize})'

    @jit
    def _test(self, target, approx):
        # Metric calculation
        pass

    def __call__(self, target, approx):
        return self._test(target, approx)

    ### CatBoost
    def get_final_error(self, error, weight):
        return error / weight

    def is_max_optimal(self):
        return self.maximize

    def evaluate(self, approxes, target, weight=None):
        # approxes - list of list-like objects (one object per approx dimension)
        # target - list-like object
        # weight - list-like object, can be None
        assert len(approxes[0]) == len(target)
        if not isinstance(target, np.ndarray):
            target = np.array(target)

        approx = np.array(approxes[0])
        error_sum = self._test(target, approx)
        weight_sum = 1.0

        return error_sum, weight_sum
    
    ### LightGBM
    def lgbm(self, target, approx):
        return self.__class__.__name__, self._test(target, approx), self.maximize

    ### PyTorch
    def torch(self, approx, target):
        if str(type(target))=="<class 'torch.Tensor'>":
            target = target.detach().cpu().numpy()
        if str(type(approx))=="<class 'torch.Tensor'>":
            approx = approx.detach().cpu().numpy()
        return self._test(target, approx)

class SeUnderSp(MetricTemplate):
    '''
    Maximize sensitivity under specific specificity threshold
    '''
    def __init__(self, sp=0.88, maximize=True):
        self.sp = 0.88
        self.maximize = maximize

    def _get_threshold(self, target, approx):
        tn_idx = (target == 0)
        p_tn = np.sort(approx[tn_idx])

        return p_tn[int(len(p_tn) * self.sp)]

    def _test(self, target, approx):
        if not isinstance(target, np.ndarray):
            target = np.array(target)
        if not isinstance(approx, np.ndarray):
            approx = np.array(approx)

        if len(approx.shape) == 1:
            pass
        elif approx.shape[1] == 1:
            approx = np.squeeze(approx)
        elif approx.shape[1] == 2:
            approx = approx[:, 1]
        else:
            raise ValueError(f'Invalid approx shape: {approx.shape}')

        if min(approx) < 0:
            approx -= min(approx) # make all values positive
        target = target.astype(int)
        thres = self._get_threshold(target, approx)
        pred = (approx > thres).astype(int)
        tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
        se = tp / (tp + fn)
        sp = tn / (tn + fp)

        return se

# from global_objectives import util

def _prepare_labels_logits_weights(labels, logits, weights):
    """Validates labels, logits, and weights.

    Converts inputs to tensors, checks shape compatibility, and casts dtype if
    necessary.

    Args:
      labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
      logits: A `Tensor` with the same shape as `labels`.
      weights: Either `None` or a `Tensor` with shape broadcastable to `logits`.

    Returns:
      labels: Same as `labels` arg after possible conversion to tensor, cast, and
        reshape.
      logits: Same as `logits` arg after possible conversion to tensor and
        reshape.
      weights: Same as `weights` arg after possible conversion, cast, and reshape.
      original_shape: Shape of `labels` and `logits` before reshape.

    Raises:
      ValueError: If `labels` and `logits` do not have the same shape.
    """
    # Convert `labels` and `logits` to Tensors and standardize dtypes.
    logits = tf.convert_to_tensor(logits, name='logits')
    # labels = util.convert_and_cast(labels, 'labels', logits.dtype.base_dtype)
    # weights = util.convert_and_cast(weights, 'weights', logits.dtype.base_dtype)

    try:
        labels.get_shape().merge_with(logits.get_shape())
    except ValueError:
        raise ValueError('logits and labels must have the same shape (%s vs %s)' %
                       (logits.get_shape(), labels.get_shape()))

    original_shape = labels.get_shape().as_list()
    if labels.get_shape().ndims > 0:
        original_shape[0] = -1
    if labels.get_shape().ndims <= 1:
        labels = tf.reshape(labels, [-1, 1])
        logits = tf.reshape(logits, [-1, 1])

    if weights.get_shape().ndims == 1:
        # Weights has shape [batch_size]. Reshape to [batch_size, 1].
        weights = tf.reshape(weights, [-1, 1])
    if weights.get_shape().ndims == 0:
        # Weights is a scalar. Change shape of weights to match logits.
        weights *= tf.ones_like(logits)

    return labels, logits, weights, original_shape

# import tensorflow as tf
def roc_auc_loss(
    labels,
    logits,
    weights=1.0,
    surrogate_type='xent',
    scope=None):
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(labels, logits, weights)

    # Create tensors of pairwise differences for logits and labels, and
    # pairwise products of weights. These have shape
    # [batch_size, batch_size, num_labels].
    logits_difference = tf.expand_dims(logits, 0) - tf.expand_dims(logits, 1)
    labels_difference = tf.expand_dims(labels, 0) - tf.expand_dims(labels, 1)
    weights_product = tf.expand_dims(weights, 0) * tf.expand_dims(weights, 1)

    signed_logits_difference = labels_difference * logits_difference
    raw_loss = util.weighted_surrogate_loss(
        labels=tf.ones_like(signed_logits_difference),
        logits=signed_logits_difference,
        surrogate_type=surrogate_type)
    weighted_loss = weights_product * raw_loss

    # Zero out entries of the loss where labels_difference zero (so loss is only
    # computed on pairs with different labels).
    loss = tf.reduce_mean(tf.abs(labels_difference) * weighted_loss, 0) * 0.5
    loss = tf.reshape(loss, original_shape)
    return loss, {}


class RMSE(MetricTemplate):
    '''
    Root mean square error
    '''
    def _test(self, target, approx):
        return np.sqrt(mean_squared_error(target, approx))

class AUC(MetricTemplate):
    '''
    Area under ROC curve
    '''
    def __init__(self, maximize=True):
        self.maximize = maximize

    def _test(self, target, approx):
        if len(approx.shape) == 1:
            approx = approx
        elif approx.shape[1] == 1:
            approx = np.squeeze(approx)
        elif approx.shape[1] == 2:
            approx = approx[:, 1]
        else:
            raise ValueError(f'Invalid approx shape: {approx.shape}')
        return roc_auc_score(target, approx)

class MultiAUC(MetricTemplate):
    '''
    Area under ROC curve
    '''
    def __init__(self,
                 maximize=True,
                 label_features=['typical', 'indeterminate', 'atypical', 'negative'],
                 use_label_features=['typical', 'indeterminate', 'atypical', 'negative']):
        self.maximize = maximize
        self.label_features = label_features
        self.use_label_features = use_label_features

    def _test(self, target, approx):
        scores = []
        print('-'*20)
        for i, label_f in enumerate(self.label_features):
            # try:
            # st()
            score = roc_auc_score(np.array(target)[:, i], np.array(approx)[:, i])
            print(self.label_features[i], score)
            # except:
            #     pass
            if label_f in self.use_label_features:
                scores.append(score)
        print('-'*20)
        score = np.mean(scores)
        return score

class MultiAP(MetricTemplate):
    '''
    Area under ROC curve
    '''
    def __init__(self,
                 maximize=True,
                 label_features=['typical', 'indeterminate', 'atypical', 'negative', 'none'],
                 use_label_features=['typical', 'indeterminate', 'atypical', 'negative', 'none']):
        self.maximize = maximize
        self.label_features = label_features
        self.use_label_features = use_label_features

    def _test(self, target, approx):
        scores = []
        print('-'*20)
        for i, label_f in enumerate(self.label_features):
            # try:
            # st()
            score = average_precision_score(np.array(target)[:, i], np.array(approx)[:, i])
            print(self.label_features[i], score)
            # except:
            #     pass
            if label_f in self.use_label_features:
                scores.append(score)
        print('-'*20)
        score = np.mean(scores)
        return score

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class AUC_Anno(MetricTemplate):
    '''
    Area under ROC curve
    '''
    def __init__(self, maximize=True):
        self.maximize = maximize

    def _test(self, target, approx):
        return roc_auc_score(target[:, 1], sigmoid(approx[:, 1]))

class Accuracy(MetricTemplate):
    '''
    Accuracy
    '''

    def __init__(self, maximize=True):
        self.maximize = maximize

    def _test(self, target, approx):
        assert(len(target) == len(approx))
        target = np.asarray(target, dtype=int)
        approx = np.asarray(approx, dtype=float)
        if len(approx.shape) == 1:
            approx = approx
        elif approx.shape[1] == 1:
            approx = np.squeeze(approx)
        elif approx.shape[1] >= 2:
            approx = np.argmax(approx, axis=1)
        approx = approx.round().astype(int)
        return np.mean((target == approx).astype(int))

class F1(MetricTemplate):
    __name__ = 'F1 macro'
    def __init__(self):
        pass

    def _test(self, target, approx):
        assert(len(target) == len(approx))
        target = np.asarray(target, dtype=int)
        approx = np.asarray(approx, dtype=float)
        approx = (approx > 0.5).astype(int)

        tp = (approx * target).sum(0)
        fp = (approx > target).sum(0)
        fn = (approx < target).sum(0)

        score = (2.0*tp/(2.0*tp + fp + fn + 1e-6)).mean()
        return score

class MultiLabelAccuracy(MetricTemplate):
    __name__ = 'MultiLabelAccuracy'
    def __init__(self):
        pass

    def _test(self, target, approx):
        assert(len(target) == len(approx))
        target = np.asarray(target, dtype=int)
        approx = np.asarray(approx, dtype=float)
        approx = (approx > 0.5).astype(int)

        tp = (approx * target).sum(0)
        tn = ((approx-1)*(target-1)).sum(0)
        fp = (approx > target).sum(0)
        fn = (approx < target).sum(0)
        score = ((tn+tp)/(tn+fn+fp+tp)).mean()
        return score

class QWK(MetricTemplate):
    '''
    Quandric Weight Kappa :))
    '''

    def __init__(self, max_rat, maximize=True):
        self.max_rat = max_rat
        self.maximize = maximize

    def _test(self, target, approx):
        assert(len(target) == len(approx))
        target = np.asarray(target, dtype=int)
        approx = np.asarray(approx, dtype=float)
        if len(approx.shape) == 1:
            approx = approx
        elif approx.shape[1] == 1:
            approx = np.squeeze(approx)
        elif approx.shape[1] >= 2:
            approx = np.argmax(approx, axis=1)
        approx = np.clip(approx.round(), 0, self.max_rat-1).astype(int)

        hist1 = np.zeros((self.max_rat+1, ))
        hist2 = np.zeros((self.max_rat+1, ))

        o = 0
        for k in range(target.shape[0]):
            i, j = target[k], approx[k]
            hist1[i] += 1
            hist2[j] += 1
            o += (i - j) * (i - j)

        e = 0
        for i in range(self.max_rat + 1):
            for j in range(self.max_rat + 1):
                e += hist1[i] * hist2[j] * (i - j) * (i - j)

        e = e / target.shape[0]

        return 1 - o / e

