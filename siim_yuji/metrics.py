import numpy as np
import torch
from functools import partial
import scipy as sp


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1/(1 + np.exp(-x))
