import numpy as np
import pandas as pd
import pickle

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data

def pickle_dump(data, path):
    with open(path, mode='wb') as f:
        pickle.dump(data, f)
