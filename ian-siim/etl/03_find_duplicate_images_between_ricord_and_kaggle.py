import glob
import imagehash
import numpy as np 
import os.path as osp
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm


def get_hashes(imgfiles, resize=(1024,1024)):
    imgids, hashes = [], []
    for imfi in tqdm(imgfiles, total=len(imgfiles)):
        imgids.append(imfi.split('/')[-1])
        img = Image.open(imfi).resize(resize)
        hashes.append(np.array([f(img).hash for f in FUNCS]).reshape(256))
    return hashes


def find_duplicates(hash1, hash2, imgs1, thresh=0.95):
    sims = np.array([np.where((hash2[i] == hash1).sum(dim=1).numpy()/256 > thresh) for i in tqdm(range(hash2.shape[0]), total=len(hash2))])
    lengths = np.asarray([len(i[0]) for i in sims])
    indices = np.unique(np.concatenate([i[0] for i in sims]))
    dupes = np.asarray(imgs1)[indices]
    return sims, dupes


def write_list_of_filenames(lst, filename):
    with open(filename, 'w') as f:
        for element in lst:
            _ = f.write(f'{element}\n')


FUNCS = [
    imagehash.average_hash,
    imagehash.phash,
    imagehash.dhash,
    imagehash.whash,
]


ricord_img = glob.glob('../data/ricord-kaggle/images/MIDRC-RICORD/*.jpg')
kaggle_train_img = glob.glob('../data/covid/train_pngs/*/*/*.png')
kaggle_test_img = glob.glob('../data/covid/test_pngs/*/*/*.png')

ricord_hash = get_hashes(ricord_img)
kaggle_train_hash = get_hashes(kaggle_train_img)
kaggle_test_hash = get_hashes(kaggle_test_img)

np.save('../data/ricord_hash.npy', ricord_hash)
np.save('../data/kaggle_train_hash.npy', kaggle_train_hash)
np.save('../data/kaggle_test_hash.npy', kaggle_test_hash)

ricord_hash = torch.Tensor(np.asarray(ricord_hash).astype(int))
kaggle_train_hash = torch.Tensor(np.asarray(kaggle_train_hash).astype(int))
kaggle_test_hash = torch.Tensor(np.asarray(kaggle_test_hash).astype(int))

train_sims, train_dupes = find_duplicates(ricord_hash, kaggle_train_hash, ricord_img, 0.95)
test_sims, test_dupes = find_duplicates(ricord_hash, kaggle_test_hash, ricord_img, 0.95)

write_list_of_filenames(train_dupes, '../data/ricord_in_kaggle_train.txt')
write_list_of_filenames(test_dupes, '../data/ricord_in_kaggle_test.txt')

