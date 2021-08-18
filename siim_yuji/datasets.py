import numpy as np
import pandas as pd
import json
from pathlib import Path
import pickle
from glob import glob
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from pdb import set_trace as st

from kuma_utils.blur_detector import BlurDetector
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def crop_by_mask(im, mask, box_mask=None, pad=100):
    if im.shape[0] != mask.shape[0]:
        mask = cv2.resize(mask.astype(np.uint8), dsize=(im.shape[0], im.shape[1]), interpolation=cv2.INTER_AREA)
        if box_mask is not None:
            box_mask = cv2.resize(box_mask.astype(np.uint8), dsize=(im.shape[0], im.shape[1]), interpolation=cv2.INTER_AREA)

    w=np.sum(mask, axis=0)
    h=np.sum(mask, axis=1)
    w=np.where(w>=1, 1, 0)
    h=np.where(h>=1, 1, 0)
    x_min, y_min = w.argmax()-pad, h.argmax()-pad
    w, h = w.tolist(), h.tolist()
    w.reverse()
    h.reverse()
    x_max, y_max = len(w)-np.argmax(w)+pad, len(h)-np.argmax(h)+pad
    shape=im.shape
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    im = im[np.max([y_min,0]):np.min([y_max,shape[0]]), np.max([x_min,0]):np.min([x_max, shape[1]]), ...]
    if box_mask is None:
        return im
    box_mask = box_mask[np.max([y_min,0]):np.min([y_max,shape[0]]), np.max([x_min,0]):np.min([x_max, shape[1]]), ...]
    return im, box_mask.astype(np.uint8)

class DatasetTrain(Dataset):
    def __init__(self, df, transforms, cfg, split):
        self.paths = df['path'].values
        self.mask_paths = df['mask_path'].values
        self.transforms = transforms
        self.labels = df[cfg.label_features].values
        self.cfg = cfg
        self.im_size_df = pd.read_csv('input/train_with_size.csv')
        self.im_size_df = self.im_size_df.rename(columns={'dim0': 'height', 'dim1': 'width'})
        self.pseudo_box_df = cfg.detection_pred_df
        self.split = split

    def __len__(self):
        return len(self.paths)

    def get_pseudo_mask(self, image_id, conf_th=0.3, use_top_conf=False):
        im_size_df = self.im_size_df[self.im_size_df.image_id == image_id]
        df = self.pseudo_box_df[self.pseudo_box_df['image_id']==image_id]
        height = int(im_size_df.height.values[0])
        width = int(im_size_df.width.values[0])
        if height > width:
            height_width_ratio = width / height
            new_height = 1536
            new_width = int(1536*height_width_ratio)

            box = np.zeros((new_height, new_width)).astype(np.uint8)
            if use_top_conf:
                df = df[df.conf == df.conf.max()]
            else:
                df = df[df.conf>conf_th]
            if len(df) == 0:
                return box.astype(np.uint8)
            x_min = df.x_min.min() * new_width / width
            y_min = df.y_min.min() * new_height / height
            x_max = df.x_max.max() * new_width / width
            y_max = df.y_max.max() * new_height / height
        else:
            height_width_ratio = height / width
            new_width = 1536
            new_height = int(1536*height_width_ratio)

            box = np.zeros((new_height, new_width)).astype(np.uint8)
            if use_top_conf:
                df = df[df.conf == df.conf.max()]
            else:
                df = df[df.conf>conf_th]
            if len(df) == 0:
                return box.astype(np.uint8)
            x_min = df.x_min.min() * new_width / width
            y_min = df.y_min.min() * new_height / height
            x_max = df.x_max.max() * new_width / width
            y_max = df.y_max.max() * new_height / height


        if self.cfg.ellipse:
            x = int(x_min)
            y = int(y_min)
            w = int(x_max-x_min)
            h = int(y_max-y_min)
            box = cv2.ellipse(box, (x+w//2, y+h//2), (w//2, h//2), 0,0,360, color=(1,1,1), thickness=-1)
        else:
            box[int(y_min):int(y_max), int(x_min):int(x_max)] = 1
        return box.astype(np.uint8)

    def __getitem__(self, idx):
        path = self.paths[idx]
        path = path.replace('train_1024', 'train_png_ratio')

        mask_path = self.mask_paths[idx]

        if self.cfg.aux_criterion is None:
            image = cv2.imread(path)
            if self.cfg.crop_by_mask:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                image, _ = crop_by_mask(image, mask, mask, pad=self.cfg.crop_pad)

            if self.transforms:
                image = self.transforms(image=image)['image']
            if image.shape[2] < 10:
                image = image.transpose(2, 0, 1)

            label = self.labels[idx]
            return image, torch.FloatTensor(label)

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image_id = path.split('/')[-1].replace('.jpg', '').replace('.png', '')

        image = cv2.resize(image, dsize=(1536, 1536), interpolation=cv2.INTER_AREA)

        if self.cfg.crop_by_mask:
            if self.cfg.box_mask_v1:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                box_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                image, box_mask = crop_by_mask(image, mask, box_mask, pad=self.cfg.crop_pad)
            elif self.cfg.box_mask_v3:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                mask = cv2.resize(mask, dsize=(1536, 1536), interpolation=cv2.INTER_AREA).astype(np.uint8)
                image_id = path.split('/')[-1].replace('.jpg', '').replace('.png', '')
                box_mask = self.get_pseudo_mask(image_id, self.cfg.conf_th)
                box_mask = cv2.resize(box_mask, dsize=(1536, 1536), interpolation=cv2.INTER_AREA)

                box_mask = (mask|box_mask).astype(np.uint8)
                del mask

                image, box_mask = crop_by_mask(image, box_mask, box_mask, pad=self.cfg.crop_pad)

        image = cv2.resize(image.astype(np.uint8), dsize=(self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_AREA)
        box_mask = cv2.resize(box_mask.astype(np.uint8), dsize=(self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_AREA)
        if self.transforms:
            image = self.transforms(image=image, mask=box_mask)
            box_mask = image[1]
            image = image[0]

        label = self.labels[idx]

        return image, box_mask, torch.FloatTensor(label)

class DatasetTest(Dataset):
    def __init__(self, df, transforms, cfg):
        self.paths = df.path.values
        self.mask_paths = df.mask_path.values
        self.transforms = transforms
        self.cfg = cfg

    def __len__(self):
        return len(self.paths)

    def get_pseudo_mask(self, image_id, conf_th=0.3, use_top_conf=False):
        im_size_df = self.im_size_df[self.im_size_df.image_id == image_id]
        df = self.pseudo_box_df[self.pseudo_box_df['image_id']==image_id]
        box = np.zeros((int(im_size_df.height.values[0]), int(im_size_df.width.values[0])))
        if use_top_conf:
            df = df[df.conf == df.conf.max()]
        else:
            df = df[df.conf>conf_th]
        if len(df) == 0:
            return box.astype(np.uint8)
        x_min = df.x_min.min()
        y_min = df.y_min.min()
        x_max = df.x_max.max()
        y_max = df.y_max.max()

        if self.cfg.ellipse:
            x = int(x_min)
            y = int(y_min)
            w = int(x_max-x_min)
            h = int(y_max-y_min)
            box = cv2.ellipse(box, (x+w//2, y+h//2), (w//2, h//2), 0,0,360, color=(1,1,1), thickness=-1)
        else:
            box[int(y_min):int(y_max), int(x_min):int(x_max)] = 1
        return box.astype(np.uint8)

    def __getitem__(self, idx):
        path = self.paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.cfg.crop_by_mask:
            if self.cfg.box_mask_v1:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                image = crop_by_mask(image, mask, pad=self.cfg.crop_pad)

            elif self.cfg.box_mask_v3:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                image_id = path.split('/')[-1].replace('.jpg', '').replace('.png', '')
                box_mask = self.get_pseudo_mask(image_id, self.cfg.conf_th)
                box_mask = (mask|box_mask)
                box_mask = np.where(box_mask>0, 1, 0)

                image, _ = crop_by_mask(image, box_mask, box_mask, pad=self.cfg.crop_pad)

        image = cv2.resize(image, dsize=(self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_AREA)
        return image, idx
