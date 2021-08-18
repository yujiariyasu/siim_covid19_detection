import albumentations as A
import ast
import cv2
import random
import numpy as np
import os.path as osp
import pydicom
import torch

from torch.utils import data
from pydicom.pixel_data_handlers.util import apply_voi_lut


NONETYPE = type(None)


def load_dicom(fp, fake_rgb=False):
    dcm = pydicom.dcmread(fp)
    try:
        arr = apply_voi_lut(dcm.pixel_array, dcm)
    except Exception as e:
        print(e)
        arr = dcm.pixel_array.astype('float32')
    arr = arr - np.min(arr)
    arr = arr / np.max(arr)
    arr = arr * 255.0
    arr = arr.astype('uint8')
    if dcm.PhotometricInterpretation != 'MONOCHROME2':
        arr = np.invert(arr)
    arr = np.expand_dims(arr, axis=-1)
    if fake_rgb: arr = np.repeat(arr, 3, axis=-1)
    return arr


class ImageDataset(data.Dataset):

    def __init__(self,
                 inputs,
                 labels,
                 resize=None,
                 augment=None,
                 crop=None,
                 preprocess=None,
                 flip=False,
                 verbose=True,
                 test_mode=False,
                 return_name=False,
                 return_imsize=False,
                 invert=False,
                 add_invert_label=False,
                 **kwargs):
        self.inputs = inputs
        self.labels = labels
        self.resize = resize
        self.augment = augment
        self.crop = crop 
        self.preprocess = preprocess
        self.flip = flip
        self.verbose = verbose
        self.test_mode = test_mode
        self.return_name = return_name
        self.return_imsize = return_imsize
        self.invert = invert
        self.add_invert_label = add_invert_label

    def __len__(self): return len(self.inputs)

    def process_image(self, X):
        if self.resize: X = self.resize(image=X)['image']
        if self.augment: X = self.augment(image=X)['image']
        if self.crop: X = self.crop(image=X)['image']
        if self.invert: X = np.invert(X)
        if self.preprocess: X = self.preprocess(X)
        return X.transpose(2, 0, 1)

    @staticmethod
    def flip_array(X):
        # X.shape = (C, H, W)
        if random.random() > 0.5:
            X = X[:, :, ::-1]
        if random.random() > 0.5:
            X = X[:, ::-1, :]
        if random.random() > 0.5 and X.shape[-1] == X.shape[-2]:
            X = X.transpose(0, 2, 1)
        X = np.ascontiguousarray(X)
        return X

    def get(self, i):
        try:
            X = cv2.imread(self.inputs[i])
            return X
        except Exception as e:
            if self.verbose: print(e)
            return None

    def __getitem__(self, i):
        X = self.get(i)
        while isinstance(X, NONETYPE):
            if self.verbose: print('Failed to read {} !'.format(self.inputs[i]))
            i = np.random.randint(len(self))
            X = self.get(i)

        imsize = X.shape[:2]
        
        if self.add_invert_label:
            inverted = False
            if np.random.binomial(1, 0.5):
                X = 255 - X
                inverted = True

        X = self.process_image(X)

        if self.flip and not self.test_mode:
            X = self.flip_array(X)

        y = self.labels[i]

        if self.add_invert_label: 
            if inverted:
                y = np.concatenate([y, np.asarray([1])])
            else:
                y = np.concatenate([y, np.asarray([0])])

        X = torch.tensor(X).float()
        y = torch.tensor(y)

        out = [X, y]
        if self.return_name:
            out.append(self.inputs[i])
        if self.return_imsize:
            out.append(imsize)

        return tuple(out)


class DICOMDataset(ImageDataset):

    def __init__(self, *args, **kwargs):
        self.repeat_rgb = kwargs.pop('repeat_rgb', True)
        super().__init__(*args, **kwargs)

    def get(self, i):
        try:
            X = load_dicom(self.inputs[i], fake_rgb=self.repeat_rgb)
            return X
        except Exception as e:
            if self.verbose: print(e)
            return None


class SegmentClassify(ImageDataset):

    def __init__(self, *args, **kwargs):
        self.bboxes = kwargs.pop('bboxes', [])
        self.use_dicom = kwargs.pop('dicom', False)
        self.multiclass_seg = kwargs.pop('multiclass_seg', False)
        super().__init__(*args, **kwargs)

    def process_image(self, X, y):
        if self.resize: 
            X = self.resize(image=X)['image']
            y = self.resize(image=y)['image']
        if self.augment: 
            augmented = self.augment(image=X, mask=y)
            X, y = augmented['image'], augmented['mask']
        if self.crop: 
            cropped = self.crop(image=X, mask=y)
            X, y = cropped['image'], cropped['mask']
        if self.invert: X = np.invert(X)
        if self.preprocess: X = self.preprocess(X)
        return X.transpose(2, 0, 1), y.transpose(2, 0, 1)

    @staticmethod
    def flip_array(X, y):
        # X.shape = (C, H, W)
        if random.random() > 0.5:
            X = X[:, :, ::-1]
            y = y[:, :, ::-1]
        if random.random() > 0.5:
            X = X[:, ::-1, :]
            y = y[:, ::-1, :]
        if random.random() > 0.5 and X.shape[-1] == X.shape[-2]:
            X = X.transpose(0, 2, 1)
            y = y.transpose(0, 2, 1)
        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)
        return X, y

    def get(self, i):
        try:
            if self.use_dicom: 
                X = load_dicom(self.inputs[i], fake_rgb=True)
            else:
                X = cv2.imread(self.inputs[i])
            return X
        except Exception as e:
            if self.verbose: print(e)
            return None

    def draw_ellipse_mask(self, mask, boxes):
        for b in boxes:
            x, y, w, h = [int(_) for _ in b]
            mask = cv2.ellipse(mask, (x+w//2, y+h//2), (w//2, h//2), 0,0,360, color=(1,1,1), thickness=-1)
        return mask

    def __getitem__(self, i):
        X = self.get(i)
        while isinstance(X, NONETYPE):
            if self.verbose: print('Failed to read {} !'.format(self.inputs[i]))
            i = np.random.randint(len(self))
            X = self.get(i)

        y_seg = np.zeros((X.shape[0], X.shape[1], 1)).astype('uint8')
        pseudo = False
        if str(self.bboxes[i]) != 'nan':
            if 'jpg' in str(self.bboxes[i]) or 'png' in str(self.bboxes[i]):
                y_seg = cv2.imread(self.bboxes[i], 0)
                y_seg = np.expand_dims(y_seg, axis=-1)
                pseudo = True
            else:
                y_seg = self.draw_ellipse_mask(y_seg, ast.literal_eval(self.bboxes[i]))

        imsize = X.shape[:2]
        X, y_seg = self.process_image(X, y_seg)
        if pseudo:
            y_seg = y_seg / 255.

        if self.flip and not self.test_mode:
            X, y_seg = self.flip_array(X, y_seg)

        X = torch.tensor(X).float()
        y = torch.tensor(self.labels[i])
        if self.multiclass_seg: 
            _y_seg = np.zeros_like(y_seg)
            _y_seg = np.repeat(_y_seg, 4, axis=0)
            lab = np.argmax(self.labels[i][:-1])
            if lab != 0:
                _y_seg[lab-1] = y_seg[0]
                _y_seg[3] = y_seg[0]
            y_seg = _y_seg

        y_seg = torch.tensor(y_seg).float()

        out = [X, (y, y_seg)]
        if self.return_name:
            out.append(self.inputs[i])
        if self.return_imsize:
            out.append(imsize)

        return tuple(out) 

