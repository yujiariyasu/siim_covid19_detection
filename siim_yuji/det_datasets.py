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

class DetDatasetTrain(Dataset):
    def __init__(self, df, transforms=None):
        super().__init__()

        self.df = df
        self.transforms = transforms
        self.image_ids = df.image_id.unique()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        image, boxes, labels, image_size = self.load_image_and_boxes(image_id)

        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.tensor(labels)
        target['image_id'] = torch.tensor([index])

        if self.transforms is not None:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                    break

        # print(image, target, image_size, image_id)
        return image, target, image_size, image_id

    def load_image_and_boxes(self, image_id):
        records = self.df[self.df['image_id'] == image_id]
        path = records.path.values[0]
        image = cv2.imread(path, cv2.IMREAD_COLOR).copy()
        image_size = image.shape[:2]
        boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
        labels = records['class_id'].values
        return image, boxes, labels, image_size

class DetDatasetTest(Dataset):
    def __init__(self, df, transforms=None):
        super().__init__()
        self.transforms = transforms
        self.paths = df.path.unique()

    def __getitem__(self, index: int):
        path = self.paths[index]
        image_id = path.split('/')[-1].replace('.jpg', '')
        image = cv2.imread(path, cv2.IMREAD_COLOR).copy()
        image_size = image.shape[:2]
        if self.transforms:
            image = self.transforms(image=image)['image']
        return image, image_size, image_id

    def __len__(self) -> int:
        return len(self.paths)
