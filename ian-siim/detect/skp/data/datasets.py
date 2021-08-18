import albumentations as A
import numpy as np
import cv2
import pydicom
import torch

from torch.utils import data


FLIP = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ], p=1, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

TRANSPOSE = A.Compose([
        A.Transpose(p=0.5),
    ], p=1, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

NONETYPE = type(None)


class CXRDataset(data.Dataset):

    def __init__(self,
                 annotations,
                 resize=None,
                 augment=None,
                 crop=None,
                 preprocess=None,
                 flip=False,
                 dicom=False,
                 verbose=False,
                 test_mode=False,
                 return_name=False,
                 return_label=True,
                 return_class_label=False,
                 return_imsize=False):
        self.annotations = annotations
        # annotations is a list of dicts:
        # [
        #   {
        #     filename: <full path to file>,
        #     bbox: np.ndarray(N, 4), 
        #       (x1, y1, x2, y2)
        #     cls: np.ndarray(N, ),
        #     (Optional)
        #     img_width:  int,
        #     img_height: int
        #   },
        #   ...
        # ]
        self.resize = resize
        self.augment = augment
        self.crop = crop
        self.preprocess = preprocess
        self.dicom = dicom
        self.flip = flip
        self.verbose = verbose
        self.test_mode = test_mode
        self.return_name = return_name
        self.return_label = return_label
        self.return_class_label = return_class_label
        self.return_imsize = return_imsize

        self.length = len(self.annotations)

    def __len__(self): return self.length

    def calculate_scale_factor(self, orig_shape):
        r = self.resize[0]
        if isinstance(r, A.LongestMaxSize):
            scale_factor = [r.max_size / max(orig_shape)] * 2
        elif isinstance(r, A.Resize):
            scale_factor = [r.width / orig_shape[1], r.height / orig_shape[0]]
        return tuple(scale_factor)

    @staticmethod
    def apply_op(op, img, ann):
        if op: 
            try:
                result = op(image=img, bboxes=list(ann['bbox']), labels=list(ann['cls']))
                ann['bbox'] = np.asarray(result['bboxes'])
                ann['cls'] = np.asarray(result['labels'])
                return result['image'], ann
            except ValueError as e:
                # Sometimes there are errors where augmented bbox is slightly out of range
                # In these cases just forgo augmentation
                print(e)
                print(ann)
                return img, ann
        return img, ann

    def flip_array(self, img, ann):
        img, ann = self.apply_op(FLIP, img, ann)
        if img.shape[0] == img.shape[1]:
            img, ann = self.apply_op(TRANSPOSE, img, ann)
        return img, ann

    def process_image(self, img, ann, skip_augment=False):
        ann['img_scale'] = self.calculate_scale_factor(img.shape)
        img, ann = self.apply_op(self.resize, img, ann)
        if not self.test_mode and not skip_augment: 
            img, ann = self.apply_op(self.augment, img, ann)
        img, ann = self.apply_op(self.crop, img, ann)
        img = self.preprocess(img)
        return img, ann

    def load_image(self, fp):
        if self.dicom:
            array = pydicom.dcmread(fp).pixel_array
            array = array - np.min(array)
            array = array / np.max(array)
            array = (array * 255).astype('uint8')
            array = np.repeat(np.expand_dims(array, axis=-1), 3, axis=-1)
        else:
            array = cv2.imread(fp)
        return array

    def get(self, i):
        try:
            array = self.load_image(self.annotations[i]['filename'])
            ann = self.annotations[i].copy()
            ann['img_size'] = (ann.pop('img_width', array.shape[1]), ann.pop('img_height', array.shape[0]))
            return array, ann
        except Exception as e:
            if self.verbose: print(e)
            return None

    def __getitem__(self, i):
        data = self.get(i)
        while type(data) == NONETYPE:
            i = np.random.randint(len(self))
            data = self.get(i)

        img, ann = data
        imsize = img.shape

        boxes_before = len(ann['bbox']) > 0
        img, ann = self.process_image(img, ann)
        if self.flip and not self.test_mode:
            img, ann = self.flip_array(img, ann)

        # If augmentation causes image to have 0 bounding boxes, 
        # start over and try again (w/o augmentation, to be safe)
        while len(ann['bbox']) == 0 and boxes_before:
            img, ann = self.get(i)
            imsize = img.shape
            img, ann = self.process_image(img, ann, skip_augment=True)
            if self.flip and not self.test_mode:
                img, ann = self.flip_array(img, ann)

        # If image has no boxes, fill with -1 to match EffDet
        if np.sum(ann['cls']) == 0:
            ann['bbox'] = np.zeros((1,4)) - 1
            ann['cls'] = np.zeros((1,)) - 1

        img = img.transpose(2, 0, 1)
        # loader.fast_collate will convert to torch.tensor
        # img = torch.tensor(img).float()
        # ann['bbox'] = torch.tensor(ann['bbox']).float()
        # ann['cls'] = torch.tensor(ann['cls']).long()
        _ = ann.pop('filename')
        _ = ann.pop('folds')

        out = [img]
        class_label = ann.pop('class_label', None)

        if self.return_label:
            out.append(ann)
        if self.return_class_label:
            out.append(class_label)
        if self.return_name:
            out.append(self.annotations[i]['filename'])
        if self.return_imsize:
            out.append(imsize)

        return tuple(out)


class DICOMDataset(CXRDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dicom = True


class ConcatDataset(CXRDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.test_mode:
            self.positives = [a for a in self.annotations if len(a['bbox']) > 0]
            self.negatives = [a for a in self.annotations if len(a['bbox']) == 0]
            self.length = len(self.positives)

    def get(self, i):
        try:
            if self.test_mode:
                array = self.load_image(self.annotations[i]['filename'])
                ann = self.annotations[i].copy()
                ann['img_size'] = (ann.pop('img_width', array.shape[1]), ann.pop('img_height', array.shape[0]))
                return array, ann
            arr1, ann1 = self.load_image(self.positives[i]['filename']), self.positives[i].copy()
            ann1['img_size'] = (ann1.pop('img_width', arr1.shape[1]), ann1.pop('img_height', arr1.shape[0]))
            # Then, randomly sample from positive/negative with 50% chance
            pos = np.random.binomial(1, 0.5)
            if pos:
                j = np.random.randint(len(self.positives))
                arr2, ann2 = self.load_image(self.positives[j]['filename']), self.positives[j].copy()
            else:
                j = np.random.randint(len(self.negatives))
                arr2, ann2 = self.load_image(self.negatives[j]['filename']), self.negatives[j].copy()
            ann2['img_size'] = (ann2.pop('img_width', arr2.shape[1]), ann2.pop('img_height', arr2.shape[0]))
            return arr1, ann1, arr2, ann2
        except Exception as e:
            if self.verbose: print(e)
            return None

    def prepare_one_image(self, img, ann):
        boxes_before = len(ann['bbox']) > 0
        img, ann = self.process_image(img, ann)
        if self.flip and not self.test_mode:
            img, ann = self.flip_array(img, ann)

        # If augmentation causes image to have 0 bounding boxes, 
        # start over and try again (w/o augmentation, to be safe)
        while len(ann['bbox']) == 0 and boxes_before:
            img, ann = self.get(i)
            img, ann = self.process_image(img, ann, skip_augment=True)
            if self.flip and not self.test_mode:
                img, ann = self.flip_array(img, ann)

        if ann['bbox'].shape == (0,):
            # albumentations will mess up dimensions of images w/o bbox
            ann['bbox'] = np.zeros((0,4))
        return img, ann

    def concatenate_images_and_annotations(self, img_lt, ann_lt, img_rt, ann_rt):
        img = np.concatenate([img_lt, img_rt], axis=1)
        if len(ann_rt['bbox']) > 0:
            ann_rt['bbox'][:,[1,3]] = ann_rt['bbox'][:,[1,3]] + img_lt.shape[1]
        ann = ann_lt.copy()
        ann['bbox'] = np.concatenate([ann_lt['bbox'], ann_rt['bbox']])
        ann['cls'] = np.concatenate([ann_lt['cls'], ann_rt['cls']])
        print(img.shape)
        return img, ann

    def __getitem__(self, i):
        data = self.get(i)
        while type(data) == NONETYPE:
            i = np.random.randint(len(self))
            data = self.get(i)

        if not self.test_mode:
            img1, ann1, img2, ann2 = data
            img1, ann1 = self.prepare_one_image(img1, ann1)
            img2, ann2 = self.prepare_one_image(img2, ann2)

            if np.random.binomial(1, 0.5):
                img, ann = self.concatenate_images_and_annotations(img1, ann1, img2, ann2)
            else:
                img, ann = self.concatenate_images_and_annotations(img2, ann2, img1, ann1)

        else:
            img, ann = data
            img, ann = self.prepare_one_image(img, ann)
            img = np.concatenate([img, np.zeros_like(img)], axis=1)

        # If image has no boxes, fill with -1 to match EffDet
        if np.sum(ann['cls']) == 0:
            ann['bbox'] = np.zeros((1,4)) - 1
            ann['cls'] = np.zeros((1,)) - 1

        img = img.transpose(2, 0, 1)
        # loader.fast_collate will convert to torch.tensor
        # img = torch.tensor(img).float()
        # ann['bbox'] = torch.tensor(ann['bbox']).float()
        # ann['cls'] = torch.tensor(ann['cls']).long()
        _ = ann.pop('filename')
        _ = ann.pop('folds')

        return img, ann
