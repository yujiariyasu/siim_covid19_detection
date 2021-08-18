import torch
import torchvision.transforms as T
import albumentations as A
import cv2
from albumentations.pytorch import ToTensor, ToTensorV2
import numpy as np
import random
from .keroppi import *

mean = (0.485, 0.456, 0.406)  # RGB
std = (0.229, 0.224, 0.225)  # RGB


def base_aug_v1(size):
    return {
        'dataset_train': A.Compose([
            A.Resize(size, size)
        ]),
        'dataset_tta': A.Compose([
            A.Resize(size, size)
        ]),
        'batch_train': T.Compose([
            T.RandomHorizontalFlip(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ]),
        'batch_tta': T.Compose([
            T.RandomHorizontalFlip(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ]),
        'dataset_val': A.Compose([
            A.Resize(size, size)
        ]),
        'batch_val': T.Compose([
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ])
    }

def base_aug_v2(size):
    return {
        'dataset_train': A.Compose([
            A.Resize(size, size)
        ]),
        'dataset_tta': A.Compose([
            A.Resize(size, size)
        ]),
        'dataset_val': A.Compose([
            A.Resize(size, size)
        ]),
        'batch_train': T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ]),
        'batch_tta': T.Compose([
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ]),
        'batch_val': T.Compose([
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ])
    }

def base_aug_v1_4ch(size):
    mean = (0.485, 0.456, 0.406, 0.406)  # RGBY
    std = (0.229, 0.224, 0.225, 0.225)  # RGBY
    return {
        'dataset_train': A.Compose([
            A.Resize(size, size)
        ]),
        'dataset_tta': A.Compose([
            A.Resize(size, size)
        ]),
        'batch_train': T.Compose([
            T.RandomHorizontalFlip(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ]),
        'batch_tta': T.Compose([
            T.RandomHorizontalFlip(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ]),
        'dataset_val': A.Compose([
            A.Resize(size, size)
        ]),
        'batch_val': T.Compose([
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ])
    }

def base_aug_4ch_random_crop(size):
    mean = (0.485, 0.456, 0.406, 0.406)  # RGBY
    std = (0.229, 0.224, 0.225, 0.225)  # RGBY
    return {
        'dataset_train': A.Compose([
            A.Resize(int(size*1.5), int(size*1.5)),
            A.RandomCrop(size, size)
        ]),
        'dataset_tta': A.Compose([
            A.Resize(int(size*1.5), int(size*1.5)),
            A.RandomCrop(size, size)
        ]),
        'batch_train': T.Compose([
            T.RandomHorizontalFlip(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ]),
        'batch_tta': T.Compose([
            # T.RandomHorizontalFlip(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ]),
        'dataset_val': A.Compose([
            A.Resize(int(size*1.5), int(size*1.5)),
            A.RandomCrop(size, size)
        ]),
        'batch_val': T.Compose([
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ])
    }

def base_aug_v3(size):
    return {
        'dataset_train': A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9,
                               border_mode=cv2.BORDER_CONSTANT),
            A.Resize(size, size),
            ToTensor(),
        ],
        p=1),
        "dataset_val": A.Compose([A.Resize(size, size), ToTensor()],
            p=1),
        "dataset_tta": A.Compose([A.Resize(size, size), ToTensor()],
            p=1),
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def image_clasification_medical_v1(size):
    return {
        'dataset_train': A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9,
                               border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([#off in most cases
                A.MotionBlur(blur_limit=3, p=0.1),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([#off in most cases
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.3),
            A.Resize(size, size),
            ToTensor(),
        ],
        p=1),
        "dataset_val": A.Compose([A.Resize(size, size), ToTensor()],
            p=1),
        "dataset_tta": A.Compose([A.Resize(size, size), ToTensor()],
            p=1),
        # "dataset_tta": A.Compose([
        #     A.HorizontalFlip(),
        #     A.VerticalFlip(),
        #     A.RandomRotate90(),
        #     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9,
        #                        border_mode=cv2.BORDER_CONSTANT),
        #     A.OneOf([#off in most cases
        #         A.MotionBlur(blur_limit=3, p=0.1),
        #         A.MedianBlur(blur_limit=3, p=0.1),
        #         A.Blur(blur_limit=3, p=0.1),
        #     ], p=0.2),
        #     A.OneOf([#off in most cases
        #         A.OpticalDistortion(p=0.3),
        #         A.GridDistortion(p=.1),
        #         A.IAAPiecewiseAffine(p=0.3),
        #     ], p=0.3),
        #     A.Resize(size, size),
        #     ToTensor(),
        # ],
        # p=1),
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }


def image_clasification_medical_v2(size):
    return {
        'dataset_train': A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9,
                               border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([#off in most cases
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.3),
            A.Resize(size, size),
            ToTensor(),
        ],
        p=1),
        "dataset_val": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        "dataset_tta": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def ian_grayscale_augment(p, n):
    augs = A.OneOf([
        A.RandomGamma(),
        A.RandomContrast(),
        A.RandomBrightness(),
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0, rotate_limit=0),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.15, rotate_limit=0),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0, rotate_limit=30),
        A.GaussianBlur(),
        A.IAAAdditiveGaussianNoise()
    ], p=1)
    return A.Compose([A.VerticalFlip()]+[augs]*n, p=p)

def grayscale_augment(p, n):
    return {
        'dataset_train': ian_grayscale_augment(p, n),
        "dataset_val": None,
        "dataset_tta": None,
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }


def image_clasification_medical_v2_resize_first(size):
    return {
        'dataset_train': A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9,
                               border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([#off in most cases
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.3),
            ToTensor(),
        ],
        p=1),
        "dataset_val": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        "dataset_tta": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def image_clasification_medical_v2_resize_first_with_mask():
    return {
        'dataset_train': A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9,
                               border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([#off in most cases
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.3),
        ]),
        "dataset_val": None,
        "dataset_tta": None,
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def fine_tune_v1(size):
    return {
        'dataset_train': A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            ToTensor(),
        ],
        p=1),
        "dataset_val": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        "dataset_tta": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def with_mask_v1(image, mask, image_size=384):
    if 1:
        for fn in np.random.choice([
            lambda image, mask : do_random_scale(image, mask, mag=0.20),
            lambda image, mask : do_random_stretch_y(image, mask, mag=0.20),
            lambda image, mask : do_random_stretch_x(image, mask, mag=0.20),
            lambda image, mask : do_random_shift(image, mask, mag=int(0.20*image_size)),
            lambda image, mask : (image, mask)
        ],1):
            image, mask = fn(image, mask)

        for fn in np.random.choice([
            # lambda image, mask : do_random_rotate(image, mask, mag=15),
            lambda image, mask : do_random_hflip(image, mask),
            # lambda image, mask : (image, mask)
        ],1):
            image, mask = fn(image, mask)

        for fn in np.random.choice([
            lambda image, mask : do_random_wflip(image, mask),
            # lambda image, mask : (image, mask)
        ],1):
            image, mask = fn(image, mask)

        for fn in np.random.choice([
            lambda image, mask : do_random_rotate(image, mask, mag=15),
            lambda image, mask : (image, mask),
            lambda image, mask : (image, mask)
        ],1):
            image, mask = fn(image, mask)

        # ------------------------
        for fn in np.random.choice([
            lambda image : do_random_intensity_shift_contast(image, mag=[0.5,0.5]),
            lambda image : do_random_noise(image, mag=0.05),
            lambda image : do_random_guassian_blur(image),
            lambda image : do_random_blurout(image, size=0.25, num_cut=2),
            #lambda image : do_random_clahe(image),
            #lambda image : do_histogram_norm(image),
            lambda image : image,
        ],1):
            image = fn(image)

def with_mask_v2(image, mask, image_size=384):
    for fn in np.random.choice([
        lambda image, mask : do_random_scale(image, mask, mag=0.20),
        lambda image, mask : do_random_stretch_y(image, mask, mag=0.20),
        lambda image, mask : do_random_stretch_x(image, mask, mag=0.20),
        lambda image, mask : do_random_shift(image, mask, mag=int(0.20*image_size)),
        lambda image, mask : (image, mask)
    ],1):
        image, mask = fn(image, mask)

    for fn in np.random.choice([
        lambda image, mask : do_random_rotate(image, mask, mag=15),
        lambda image, mask : do_random_hflip(image, mask),
        lambda image, mask : (image, mask)
    ],1):
        image, mask = fn(image, mask)

    # ------------------------
    for fn in np.random.choice([
        lambda image : do_random_intensity_shift_contast(image, mag=[0.5,0.5]),
        lambda image : do_random_noise(image, mag=0.05),
        lambda image : do_random_guassian_blur(image),
        lambda image : do_random_blurout(image, size=0.25, num_cut=2),
        #lambda image : do_random_clahe(image),
        #lambda image : do_histogram_norm(image),
        lambda image : image,
    ],1):
        image = fn(image)

    return image, mask

def with_mask_v3(image, mask, image_size=384):
    for fn in np.random.choice([
        lambda image, mask : do_random_rotate(image, mask, mag=15),
        lambda image, mask : do_random_hflip(image, mask),
        lambda image, mask : (image, mask)
    ],1):
        image, mask = fn(image, mask)

    # ------------------------
    for fn in np.random.choice([
        lambda image : do_random_intensity_shift_contast(image, mag=[0.5,0.5]),
        lambda image : do_random_noise(image, mag=0.05),
        lambda image : do_random_guassian_blur(image),
        lambda image : do_random_blurout(image, size=0.25, num_cut=2),
        #lambda image : do_random_clahe(image),
        #lambda image : do_histogram_norm(image),
        lambda image : image,
    ],1):
        image = fn(image)

    return image, mask

def with_mask_aug_v2():
    return {
        'dataset_train': with_mask_v2,
        "dataset_val": None,
        "dataset_tta": None,
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def with_mask_aug_v3():
    return {
        'dataset_train': with_mask_v3,
        "dataset_val": None,
        "dataset_tta": None,
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def with_mask_light(image, mask, image_size=384):
    for fn in np.random.choice([
        lambda image, mask : do_random_hflip(image, mask),
    ],1):
        image, mask = fn(image, mask)

    return image, mask

def with_mask_aug_light():
    return {
        'dataset_train': with_mask_light,
        "dataset_val": None,
        "dataset_tta": None,
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def image_clasification_medical_v2_random_crop(size):
    return {
        'dataset_train': A.Compose([
            A.RandomResizedCrop(size, size, scale=(0.9, 1), p=1), 
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9,
                               border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([#off in most cases
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.3),
            A.Resize(size, size),
            ToTensor(),
        ],
        p=1),
        "dataset_val": A.Compose([A.RandomResizedCrop(size, size, scale=(0.9, 1), p=1), A.Resize(size, size), ToTensor()], p=1),
        "dataset_tta": A.Compose([A.RandomResizedCrop(size, size, scale=(0.9, 1), p=1), A.Resize(size, size), ToTensor()], p=1),
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def image_clasification_medical_v3(size):
    return {
        'dataset_train': A.Compose([
             A.HorizontalFlip(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.Resize(size, size),
             A.OneOf([
                 A.JpegCompression(),
                 A.Downscale(scale_min=0.1, scale_max=0.15),
             ], p=0.2),
             A.IAAPiecewiseAffine(p=0.2),
             A.IAASharpen(p=0.2),
             A.Cutout(max_h_size=int(size * 0.1), max_w_size=int(size * 0.1), num_holes=5, p=0.5),
             ToTensor(),
        ]),
        "dataset_val": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        "dataset_tta": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def image_clasification_medical_v4(size):
    return {
        'dataset_train': A.Compose([
             A.RandomResizedCrop(size, size, scale=(0.9, 1), p=1), 
             A.HorizontalFlip(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.Resize(size, size),
             A.OneOf([
                 A.JpegCompression(),
                 A.Downscale(scale_min=0.1, scale_max=0.15),
             ], p=0.2),
             A.IAAPiecewiseAffine(p=0.2),
             A.IAASharpen(p=0.2),
             A.Cutout(max_h_size=int(size * 0.1), max_w_size=int(size * 0.1), num_holes=5, p=0.5),
             ToTensor(),
        ]),
        "dataset_val": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        "dataset_tta": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def image_clasification_medical_v5(size):
    return {
        'dataset_train': A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Transpose(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9,
                               border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([#off in most cases
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.3),
            A.Resize(size, size),
            ToTensor(),
        ],
        p=1),
        "dataset_val": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        "dataset_tta": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def image_clasification_medical_v6(size):
    return {
        'dataset_train': A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Transpose(),
            A.RandomRotate90(),
            A.OneOf([#off in most cases
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.3),
            A.Resize(size, size),
            ToTensor(),
        ],
        p=1),
        "dataset_val": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        "dataset_tta": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def image_clasification_medical_v7(size):
    return {
        'dataset_train': A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Transpose(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9,
                               border_mode=cv2.BORDER_CONSTANT),
            A.Resize(size, size),
            ToTensor(),
        ],
        p=1),
        "dataset_val": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        "dataset_tta": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def image_clasification_medical_v8(size):
    return {
        'dataset_train': A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Transpose(),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9,
                               border_mode=cv2.BORDER_CONSTANT),
            A.Resize(size, size),
            ToTensor(),
        ],
        p=1),
        "dataset_val": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        "dataset_tta": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def image_clasification_medical_v9(size):
    return {
        'dataset_train': A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Transpose(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9,
                               border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([
                A.JpegCompression(),
                A.Downscale(scale_min=0.1, scale_max=0.15),
            ], p=0.2),
            A.Resize(size, size),
            ToTensor(),
        ],
        p=1),
        "dataset_val": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        "dataset_tta": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

             # A.OneOf([
             #     A.OpticalDistortion(distort_limit=1.0),
             #     A.GridDistortion(num_steps=5, distort_limit=1.),
             #     A.ElasticTransform(alpha=3),
             # ], p=0.2),
             # A.IAAPiecewiseAffine(p=0.2),
             # A.IAASharpen(p=0.2),
             # A.Cutout(max_h_size=int(size * 0.1), max_w_size=int(size * 0.1), num_holes=5, p=0.5),


def image_clasification_medical_v10(size):
    return {
        'dataset_train': A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Transpose(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9,
                               border_mode=cv2.BORDER_CONSTANT),
            A.CLAHE(clip_limit=(1,4), p=0.5),
            A.OneOf([
                A.JpegCompression(),
                A.Downscale(scale_min=0.1, scale_max=0.15),
            ], p=0.2),
            A.Resize(size, size),
            ToTensor(),
        ],
        p=1),
        "dataset_val": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        "dataset_tta": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def image_clasification_medical_v11(size):
    return {
        'dataset_train': A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9,
                               border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([#off in most cases
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.JpegCompression(),
                A.Downscale(scale_min=0.1, scale_max=0.15),
            ], p=0.2),
            A.Resize(size, size),
            ToTensor(),
        ],
        p=1),
        "dataset_val": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        "dataset_tta": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def image_clasification_medical_v13(size):
    return {
        'dataset_train': A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.OneOf([#off in most cases
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.JpegCompression(),
                A.Downscale(scale_min=0.1, scale_max=0.15),
            ], p=0.2),
            A.Resize(size, size),
            ToTensor(),
        ],
        p=1),
        "dataset_val": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        "dataset_tta": A.Compose([A.Resize(size, size), ToTensor()], p=1),
        'batch_tta': None,
        'batch_train': None,
        'batch_val': None,
    }

def det_aug_base_v1(size):
    return {
        'train': A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.ShiftScaleRotate(),
            A.Resize(size,size,always_apply=True),
            ToTensor(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "val": A.Compose([
            A.Resize(height=size, width=size, p=1.0),
            ToTensor(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            A.Resize(height=size, width=size, p=1.0),
            ToTensor(),
        ], p=1.0)}

def det_aug_base_v2(size):
    return {
        'train': A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.ShiftScaleRotate(),
            A.OneOf([
                A.RandomContrast(),
                A.RandomGamma(),
                A.RandomBrightness(),
                ], p=0.3),
            A.CLAHE(clip_limit=2),
            A.Resize(size,size,always_apply=True),
            ToTensor(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "val": A.Compose([
            A.Resize(height=size, width=size, p=1.0),
            ToTensor(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            A.Resize(height=size, width=size, p=1.0),
            ToTensor(),
        ], p=1.0)}

def det_aug_base_v3(size):
    return {
        'train': A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9,
                               border_mode=cv2.BORDER_CONSTANT),
            # A.OneOf([#off in most cases
                # A.OpticalDistortion(p=0.3),
            #     A.GridDistortion(p=.1),
            #     A.IAAPiecewiseAffine(p=0.3),
            # ], p=0.3),
            ToTensor(),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "val": A.Compose([
            A.Resize(height=size, width=size, p=1.0),
            ToTensor(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )),
        "test": A.Compose([
            A.Resize(height=size, width=size, p=1.0),
            ToTensor(),
        ], p=1.0)}

