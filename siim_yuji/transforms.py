import albumentations as A
import torchvision.transforms as T

from albumentations.pytorch import ToTensor, ToTensorV2
import cv2

def image_clasification_medical0(size):
    transform {
        'train': A.Compose([
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
            A.Resize(self.image_size, self.image_size),
            ToTensor(),
        ],
        p=1),
        "test": A.Compose([A.Resize(self.image_size, self.image_size), ToTensor()],
            p=1),
        "tta": A.Compose([A.Resize(self.image_size, self.image_size), ToTensor()],
            p=1),
        "fine_tune": A.Compose([A.Resize(self.image_size, self.image_size), ToTensor()],
            p=1),
    }

def met_transform1(size):
    size = list(map(int, size.split(',')))
    transform = {
        'albu_train': A.Compose([
            A.RandomResizedCrop(size[0], size[1])
        ]),
        'torch_train': T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomAffine(15, translate=(0.1, 0.1)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ]),
        'albu_val': A.Compose([
            A.Resize(size[0], size[1])
        ]),
        'torch_val': T.Compose([
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ])
    }
    return transform

