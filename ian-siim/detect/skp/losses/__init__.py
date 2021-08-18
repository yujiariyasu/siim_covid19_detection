from torch.nn import (
    CrossEntropyLoss,
    BCEWithLogitsLoss,
    MSELoss,
    L1Loss,
    SmoothL1Loss
)

from ..effdet.loss import DetectionLoss
from .custom import *
