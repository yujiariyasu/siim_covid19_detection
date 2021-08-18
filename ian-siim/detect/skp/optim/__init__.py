# Optimizers
from torch.optim import (
    Adam,
    AdamW,
    SGD,
    RMSprop
)

from .radam import RAdam
from .madgrad import MADGRAD

# Schedulers
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)

from .onecycle import CustomOneCycleLR
