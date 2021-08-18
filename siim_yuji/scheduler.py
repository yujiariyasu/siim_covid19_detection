import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from kuma_utils.nn.training import DummyEvent


class MyScheduler(_LRScheduler):

    def __init__(self, optimizer, config={10: 0.5, 20: 0.5, 30: 0.1}, last_epoch=-1):
            self.config = config
            super(MyScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        old_lr = [group['lr'] for group in self.optimizer.param_groups]
        if not self.last_epoch in self.config.keys():
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            new_lr = [group['lr'] * self.config[self.last_epoch]
                      for group in self.optimizer.param_groups]
            print(f'learning rate -> {new_lr}')
            return new_lr


class MyEvent(DummyEvent):

    def __init__(self, stopper=10):
        self.stopper = stopper

    def __call__(self, **kwargs):
        # Earlystopping control
        if self.stopper:
            if kwargs['global_epoch'] == 1:
                kwargs['stopper'].freeze()
                kwargs['stopper'].reset()
                print(f"Epoch\t{kwargs['global_epoch']}: Earlystopping is frozen.")

            if kwargs['global_epoch'] < self.stopper:
                kwargs['stopper'].reset()

            if kwargs['global_epoch'] == self.stopper:
                kwargs['stopper'].unfreeze()
                print(f"Epoch\t{kwargs['global_epoch']}: Earlystopping is unfrozen.")

    def __repr__(self):
        return f'Unfreeze(stopper={self.stopper})'
