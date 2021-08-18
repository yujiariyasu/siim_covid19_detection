
import os
import re
import sys
import time
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from copy import copy, deepcopy

import numpy as np
import pandas as pd
from pdb import set_trace as st

import torch
import torch.utils.data as D
from .snapshot import *
from .logger import *
from .temperature_scaling import *
from .fp16util import network_to_half
from scipy.special import softmax

try:
    from torchsummary import summary
except ModuleNotFoundError:
    print('torch summary not found.')

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ModuleNotFoundError:
    print('torch_xla not found.')

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    APEX_FLAG = True
except:
    print('nvidia apex not found.')
    APEX_FLAG = False

'''
Stopper

# Methods
__call__(score) : bool  = return whether score is improved
stop() : bool           = return whether to stop training or not
state() : int, int      = return current / total
score() : float         = return best score
freeze()                = update score but never stop
unfreeze()              = unset freeze()
'''

class DummyStopper:
    ''' No stopper '''

    def __init__(self):
        pass

    def __call__(self, val_loss):
        return True

    def stop(self):
        return False

    def state(self):
        return 0, 0

    def score(self):
        return 0.0

    def dump_state_dict(self):
        return {}

    def load_state_dict(self, checkpoint):
        pass

    def __repr__(self):
        return 'No Stopper'


class EarlyStopping(DummyStopper):
    '''
    Early stops the training if validation loss doesn't improve after a given patience.
    patience: int   = early stopping rounds
    maximize: bool  = whether maximize or minimize metric
    '''

    def __init__(self, patience=5, maximize=False):
        self.patience = patience
        self.counter = 0
        self.log = []
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        if maximize:
            self.coef = 1
        else:
            self.coef = -1
        self.frozen = False

    def __call__(self, val_loss):
        score = self.coef * val_loss
        self.log.append(score)
        if score is None:
            return False
        elif self.best_score is None:
            self.best_score = score
            return True
        elif score <= self.best_score:
            if not self.frozen:
                self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else: # score improved
            self.best_score = score
            self.counter = 0
            return True

    def stop(self):
        return self.early_stop

    def state(self):
        return self.counter, self.patience

    def score(self):
        return self.best_score

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def reset(self):
        self.best_score = None

    def dump_state_dict(self):
        return {
            'best_score': self.best_score,
            'counter': self.counter,
        }

    def load_state_dict(self, checkpoint):
        self.best_score = checkpoint['best_score']
        self.counter = checkpoint['counter']

    def __repr__(self):
        return f'EarlyStopping({self.patience})'


'''
Event
'''

class DummyEvent:
    ''' Dummy event does nothing '''

    def __init__(self):
        pass

    def __call__(self, **kwargs):
        pass

    def dump_state_dict(self):
        return {}

    def load_state_dict(self, checkpoint):
        pass

    def __repr__(self):
        return 'No Event'


class NoEarlyStoppingNEpochs(DummyEvent):

    def __init__(self, n):
        self.n = n

    def __call__(self, **kwargs):
        if kwargs['global_epoch'] == 0:
            kwargs['stopper'].freeze()
            kwargs['stopper'].reset()
            print(f"Epoch\t{kwargs['epoch']}: Earlystopping is frozen.")
        elif kwargs['global_epoch'] < self.n:
             kwargs['stopper'].reset()
        elif kwargs['global_epoch'] == self.n:
            kwargs['stopper'].unfreeze()
            print(f"Epoch\t{kwargs['epoch']}: Earlystopping is unfrozen.")

    def __repr__(self):
        return f'NoEarlyStoppingNEpochs({self.n})'

def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def choice_mix_image(batch_inputs, i):
    mix_idxes = list(range(len(batch_inputs[0])))
    mix_idxes.remove(i)
    mix_idx = random.choice(mix_idxes)
    return batch_inputs[0][mix_idx], batch_inputs[1][mix_idx]

def mixup_for_bag(inputs):
    images = []
    labels = []
    for i, (image1, label1) in enumerate(zip(inputs[0], inputs[1])):
        image2, label2 = choice_mix_image(inputs, i)

        lam = np.random.beta(0.5, 0.5)
        img = (lam*image1 + (1-lam)*image2)
        img = np.clip(img, 0, 1)
        img = np.float32(img)
        label = lam*label1+(1-lam)*label2
        label = np.clip(label, 0, 1)
        images.append(img)
        labels.append(label)
    return [torch.from_numpy(np.array(images).astype(np.float32)), torch.from_numpy(np.array(labels).astype(np.float32))]

'''
Trainer
'''
def cutmix_for_bag(inputs, alpha=1.0):
    data, target = inputs
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    new_data = data.clone()
    bbx1, bby1, bbx2, bby2 = rand_bbox(data[i].size(), lam)
    new_data[:, :, :, bby1:bby2, bbx1:bbx2] = shuffled_data[:, :, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets

class TorchTrainer:
    '''
    Simple Trainer for PyTorch models

    # Usage
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-3)
    NN_FIT_PARAMS = {
        'loader': loader_train,
        'loader_valid': loader_valid,
        'loader_test': loader_test,
        'criterion': nn.BCEWithLogitsLoss(),
        'optimizer': optimizer,
        'scheduler': StepLR(optimizer, step_size=10, gamma=0.9),
        'num_epochs': 100,
        'stopper': EarlyStopping(patience=20, maximize=True),
        'logger': Logger('results/test/'),
        'snapshot_path': Path('results/test/nn_best.pt'),
        'eval_metric': auc,
        'info_format': '[epoch] time data loss metric earlystopping',
        'info_train': False,
        'info_interval': 3
    }
    trainer = TorchTrainer(model, serial='test')
    trainer.fit(**NN_FIT_PARAMS)
    '''

    def __init__(self,
                 model, device=None, serial='Trainer',
                 fp16=False, xla=False):

        if device is None:
            if xla:
                device = xm.xla_device()
            else:
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.serial = serial
        self.is_fp16 = fp16 # Automatically use apex if available
        self.is_xla = xla
        self.apex_opt_level = 'O1'
        self.model = model
        self.all_inputs_to_model = False
        print(f'[{self.serial}] On {self.device}.')

    def model_to_fp16(self):
        if APEX_FLAG:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer,
                opt_level=self.apex_opt_level, verbosity=0)
            print(f'[{self.serial}] Model, Optimizer -> fp16 (apex)')
        else:
            self.model = network_to_half(self.model)
            print(f'[{self.serial}] Model -> fp16 (simple)')

    def model_to_parallel(self):
        if self.is_xla:
            print(
                f'[{self.serial}] Multi parallel training for xla devices is WIP.')

        if torch.cuda.device_count() > 1:
            all_devices = list(range(torch.cuda.device_count()))
            if self.is_fp16 and APEX_FLAG:
                self.model = nn.parallel.DataParallel(self.model)
            else:
                self.model = nn.parallel.DataParallel(self.model)

            print(f'[{self.serial}] {torch.cuda.device_count()}({all_devices}) gpus found.')


    def train_loop(self, loader, grad_accumulations=1, logger_interval=1, epoch=0):
        loss_total = 0.0
        # metric_total = 0.0
        total_batch = len(loader.dataset) / loader.batch_size
        approx = []
        target = []

        if 'OUSM' in str(self.criterion):
            self.criterion.update(epoch)

        self.model.train()
        # self.wandb.watch(self.model, self.criterion, log="all", log_freq=2)
        with tqdm(loader, total=len(loader), leave=False) as pbar:
            for batch_i, (images, y) in enumerate(pbar):
                batches_done = len(loader) * self.current_epoch + batch_i
                images, y = images.to(self.device), y.to(self.device)
                if self.train_transform is not None:
                    images = self.train_transform(images)

                mix = False
                if self.cfg.ian_mixup_aug is not None:
                    images, y = self.mixup_aug(images, y, self.cfg.self.mixup_params)

                elif self.mixup_aug is not None:
                    mix_decision = np.random.rand()
                    if mix_decision < 0.25:
                        images, labels_a, labels_b, lam = self.mixup_aug(images, y, mix_decision)
                        mix = True
                # elif self.cutmix_aug is not None:
                #     mix_decision = np.random.rand()
                #     if mix_decision < 0.25:
                #         inputs = cutmix_for_bag(inputs)
                # if (self.mixup_aug in not None) or (self.cutmix_aug in not None):
                #     raise
                elif self.resizemix_aug is not None:
                    images, labels_a, labels_b, lam = self.resizemix_aug(images, y, alpha=0.1, beta=0.6)
                    mix = True

                _y = self.model(images)
                if isinstance(_y, tuple):
                    _y_aux = _y[1]
                    _y = _y[0]
                else:
                    _y_aux = None
                if self.is_fp16:
                    if isinstance(_y, tuple):
                        _y = _y.float()
                        _y_aux = _y_aux.float()
                    else:
                        _y = _y.float()

                if mix:
                    loss = self.criterion(_y, labels_a) * lam + self.criterion(_y, labels_b) * (1 - lam)
                    if _y_aux is not None:
                        loss_aux = self.criterion(_y_aux, labels_a) * lam + self.criterion(_y_aux, labels_b) * (1 - lam)
                        loss = loss + 0.4*loss_aux

                else:
                    # st()
                    loss = self.criterion(_y, y)
                    # if _y_aux is not None:
                    #     loss_aux = self.criterion(_y_aux, y)
                    #     loss = loss + 0.4*loss_aux

                loss = loss / grad_accumulations # normalize loss
                # if _y.size()[1] in [1, 2]:
                #     approx.append(_y.clone().detach())
                # else:
                #     approx.append(_y.sigmoid().clone().detach())
                approx.append(_y.clone().detach())
                    # approx.append(softmax(_y.clone().detach(), axis=1))
                target.append(y.clone().detach())

                # if self.mixup or self.cutmix:
                #     _y = self.softmax(_y)[:, 1]
                #     _y = torch.squeeze(_y)
                #     y = y.float()
                #     # _y = torch.float(_y)

                if self.is_fp16 and APEX_FLAG and (str(self.device) != 'cpu'):
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                pbar.set_description(f'{loss.item()}')

                if batch_i == 0:
                    # Save output dimension in the first run
                    self.out_dim = _y.shape[1:]

                if ((batch_i + 1) % grad_accumulations == 0) or ((batch_i == len(loader)-1) & ((batch_i + 1) % grad_accumulations > (grad_accumulations/2))):
                    # Accumulates gradient before each step
                    if self.is_xla:
                        xm.optimizer_step(self.optimizer, barrier=True)
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                if batch_i % logger_interval == 0:
                    # metric = self.eval_metric(_y, y)
                    for param_group in self.optimizer.param_groups:
                        learning_rate = param_group['lr']
                    log_train_batch = [
                        # (f'batch_metric_train[{self.serial}]', metric),
                        (f'batch_loss_train[{self.serial}]', loss.item()),
                        (f'batch_lr_train[{self.serial}]', learning_rate)
                    ]
                    self.logger.list_of_scalars_summary(log_train_batch, batches_done)

                batch_weight = len(images) / loader.batch_size
                loss_total += loss.item() / total_batch * batch_weight
                # metric_total += metric / total_batch * batch_weight

        approx = torch.cat(approx).cpu()
        target = torch.cat(target).cpu()
        if (self.eval_metric is None) or (self.train_skip_eval):
            metric_total = -loss_total
        else:
            metric_total = self.eval_metric(approx, target)
        log_metrics_total = []
        for log_metric in self.log_metrics:
            log_metrics_total.append(log_metric(approx, target))

        log_train = [
            (f'epoch_metric_train[{self.serial}]', metric_total),
            (f'epoch_loss_train[{self.serial}]', loss_total)
        ]
        self.logger.list_of_scalars_summary(log_train, self.current_epoch)
        self.log['train']['loss'].append(loss_total)
        self.log['train']['metric'].append(metric_total)

        return loss_total, metric_total, log_metrics_total







        return None, tta_predictions

    def valid_loop(self, loader, grad_accumulations=1, logger_interval=1, epoch=0, tta=1):
        loss_total = 0.0
        total_batch = len(loader.dataset) / loader.batch_size
        approx = []
        target = []

        self.model.eval()
        with torch.no_grad():
            tta_predictions = []
            for tta_n in tqdm(range(tta)):
                predictions = []

                for images, y in loader:
                    images, y = images.to(self.device), y.to(self.device)
                    if self.val_transform is not None:
                        images = self.val_transform(images)
                    if tta_n % 2 == 1:
                        images = torch.flip(images, (3,))
                    if tta_n % 4 >= 2:
                        images = torch.flip(images, (2,))
                    if tta_n % 8 >= 4:
                        images = torch.transpose(images, 2,3)

                    if self.is_fp16 and APEX_FLAG and (str(self.device) != 'cpu'):
                        with amp.disable_casts():
                            _y = self.model(images)
                    else:
                        _y = self.model(images)

                    if isinstance(_y, tuple):
                        # _y_aux = _y[1]
                        _y = _y[0]
                    predictions.append(_y.detach())
                    if tta_n==0:
                        target.append(y.clone().detach())
                tta_predictions.append(torch.cat(predictions).cpu().numpy())
                # if _y.size()[1] in [1, 2]:
                #     approx.append(_y.clone().detach())
                # else:
                    # approx.append(softmax(_y.clone().detach(), axis=1))
                #     approx.append(_y.sigmoid().clone().detach())


        approx = np.mean(tta_predictions, axis=0)
        target = torch.cat(target).cpu()
        # st()
        loss_total = self.criterion(torch.tensor(approx), torch.tensor(target))
        if self.eval_metric is None:
            metric_total = -loss_total
        else:
            if 'none' in self.label_features:
                approx[:, :len(self.label_features)-1] = softmax(approx[:, :len(self.label_features)-1], axis=1)
            else:
                approx = softmax(approx, axis=1)
            metric_total = self.eval_metric(approx, target)
        log_metrics_total = []
        for log_metric in self.log_metrics:
            log_metrics_total.append(log_metric(approx, target))

        if tta == 1:
            log_valid = [
                (f'epoch_metric_valid[{self.serial}]', metric_total),
                (f'epoch_loss_valid[{self.serial}]', loss_total)
            ]
            self.log['valid']['loss'].append(loss_total)
            self.log['valid']['metric'].append(metric_total)
        else:
            log_valid = [
                (f'epoch_metric_valid_tta[{self.serial}]', metric_total),
                (f'epoch_loss_valid_tta[{self.serial}]', loss_total)
            ]
            self.log['valid']['loss'].append(loss_total)
            self.log['valid']['metric'].append(metric_total)
        self.logger.list_of_scalars_summary(log_valid, self.current_epoch)

        return loss_total, metric_total, log_metrics_total

    def predict(self, loader, path=None, verbose=True):
        prediction = []
        features_list = []

        self.model.eval()

        with torch.no_grad():
            with tqdm(loader, total=len(loader), leave=False) as pbar:
                for inputs in pbar:

                # for inputs in tqdm(loader):
                    images = inputs[0].to(self.device)
                    if self.val_transform is not None:
                        images = self.val_transform(images)
                    if self.is_fp16 and APEX_FLAG and (str(self.device) != 'cpu'):
                        with amp.disable_casts():
                            _y = self.model(images)
                    else:
                        _y = self.model(images)
                    # features_list.append(features.detach())
                    prediction.append(_y.detach())
                    # prediction.append(_y.sigmoid().detach())

        prediction = torch.cat(prediction).cpu().numpy()
        # features = torch.cat(features_list).cpu().numpy()
        return None, prediction

    def predict_with_tta(self, loader, path=None, tta=1, verbose=True):
        self.model.eval()
        with torch.no_grad():
            tta_predictions = []
            for tta_n in tqdm(range(tta)):
                predictions = []
                for inputs in loader:
                    images = inputs[0].to(self.device)
                    if self.val_transform is not None:
                        images = self.val_transform(images)
                    if tta_n % 2 == 1:
                        images = torch.flip(images, (3,))
                    if tta_n % 4 >= 2:
                        images = torch.flip(images, (2,))
                    if tta_n % 8 >= 4:
                        images = torch.transpose(images, 2,3)

                    if self.is_fp16 and APEX_FLAG and (str(self.device) != 'cpu'):
                        with amp.disable_casts():
                            _y = self.model(images)
                    else:
                        _y = self.model(images)
                    # features_list.append(features.detach())
                    predictions.append(_y.detach())
                    # predictions.append(_y.sigmoid().detach())
                tta_predictions.append(torch.cat(predictions).cpu().numpy())

        return None, tta_predictions

    def print_info(self, info_items, info_seps, info):
        log_str = ''
        for sep, item in zip(info_seps, info_items):
            if item == 'time':
                current_time = time.strftime('%H:%M:%S', time.gmtime())
                log_str += current_time
            elif item == 'data':
                log_str += info[item]
            elif item in ['loss', 'metric']:
                log_str += f'{item}={info[item]:.{self.round_float}f}'
            elif item  == 'logmetrics':
                if len(info[item]) > 0:
                    for im, m in enumerate(info[item]): # list
                        log_str += f'{item}{im}={m:.{self.round_float}f}'
                        if im != len(info[item]) - 1:
                            log_str += ' '
            elif item == 'epoch':
                align = len(str(self.max_epochs))
                log_str += f'E{self.current_epoch:0{align}d}/{self.max_epochs}'
            elif item == 'earlystopping':
                if info['data'] == 'Trn':
                    continue
                counter, patience = self.stopper.state()
                best = self.stopper.score()
                if best is not None:
                    log_str += f'best={best:.{self.round_float}f}'
                    if counter > 0:
                        log_str += f'*({counter}/{patience})'
            log_str += sep
        if len(log_str) > 0:
            print(f'[{self.serial}] {log_str}')

    def freeze(self):
        if isinstance(self.model, torch.nn.DataParallel):
            children = self.model.module.children()
        else:
            children = self.model.children()

        for child in children:
            if (('LayerNorm' not in str(type(child))) & ('AdaptiveAvgPool1d' not in str(type(child))) & ('Linear' not in str(type(child)))):
                for param in child.parameters():
                    param.requires_grad = False
        print('freeze exclude fc layer.')

    def unfreeze(self):
        if isinstance(self.model, torch.nn.DataParallel):
            children = self.model.module.children()
        else:
            children = self.model.children()

        for child in children:
            for param in child.parameters():
                param.requires_grad = True
        print('unfreeze all layer.')

    def fit(self, optimizer, scheduler,
            loader, loader_valid=None, loader_test=None, loader_valid_tta=None, fine_tune_loader=None,
            loader_test_tta=None, snapshot_path=None, multi_gpu=True, calibrate_model=False,
            eval_interval=1, logger=DummyLogger(''), logger_interval=1,
            info_train=True, info_valid=True, info_interval=1, round_float=6,
            info_format='epoch time data loss metric logmetrics earlystopping',
            verbose=True, wandb=None, cfg=None
        ):

        self.criterion = cfg.criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.eval_metric = cfg.metric
        self.log_metrics = cfg.log_metrics
        self.logger = logger
        self.event = deepcopy(cfg.event)
        self.stopper = deepcopy(cfg.stopper)
        self.current_epoch = 0
        self.best_epoch = 0
        self.mixup_aug = cfg.mixup_aug
        self.cutmix_aug = cfg.cutmix_aug
        self.fmix_aug = cfg.fmix_aug
        self.resizemix_aug = cfg.resizemix_aug
        self.train_transform = cfg.transform['batch_train']
        self.val_transform = cfg.transform['batch_val'] if cfg.tta==1 else cfg.transform['batch_tta']
        self.train_skip_eval = cfg.train_skip_eval
        self.wandb = wandb
        self.label_features = cfg.label_features
        self.aux_criterion = cfg.aux_criterion
        self.cfg = cfg
        self.log = {
            'train': {'loss': [], 'metric': []},
            'valid': {'loss': [], 'metric': []}
        }
        info_items = re.split(r'[^a-z]+', info_format)
        info_seps = re.split(r'[a-z]+', info_format)
        self.round_float = round_float

        if snapshot_path is None:
            snapshot_path = Path().cwd()
        if not isinstance(snapshot_path, Path):
            snapshot_path = Path(snapshot_path)
        if len(snapshot_path.suffix) > 0: # Is file
            self.root_path = snapshot_path.parent
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        else: # Is dir
            self.root_path = snapshot_path
            snapshot_path = snapshot_path/'snapshot.pt'
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        if not isinstance(self.log_metrics, (list, tuple, set)):
            self.log_metrics = [self.log_metrics]

        # self.model.to(self.rank)
        self.model.to(self.device)
        if cfg.resume:
            load_snapshots_to_model(
                snapshot_path, self.model, self.optimizer, self.scheduler,
                self.stopper, self.event, device=self.device)
            self.current_epoch = load_epoch(snapshot_path)
            self.best_epoch = self.current_epoch
            if verbose:
                print(
                    f'[{self.serial}] {snapshot_path} is loaded. Continuing from epoch {self.current_epoch}.')
        if self.is_fp16 & (str(self.device) != 'cpu'):
            self.model_to_fp16()
        if multi_gpu:
            self.model_to_parallel()

        self.max_epochs = self.current_epoch + cfg.epochs
        loss_valid, metric_valid = np.inf, -np.inf

        if self.cfg.freeze or self.cfg.unfreeze:
            self.freeze()

        for epoch in range(self.cfg.epochs):
            if (epoch == 1) and (self.cfg.unfreeze):
                self.unfreeze()
            #     print('before lr:', self.optimizer.param_groups[0]['lr'])
                self.optimizer.param_groups[0]['lr'] /= 4
            #     print('after lr:', self.optimizer.param_groups[0]['lr'])

            if self.cfg.freeze or self.cfg.unfreeze:
                if epoch in [0, 1]:
                    pos = 0
                    neg = 0
                    if isinstance(self.model, torch.nn.DataParallel):
                        children = self.model.module.children()
                    else:
                        children = self.model.children()
                    for child in children:
                        for param in child.parameters():
                            if param.requires_grad:
                                pos+=1
                            else:
                                neg+=1
                    print(f'epoch: {epoch}, unfreezed params: {pos}, freezed params: {neg}')


            self.current_epoch += 1
            start_time = time.time()
            if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                self.scheduler.step(loss_valid)
            else:
                self.scheduler.step()

            ### Event
            self.event(**{'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler,
                     'stopper': self.stopper, 'criterion': self.criterion, 'eval_metric': self.eval_metric,
                     'epoch': epoch, 'global_epoch': self.current_epoch, 'log': self.log})

            ### Training
            loss_train, metric_train, log_metrics_train = self.train_loop(loader, cfg.grad_accumulations, logger_interval, epoch)

            if info_train and epoch % info_interval == 0:
                self.print_info(info_items, info_seps, {
                    'data': 'Trn',
                    'loss': loss_train,
                    'metric': metric_train,
                    'logmetrics': log_metrics_train
                })

            ### Validation
            if cfg.train_by_all_data:
                early_stopping_target = metric_train
                if self.stopper(early_stopping_target):
                    save_snapshots(snapshot_path,
                                   self.current_epoch, self.model,
                                   self.optimizer, self.scheduler, self.stopper, self.event)
                continue

            if epoch % eval_interval == 0:
                loss_valid, metric_valid, log_metrics_valid = self.valid_loop(loader_valid, cfg.grad_accumulations, logger_interval, epoch)
                if cfg.tta > 1:
                    loss_valid_tta, metric_valid_tta, log_metrics_valid_tta = self.valid_loop(
                        loader_valid_tta, cfg.grad_accumulations, logger_interval, epoch, tta=cfg.tta)

                if cfg.tta > 1:
                    early_stopping_target = metric_valid_tta
                else:
                    early_stopping_target = metric_valid
                if self.wandb is not None:
                    self.wandb.log({
                        'loss_train': loss_train,
                        'metric_train': metric_train,
                        'loss_valid': loss_valid,
                        'metric_valid': metric_valid,
                    })

                if self.stopper(early_stopping_target):  # score improved
                    save_snapshots(snapshot_path,
                                   self.current_epoch, self.model,
                                   self.optimizer, self.scheduler, self.stopper, self.event)

                    # artifact = self.wandb.Artifact(name='model',
                    #                           type='moel')
                    # artifact.add_file(str(snapshot_path))
                    # self.wandb.log_artifact(artifact)

                if info_valid and epoch % info_interval == 0:
                    self.print_info(info_items, info_seps, {
                        'data': 'Val',
                        'loss': loss_valid,
                        'metric': metric_valid,
                        'logmetrics': log_metrics_valid
                    })
                    if cfg.tta > 1:
                        self.print_info(info_items, info_seps, {
                            'data': 'Val_tta',
                            'loss': loss_valid_tta,
                            'metric': metric_valid_tta,
                            'logmetrics': log_metrics_valid_tta
                        })

            # Stopped by overfit detector
            if self.stopper.stop():
                if verbose:
                    print("[{}] Training stopped by overfit detector. ({}/{})".format(
                        self.serial, self.current_epoch-self.stopper.state()[1]+1, self.max_epochs))
                break

        if verbose & (not cfg.train_by_all_data):
            print(
                f"[{self.serial}] Best score is {self.stopper.score():.{self.round_float}f}")
        load_snapshots_to_model(str(snapshot_path), self.model, self.optimizer)

        if calibrate_model:
            print('calibrate...')
            self.calibrate_model(loader_valid)

        if cfg.predict_valid:
            if cfg.tta > 1:
                self.tta_oof_features, self.tta_oof = self.predict_with_tta(
                    loader_valid_tta, tta=cfg.tta, verbose=verbose)
            else:
                self.oof_features, self.oof = self.predict(loader_valid, verbose=verbose)
        if cfg.predict_test:
            if cfg.tta > 1:
                self.tta_pred_features, self.tta_pred = self.predict_with_tta(
                    loader_test_tta, tta=cfg.tta, verbose=verbose)
            else:
                self.pred_features, self.pred = self.predict(loader_test, verbose=verbose)


    def calibrate_model(self, loader):
        self.model = TemperatureScaler(self.model).to(self.device)
        self.model.set_temperature(loader)



