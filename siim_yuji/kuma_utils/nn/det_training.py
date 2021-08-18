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


def choice_mix_image(batch_inputs, i):
    mix_idxes = list(range(len(batch_inputs[0])))
    mix_idxes.remove(i)
    mix_idx = random.choice(mix_idxes)
    return batch_inputs[0][mix_idx], batch_inputs[1][mix_idx]

def mixup(inputs):
    images = []
    labels = []
    for i, (image1, label1) in enumerate(zip(inputs[0], inputs[1])):
        image2, label2 = choice_mix_image(inputs, i)

        lam = np.random.beta(0.5, 0.5)
        img = (lam*image1 + (1-lam)*image2)
        img = np.clip(img, 0, 255)
        img = np.uint8(img)
        label = lam*label1+(1-lam)*label2
        label = np.clip(label, 0, 1)
        images.append(img)
        labels.append(label)
    return [torch.from_numpy(np.array(images).astype(np.float32)), torch.from_numpy(np.array(labels).astype(np.float32))]

'''
Trainer
'''

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
        targets = []
        target_boxes = []
        approx_boxes = []
        self.model.train()
        with tqdm(loader, total=len(loader), leave=False) as pbar:
            for batch_i, (images, target_maps, _, _) in enumerate(pbar):
                if (batch_i == 10) & (self.debug):
                    break
                batches_done = len(loader) * self.current_epoch + batch_i
                images = torch.stack(images)
                images = images.to(self.device).float()
                batch_size = images.shape[0]
                if self.effdet:
                    boxes = [target['boxes'].to(self.device).float() for target in target_maps]
                    labels = [target['labels'].to(self.device).float() for target in target_maps]
                    losses, _, _= self.model(images, boxes, labels)
                else:
                # faster rcnn
                    targets = [{k: v.to(self.device) for k, v in tgt.items()} for tgt in target_maps]
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                loss = torch.mean(losses)
                if self.is_fp16:
                    _y = _y.float()
                    _y_box = _y_box.float()
                    # features = features.float()
                # approx.append(_y.clone().detach())
                # # targets.append(labels.clone().detach())
                # targets.append(labels)
                # approx_boxes.append(_y_box.clone().detach())
                # # target_boxes.append(boxes.clone().detach())
                # target_boxes.append(boxes)

                # loss = self.criterion(_y_box, boxes)

                if self.is_fp16 and APEX_FLAG:
                    with amp.scale_loss(losses, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    losses.backward()

                pbar.set_description(f'{loss.item()}')

                # if batch_i == 0:
                #     # Save output dimension in the first run
                #     self.out_dim = _y.shape[1:]

                if (batch_i + 1) % grad_accumulations == 0:
                    # Accumulates gradient before each step
                    loss = loss / grad_accumulations # normalize loss
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

        # approx = torch.cat(approx).cpu()
        # targets = torch.cat(targets).cpu()
        metric_total = -loss_total
        # log_metrics_total = []
        # for log_metric in self.log_metrics:
        #     log_metrics_total.append(log_metric(approx, targets))

        log_train = [
            (f'epoch_metric_train[{self.serial}]', metric_total),
            (f'epoch_loss_train[{self.serial}]', loss_total)
        ]
        self.logger.list_of_scalars_summary(log_train, self.current_epoch)
        self.log['train']['loss'].append(loss_total)
        self.log['train']['metric'].append(metric_total)

        return loss_total, metric_total#, log_metrics_total

    def valid_loop(self, loader, grad_accumulations=1, logger_interval=1, epoch=0, tta=1):
        loss_total = 0.0
        total_batch = len(loader.dataset) / loader.batch_size
        approx = []
        targets = []

        with torch.no_grad():
            for batch_i, (images, target_maps, _, _) in enumerate(loader):
                images = torch.stack(images)
                images = images.to(self.device).float()
                batch_size = images.shape[0]

                # effdet
                if self.effdet:
                    boxes = [target['boxes'].to(self.device).float() for target in target_maps]
                    labels = [target['labels'].to(self.device).float() for target in target_maps]
                    losses, _, _= self.model(images, boxes, labels)
                else:
                # faster rcnn
                    targets = [{k: v.to(self.device) for k, v in tgt.items()} for tgt in target_maps]
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                loss = torch.mean(losses)

                batch_weight = len(images) / loader.batch_size
                loss_total += loss.item() / total_batch * batch_weight

                print(f"\r{loss.item()}", end='')
                sys.stdout.flush()
        # approx = torch.cat(approx).cpu()
        # targets = torch.cat(targets).cpu()
        metric_total = -loss_total

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

        return loss_total, metric_total


    def make_predictions_det(self, images, score_threshold=0.5):
        images = torch.stack(images)
        images = images.to(self.device).float()
        box_list = []
        score_list = []
        label_list = []
        det = self.model(images, torch.tensor([1]*images.shape[0]).to(self.device).float())
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]
            scores = det[i].detach().cpu().numpy()[:,4]
            label = det[i].detach().cpu().numpy()[:,5]

            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

            box_list.append(boxes)
            score_list.append(scores)
            label_list.append(label)
        return box_list, score_list, label_list

    def predict_det(self, data_loader, verbose=True, image_size=256):
        result_image_ids = []
        results_boxes = []
        results_scores = []
        results_labels = []
        total = 0
        self.model.eval()
        with torch.no_grad():
            for inputs in tqdm(data_loader, total=len(data_loader)):
                images = inputs[0]
                image_origin_sizes = inputs[-2]
                image_ids = inputs[-1]
                box_list, score_list, label_list = self.make_predictions_det(images, score_threshold=0.4)

                for i, image in enumerate(images):
                    boxes = box_list[i]
                    scores = score_list[i]
                    image_id = image_ids[i]
                    labels = label_list[i]
                    image_origin_size = image_origin_sizes[i]

                    boxes[:, 0] = (boxes[:, 0] * image_origin_size[1] / image_size)
                    boxes[:, 2] = (boxes[:, 2] * image_origin_size[1] / image_size)
                    boxes[:, 1] = (boxes[:, 1] * image_origin_size[0] / image_size)
                    boxes[:, 3] = (boxes[:, 3] * image_origin_size[0] / image_size)

                    boxes = boxes.astype(np.int32)
                    boxes[:, 0] = boxes[:, 0].clip(min=0, max=image_origin_size[1]-1)
                    boxes[:, 2] = boxes[:, 2].clip(min=0, max=image_origin_size[1]-1)
                    boxes[:, 1] = boxes[:, 1].clip(min=0, max=image_origin_size[0]-1)
                    boxes[:, 3] = boxes[:, 3].clip(min=0, max=image_origin_size[0]-1)
                    result_image_ids += [image_id]*len(boxes)
                    results_boxes.append(boxes)
                    results_scores.append(scores)
                    results_labels.append(labels)
        box_df = pd.DataFrame(np.concatenate(results_boxes), columns=['x_min','y_min','x_max','y_max'])
        test_df = pd.DataFrame({'conf':np.concatenate(results_scores), 'class_id':np.concatenate(results_labels), 'image_id':result_image_ids})
        test_df = pd.concat([test_df, box_df], axis=1)
        # test_df = test_df[test_df.conf > 0.3]
        return test_df

    def run_wbf(self, predictions, image_index, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
        boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist() for prediction in predictions]
        scores = [prediction[image_index]['conf'].tolist() for prediction in predictions]
        labels = [np.ones(prediction[image_index]['conf'].shape[0]).astype(int).tolist() for prediction in predictions]
        boxes, scores, labels = self.ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        boxes = boxes*(image_size-1)
        return boxes, scores, labels

    def make_predictions_tta_det(self, images, score_threshold=0.25):
        images = torch.stack(images)
        images = images.to(self.device).float()
        predictions = []
        for tta_transform in self.tta_transforms:
            result = []
            det = self.model(tta_transform.batch_augment(images.clone()), torch.tensor([1]*images.shape[0]).to(self.device).float())

            for i in range(images.shape[0]):
                boxes = det[i].detach().cpu().numpy()[:,:4]
                scores = det[i].detach().cpu().numpy()[:,4]
                indexes = np.where(scores > score_threshold)[0]
                boxes = boxes[indexes]
                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                boxes = tta_transform.deaugment_boxes(boxes.copy())
                result.append({
                    'boxes': boxes,
                    'conf': scores[indexes],
                })
            predictions.append(result)
        return predictions

    def predict_with_tta_det(self, data_loader, verbose=True, image_size=256):
        result_image_ids = []
        results_boxes = []
        results_scores = []
        results_labels = []
        total = 0
        import tqdm
        self.model.eval()
        with torch.no_grad():
            for inputs in tqdm.tqdm(data_loader, total=len(data_loader)):
                images = inputs[0]
                image_ids = inputs[-1]
                predictions = self.make_predictions_tta_det(images, score_threshold=0.4)
                for i in range(len(images)):
                    boxes, scores, labels = self.run_wbf(predictions, image_index=i, image_size=image_size, iou_thr=0.4, skip_box_thr=0.3, weights=None)
                    # boxes = (boxes*2).round().astype(np.int32).clip(min=0, max=1023) # *2の意味がわからなかったので一応残す。小麦のコード。下でclip等やってるのはNFLから
                    image_id = image_ids[i]

                    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

                    boxes = boxes.astype(np.int32)
                    boxes[:, 0] = boxes[:, 0].clip(min=0, max=1280-1) # 元の大きさによるっぽい
                    boxes[:, 2] = boxes[:, 2].clip(min=0, max=1280-1)
                    boxes[:, 1] = boxes[:, 1].clip(min=0, max=720-1) # 元の大きさによるっぽい
                    boxes[:, 3] = boxes[:, 3].clip(min=0, max=720-1)
                    result_image_ids += [image_id]*len(boxes)
                    results_boxes.append(boxes)
                    results_scores.append(scores)
                    results_labels.append(labels)
        box_df = pd.DataFrame(np.concatenate(results_boxes), columns=['left', 'top', 'width', 'height'])
        test_df = pd.DataFrame({'conf':np.concatenate(results_scores), 'class_id':np.concatenate(results_labels), 'image_name':result_image_ids})
        test_df = pd.concat([test_df, box_df], axis=1)
        # test_df = test_df[test_df.conf > 0.3]
        return test_df

    def make_predictions(self, images, score_threshold=0.5):
        images = torch.stack(images)
        images = images.to(self.device).float()
        box_list = []
        score_list = []
        label_list = []
        self.model.eval()
        with torch.no_grad():
            det = self.model(images)
            for i in range(images.shape[0]):
                boxes = det[i]['boxes'].detach().cpu().numpy()
                scores = det[i]['conf'].detach().cpu().numpy()
                label = det[i]['labels'].detach().cpu().numpy()
                # useing only label = 2
                # indexes = np.where((scores > score_threshold) & (label == 2))[0]
                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                # box_list.append(boxes[indexes])
                # score_list.append(scores[indexes])
                box_list.append(boxes)
                score_list.append(scores)
                label_list.append(label)
        return box_list, score_list, label_list

    def predict(self, data_loader, path=None, verbose=True, image_size=512):
        result_image_ids = []
        results_boxes = []
        results_scores = []
        results_labels = []
        with torch.no_grad():
            for inputs in data_loader:
                images = inputs[0]
                image_ids = inputs[-1]
                box_list, score_list, label_list = self.make_predictions(images, score_threshold=0.4)
                for i, image in enumerate(images):
                    boxes = box_list[i]
                    scores = score_list[i]
                    image_id = image_ids[i]
                    labels = label_list[i]
                    boxes[:, 0] = (boxes[:, 0] * 1280 / image_size)
                    boxes[:, 1] = (boxes[:, 1] * 720 / image_size)
                    boxes[:, 2] = (boxes[:, 2] * 1280 / image_size)
                    boxes[:, 3] = (boxes[:, 3] * 720 / image_size)
                    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
                    boxes = boxes.astype(np.int32)
                    boxes[:, 0] = boxes[:, 0].clip(min=0, max=1280-1)
                    boxes[:, 2] = boxes[:, 2].clip(min=0, max=1280-1)
                    boxes[:, 1] = boxes[:, 1].clip(min=0, max=720-1)
                    boxes[:, 3] = boxes[:, 3].clip(min=0, max=720-1)
                    result_image_ids += [image_id]*len(boxes)
                    results_boxes.append(boxes)
                    results_scores.append(scores)
                    results_labels.append(labels)

        box_df = pd.DataFrame(np.concatenate(results_boxes), columns=['left', 'top', 'width', 'height'])
        test_df = pd.DataFrame({'conf':np.concatenate(results_scores), 'labels':np.concatenate(results_labels), 'image_name':result_image_ids})
        test_df = pd.concat([test_df, box_df], axis=1)

        # test_df = test_df[test_df.conf > 0.3]
        return test_df

    def predict_with_tta(self, loader, path=None, verbose=True, image_size=512):
        if loader is None:
            print(f'[{self.serial}] No data to predict. Skipping prediction...')
            return None

        self.model.eval()
        with torch.no_grad():
            for tta_n in range(self.tta_transforms):
                prediction = []
                features_list = []
        #         for batch_i, inputs in enumerate(loader):
        #             X = inputs[0]
        #             X = X.to(self.device)
        #             if self.is_fp16 and APEX_FLAG:
        #                 with amp.disable_casts():
        #                     _y = self.model(X)
        #                     # features_list.append(features.detach())
        #             else:
        #                 _y = self.model(X)
        #                 # features_list.append(features.detach())
        #             prediction.append(_y.detach())
        #         if tta_n == 0:
        #             predictions = torch.cat(prediction).cpu().numpy() / tta_transforms
        #             # features = torch.cat(features_list).cpu().numpy()
        #         else:
        #             predictions += torch.cat(prediction).cpu().numpy() / tta_transforms

        # if path is not None:
        #     np.save(path, predictions)

        # if verbose & (path is not None):
        #     print(f'[{self.serial}] Prediction done. exported to {path}')
        return predictions

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

    def unfreeze(self):
        for child in self.model.module.children():
            for param in child.parameters():
                param.requires_grad = True

    # def save(self, path):
    #     self.model.eval()
    #     if isinstance(self.model, torch.nn.DataParallel):
    #         module = self.model.module
    #     else:
    #         module = self.model

    #     torch.save({
    #         'model_state_dict': module.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'scheduler_state_dict': self.scheduler.state_dict(),
    #         'event': self.event.dump_state_dict() if self.event is not None else None
    #     }, path)

    def save(self, path, epoch):
        if self.effdet:
            self.model.eval()
            torch.save({
                'model_state_dict': self.model.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'stopper': self.stopper.dump_state_dict() if self.stopper is not None else None,
                'event': self.event.dump_state_dict() if self.event is not None else None,
                'epoch': epoch,
            }, path)

        else:
            if isinstance(self.model, torch.nn.DataParallel):
                module = self.model.module
            else:
                module = self.model

            torch.save({
                'epoch': epoch + 1,
                'model': module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'stopper': self.stopper.dump_state_dict() if self.stopper is not None else None,
                'event': self.event.dump_state_dict() if self.event is not None else None
            }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.stopper.load_state_dict(checkpoint['stopper'])
        self.event.load_state_dict(checkpoint['event'])

    def calc_metric(self, tmp_model, loader, train_df, tta=1):
        return -1

    def fit(self,
            # Essential
            optimizer, scheduler, criterion,
            loader, num_epochs, loader_valid=None, loader_oof=None, loader_test=None,
            snapshot_path=None, tmp_snapshot_path=None, resume=False,  # Snapshot
            multi_gpu=False, grad_accumulations=1, calibrate_model=False, # Train
            eval_metric=None, eval_interval=1, log_metrics=[], test_time_augmentations=1, # Evaluation
            tta_transforms=None, predict_valid=True, predict_test=True,  # Prediction
            event=DummyEvent(), stopper=DummyStopper(),  # Train add-ons
            # Logger and info
            logger=DummyLogger(''), logger_interval=1,
            info_train=True, info_valid=True, info_interval=1, round_float=6,
            info_format='epoch time data loss metric earlystopping',
            verbose=True, effdet=True, image_size=None,
            ensemble_boxes=None, debug=False, train_all=False):

        if eval_metric is None:
            print(f'[{self.serial}] eval_metric is not set. Inversed criterion will be used instead.')

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.eval_metric = eval_metric
        self.log_metrics = log_metrics
        self.logger = logger
        self.event = deepcopy(event)
        self.stopper = deepcopy(stopper)
        self.current_epoch = 0
        self.best_epoch = 0
        self.effdet = effdet
        self.tta_transforms = tta_transforms
        self.ensemble_boxes = ensemble_boxes
        self.debug = debug
        self.train_all = train_all

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
        if resume:
            if effdet:
                self.load(snapshot_path)
            else:
                load_snapshots_to_model(
                    snapshot_path, self.model, self.optimizer, self.scheduler,
                    self.stopper, self.event, device=self.device)
            self.current_epoch = load_epoch(snapshot_path)
            self.best_epoch = self.current_epoch
            if verbose:
                print(
                    f'[{self.serial}] {snapshot_path} is loaded. Continuing from epoch {self.current_epoch}.')
        if self.is_fp16:
            self.model_to_fp16()
        if multi_gpu:
            self.model_to_parallel()

        self.max_epochs = self.current_epoch + num_epochs
        loss_valid, metric_valid = np.inf, -np.inf
        for epoch in range(num_epochs):
            # if epoch == 4:
            #     self.unfreeze()
            #     print('before lr:', self.optimizer.param_groups[0]['lr'])
            #     self.optimizer.param_groups[0]['lr'] /= 4
            #     print('after lr:', self.optimizer.param_groups[0]['lr'])

            # if epoch in [0, 4]:
            #     pos = 0
            #     neg = 0
            #     for child in self.model.module.children():
            #         for param in child.parameters():
            #             if param.requires_grad:
            #                 pos+=1
            #             else:
            #                 neg+=1
            #     print(f'epoch: {epoch}, pos: {pos}, neg: {neg}')

            self.current_epoch += 1
            start_time = time.time()
            if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                self.scheduler.step(loss_valid)
            else:
                self.scheduler.step()

            ### Event
            event(**{'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler,
                     'stopper': self.stopper, 'eval_metric': self.eval_metric,
                     'epoch': epoch, 'global_epoch': self.current_epoch, 'log': self.log})

            ### Training
            loss_train, metric_train = self.train_loop(loader, grad_accumulations, logger_interval, epoch)

            if info_train and epoch % info_interval == 0:
                self.print_info(info_items, info_seps, {
                    'data': 'Trn',
                    'loss': loss_train,
                    'metric': metric_train,
                    # 'logmetrics': log_metrics_train
                })

            ### Validation
            if epoch % eval_interval == 0:
                loss_valid, metric_valid = self.valid_loop(loader_valid, grad_accumulations, logger_interval, epoch)

                early_stopping_target = metric_valid
                if (self.stopper(early_stopping_target)) or (self.train_all):  # score improved
                    self.save(snapshot_path, epoch)

                if info_valid and epoch % info_interval == 0:
                    self.print_info(info_items, info_seps, {
                        'data': 'Val',
                        'loss': loss_valid,
                        'metric': metric_valid,
                    })

            # Stopped by overfit detector
            if self.stopper.stop():
                break

        if not effdet:
            if verbose:
                print(
                    f"[{self.serial}] Best score is {self.stopper.score():.{self.round_float}f}")
            load_snapshots_to_model(str(snapshot_path), self.model, self.optimizer)

        if calibrate_model:
            if loader_valid is None:
                print('loader_valid is necessary for calibration.')
            else:
                self.calibrate_model(loader_valid)

        if predict_valid:
            if self.effdet:
                self.oof = self.predict_det(loader_oof, verbose=verbose, image_size=image_size)
                if tta_transforms is not None:
                    self.tta_oof = self.predict_with_tta_det(
                        loader_oof, verbose=verbose, image_size=image_size)
            else:
                self.oof = self.predict(loader_oof, verbose=verbose, image_size=image_size)
                if tta_transforms is not None:
                    self.tta_oof = self.predict_with_tta(
                        loader_oof, verbose=verbose, image_size=image_size)
        if predict_test:
            if self.effdet:
                self.pred = self.predict_det(loader_test, verbose=verbose, image_size=image_size)
                if tta_transforms is not None:
                    self.tta_pred = self.predict_with_tta_det(
                        loader_test, verbose=verbose, image_size=image_size)
            else:
                self.pred = self.predict(loader_test, verbose=verbose, image_size=image_size)
                if tta_transforms is not None:
                    self.tta_pred = self.predict_with_tta(
                        loader_test, verbose=verbose, image_size=image_size)

    def calibrate_model(self, loader):
        self.model = TemperatureScaler(self.model).to(self.device)
        self.model.set_temperature(loader)


'''
Cross Validation for Tabular data
'''

class TorchCV:
    IGNORE_PARAMS = {
        'loader', 'loader_valid', 'loader_test', 'snapshot_path', 'logger'
    }
    TASKS = {'binary', 'regression'}

    def __init__(self, model, datasplit, device=None):
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model
        self.device = device
        self.datasplit = datasplit

        self.models = []
        self.oof = None
        self.pred = None
        self.imps = None

    def run(self, X, y, X_test=None,
            group=None, transform=None, task='binary',
            eval_metric=None, batch_size=64, n_splits=None,
            snapshot_dir=None, logger=DummyLogger(''),
            fit_params={}, verbose=True):

        if not isinstance(eval_metric, (list, tuple, set)):
            eval_metric = [eval_metric]
        if snapshot_dir is None:
            snapshot_dir = Path().cwd()
        if not isinstance(snapshot_dir, Path):
            snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        assert snapshot_dir.is_dir()
        assert task in self.TASKS

        if n_splits is None:
            K = self.datasplit.get_n_splits()
        else:
            K = n_splits

        self.imps = np.zeros((X.shape[1], K))
        self.scores = np.zeros((len(eval_metric), K))
        self.numpy2dataset = Numpy2Dataset(task)

        default_path = snapshot_dir/'default.pt'
        save_snapshots(default_path, 0, self.model, fit_params['optimizer'], fit_params['scheduler'])
        template = {}
        for item in ['stopper', 'event']:
            if item in fit_params.keys():
                template[item] = deepcopy(fit_params[item])
        for item in self.IGNORE_PARAMS:
            if item in fit_params.keys():
                fit_params.pop(item)

        for fold_i, (train_idx, valid_idx) in enumerate(
            self.datasplit.split(X, y, group)):

            Xs = {'train': X[train_idx], 'valid': X[valid_idx],
                  'test': X_test.copy() if X_test is not None else None}
            ys = {'train': y[train_idx], 'valid': y[valid_idx]}

            if transform is not None:
                transform(Xs, ys)

            ds_train = self.numpy2dataset(Xs['train'], ys['train'])
            ds_valid = self.numpy2dataset(Xs['valid'], ys['valid'])
            ds_test = self.numpy2dataset(Xs['test'], np.arange(len(Xs['test'])))

            loader_train = D.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
            loader_valid = D.DataLoader(ds_valid, batch_size=batch_size, shuffle=False)
            loader_test = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False)

            params_fold = copy(fit_params)
            params_fold.update({
                'loader': loader_train,
                'loader_valid': loader_valid,
                'loader_test': loader_test,
                'snapshot_path': snapshot_dir / f'snapshot_fold_{fold_i}.pt',
                'logger': logger
            })
            params_fold.update(deepcopy(template))
            load_snapshots_to_model(default_path, self.model, params_fold['optimizer'], params_fold['scheduler'])

            trainer_fold = TorchTrainer(self.model, self.device, serial=f'fold_{fold_i}')
            trainer_fold.fit(**params_fold)

            if fold_i == 0: # Initialize oof and prediction
                self.oof = np.zeros((len(X), trainer_fold.oof.shape[1]), dtype=np.float)
                if X_test is not None:
                    self.pred = np.zeros((len(X_test), trainer_fold.pred.shape[1]), dtype=np.float)

            self.oof[valid_idx] = trainer_fold.oof
            if X_test is not None:
                self.pred += trainer_fold.pred / K

            for i, _metric in enumerate(eval_metric):
                score = _metric(ys['valid'], self.oof[valid_idx])
                self.scores[i, fold_i] = score

            if verbose >= 0:
                log_str = f'[CV] Fold {fold_i+1}:'
                log_str += ''.join(
                    [f' m{i}={self.scores[i, fold_i]:.5f}' for i in range(len(eval_metric))])
                print(log_str)

        log_str = f'[CV] Overall:'
        log_str += ''.join(
            [f' m{i}={me:.5f}±{se:.5f}' for i, (me, se) in enumerate(zip(
                np.mean(self.scores, axis=1),
                np.std(self.scores, axis=1)/np.sqrt(len(eval_metric))
            ))]
        )
        print(log_str)

