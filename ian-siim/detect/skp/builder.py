import copy
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .effdet.data.loader import DetectionFastCollate, PrefetchLoader

from . import data
from . import losses
from . import models 
from . import optim
from . import tasks


def get_name_and_params(base):
    name = getattr(base, 'name')
    params = getattr(base, 'params') or {}
    return name, params


def get_transform(base, transform):
    if not base: return None
    transform = getattr(base, transform)
    if not transform: return None
    name, params = get_name_and_params(transform)
    return getattr(data.transforms, name)(**params)


def build_transforms(cfg, mode):
    # 1-Resize
    resizer = get_transform(cfg.transform, 'resize')
    # 2-(Optional) Data augmentation
    augmenter = get_transform(cfg.transform, 'augment') if mode == 'train' else None
    # 3-(Optional) Crop
    cropper = get_transform(cfg.transform, 'crop')
    # 4-Preprocess
    preprocessor = get_transform(cfg.transform, 'preprocess')
    return {
        'resize': resizer,
        'augment': augmenter,
        'crop': cropper,
        'preprocess': preprocessor
    }


def build_dataset(cfg, data_info, mode):
    dataset_class = getattr(data.datasets, cfg.data.dataset.name)
    dataset_params = cfg.data.dataset.params
    dataset_params.test_mode = mode != 'train'
    dataset_params = dict(dataset_params)
    transforms = build_transforms(cfg, mode)
    dataset_params.update(transforms)
    dataset_params.update(data_info)
    return dataset_class(**dataset_params)


def build_dataloader(cfg, dataset, mode, dist_inference=False):

    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    dataloader_params = {}
    dataloader_params['num_workers'] = cfg.data.num_workers
    dataloader_params['drop_last'] = mode == 'train'
    dataloader_params['shuffle'] = mode == 'train'
    if mode in ('train', 'valid'):
        dataloader_params['batch_size'] = cfg.train.batch_size
        sampler = None
        if cfg.data.sampler and mode == 'train':
            name, params = get_name_and_params(cfg.data.sampler)
            sampler = getattr(data.samplers, name)(dataset, **params)
        if sampler:
            dataloader_params['shuffle'] = False
            if cfg.accelerator == 'ddp':
                sampler = data.DistributedWrapperSampler(sampler)
            dataloader_params['sampler'] = sampler
            print(f'Using sampler {sampler} for training ...')
        elif cfg.accelerator == 'ddp':
            dataloader_params['shuffle'] = False
            dataloader_params['sampler'] = DistributedSampler(dataset)
    else:
        dataloader_params['batch_size'] = cfg.evaluate.batch_size or cfg.train.batch_size
        if dist_inference:
            dataloader_params['sampler'] = DistributedSampler(dataset, shuffle=False)

    loader = DataLoader(dataset,
        **dataloader_params,
        collate_fn=DetectionFastCollate() if mode not in ('test','predict') else None,
        pin_memory=True,
        worker_init_fn=worker_init_fn)

    return loader


def build_model(cfg):
    name, params = get_name_and_params(cfg.model)
    model = getattr(models.engine, name)(**params)
    print(f'Creating model <{name}> ...')
    if 'backbone' in cfg.model.params:
        print(f'  Using backbone <{cfg.model.params.backbone}> ...')
    if 'pretrained' in cfg.model.params:
        print(f'  Pretrained : {cfg.model.params.pretrained}')
    return model 


def build_loss(cfg, config):
    name, params = get_name_and_params(cfg.loss)
    print(f'Using loss function <{name}> ...')
    config = copy.deepcopy(config)
    for k,v in params.items():
        assert k in config
        setattr(config, k, v)
    criterion = getattr(losses, name)(config)
    return criterion


def build_scheduler(cfg, optimizer):
    # Some schedulers will require manipulation of config params
    # My specifications were to make it more intuitive for me
    name, params = get_name_and_params(cfg.scheduler)
    print(f'Using learning rate schedule <{name}> ...')

    if name == 'CosineAnnealingLR':
        # eta_min <-> final_lr
        # Set T_max as 100000 ... this is changed in on_train_start() method
        # of the LightningModule task 

        params = {
            'T_max': 100000,
            'eta_min': max(params.final_lr, 1.0e-8)
        }

    if name in ('OneCycleLR', 'CustomOneCycleLR'):
        # Use learning rate from optimizer parameters as initial learning rate
        lr_0 = cfg.optimizer.params.lr
        lr_1 = params.max_lr
        lr_2 = params.final_lr
        # lr_0 -> lr_1 -> lr_2 
        pct_start = params.pct_start
        params = {}
        params['steps_per_epoch'] = 100000 # see above- will fix in task
        params['epochs'] = cfg.train.num_epochs
        params['max_lr'] = lr_1
        params['pct_start'] = pct_start
        params['div_factor'] = lr_1 / lr_0 # max/init
        params['final_div_factor'] = lr_0 / max(lr_2, 1.0e-8) # init/final
        
    scheduler = getattr(optim, name)(optimizer=optimizer, **params)
    
    # Some schedulers might need more manipulation after instantiation
    if name in ('OneCycleLR', 'CustomOneCycleLR'):
        scheduler.pct_start = params['pct_start']

    # Set update frequency
    if name in ('OneCycleLR', 'CustomOneCycleLR', 'CosineAnnealingLR'):
        scheduler.update_frequency = 'on_batch'
    elif name in ('ReduceLROnPlateau'):
        scheduler.update_frequency = 'on_valid'
    else:
        scheduler.update_frequency = 'on_epoch'

    return scheduler


def build_optimizer(cfg, parameters):
    name, params = get_name_and_params(cfg.optimizer)
    print(f'Using optimizer <{name}> ...')
    optimizer = getattr(optim, name)(parameters, **params)
    return optimizer


def build_task(cfg, model):
    name, params = get_name_and_params(cfg.task)
    print(f'Building task <{name}> ...')
    return getattr(tasks, name)(cfg, model, **params)


