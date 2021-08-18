from datasets import DatasetTrain, DatasetTest
from det_datasets import DetDatasetTrain, DetDatasetTest
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def collate_fn(batch):
    return tuple(zip(*batch))

def get_loaders(cfg, opt):
    tr = cfg.train_df[cfg.train_df.fold != opt.fold]
    if cfg.upsample:
        dfs = []
        dfs.append(tr[tr['Typical Appearance']==1])
        for _ in range(2):
            dfs.append(tr[tr['Negative for Pneumonia']==1])
        for _ in range(3):
            dfs.append(tr[tr['Indeterminate Appearance']==1])
        for _ in range(7):
            dfs.append(tr[tr['Atypical Appearance']==1])
        tr = pd.concat(dfs)
        tr = tr.sample(len(tr))

    val = cfg.train_df[cfg.train_df.fold == opt.fold]
    if ('type' in list(val)) and ('origin' in val['type'].unique()):
        val = val[val.type=='origin']
    val = val.drop_duplicates('path')

    train_ds = DatasetTrain(
        df=tr,
        transforms=cfg.transform['dataset_train'],
        cfg=cfg,
        split='train'
    )

    fine_tune_ds = DatasetTrain(
        df=tr,
        transforms=cfg.transform['dataset_val'],
        cfg=cfg,
        split='train'
    )

    val_transform = cfg.transform['dataset_val'] if cfg.tta==1 else cfg.transform['dataset_tta']

    valid_ds = DatasetTrain(
        df=val,
        transforms=val_transform,
        cfg=cfg,
        split='valid'
    )

    test_ds = DatasetTest(
        df=cfg.test_df,
        transforms=cfg.transform['dataset_val'],
        cfg=cfg
    )

    tta_test_ds = DatasetTest(
        df=cfg.test_df,
        transforms=cfg.transform['dataset_tta'],
        cfg=cfg
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=opt.n_cpu, worker_init_fn=worker_init_fn)
    fine_tune_loader = DataLoader(
        fine_tune_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=opt.n_cpu, worker_init_fn=worker_init_fn)
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=opt.n_cpu, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=opt.n_cpu, worker_init_fn=worker_init_fn)
    tta_test_loader = DataLoader(
        tta_test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=opt.n_cpu, worker_init_fn=worker_init_fn)

    return {
        'train': train_loader,
        'fine_tune': fine_tune_loader,
        'valid': valid_loader,
        'test': test_loader,
        'tta_test': tta_test_loader,
    }

def get_det_loaders(cfg, opt):
    tr = cfg.train_df[cfg.train_df.fold != opt.fold]
    val = cfg.train_df[cfg.train_df.fold == opt.fold]
    oof = cfg.oof_df[cfg.oof_df.fold == opt.fold]

    train_ds = DetDatasetTrain(
        df=tr,
        transforms=cfg.transform['train'],
    )

    fine_tune_ds = DetDatasetTrain(
        df=tr,
        transforms=cfg.transform['val'],
    )

    val_transform = cfg.transform['val']# if cfg.tta==1 else cfg.transform['dataset_tta']

    valid_ds = DetDatasetTrain(
        df=val,
        transforms=val_transform,
    )

    test_ds = DetDatasetTest(
        df=cfg.test_df,
        transforms=cfg.transform['test'],
    )

    oof_ds = DetDatasetTrain(
        df=oof,
        transforms=cfg.transform['val'],
    )

    tta_test_ds = DetDatasetTest(
        df=cfg.test_df,
        transforms=cfg.transform['test'],
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=opt.n_cpu, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    fine_tune_loader = DataLoader(
        fine_tune_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=opt.n_cpu, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=opt.n_cpu, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    oof_loader = DataLoader(
        oof_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=opt.n_cpu, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=opt.n_cpu, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    tta_test_loader = DataLoader(
        tta_test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=opt.n_cpu, worker_init_fn=worker_init_fn, collate_fn=collate_fn)

    return {
        'train': train_loader,
        'fine_tune': fine_tune_loader,
        'valid': valid_loader,
        'oof': oof_loader,
        'test': test_loader,
        'tta_test': tta_test_loader,
    }

def get_vae_loaders(cfg, opt):
    train_ds = VAEDataset(
        df=cfg.train_df[cfg.train_df.fold != opt.fold],
        image_dir=cfg.train_image_path,
        transforms=cfg.transform['train'],
    )

    fine_tune_ds = VAEDataset(
        df=cfg.train_df[cfg.train_df.fold != opt.fold],
        image_dir=cfg.train_image_path,
        transforms=cfg.transform['fine_tune'],
    )

    valid_ds = VAEDataset(
        df=cfg.train_df[cfg.train_df.fold == opt.fold],
        image_dir=cfg.train_image_path,
        transforms=cfg.transform['test'],
    )

    test_ds = VAEDataset(
        df=cfg.test_df,
        image_dir=cfg.test_image_path,
        transforms=cfg.transform['test'],
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=opt.n_cpu, worker_init_fn=worker_init_fn)
    fine_tune_loader = DataLoader(
        fine_tune_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=opt.n_cpu, worker_init_fn=worker_init_fn)
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=opt.n_cpu, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=opt.n_cpu, worker_init_fn=worker_init_fn)

    return {
        'train': train_loader,
        'fine_tune': fine_tune_loader,
        'valid': valid_loader,
        'test': test_loader,
        'tta_valid': None,
        'tta_test': None,
    }

