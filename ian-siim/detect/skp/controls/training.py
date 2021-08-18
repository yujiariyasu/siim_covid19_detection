import numpy as np
import os, os.path as osp
import pytorch_lightning as pl
import re
import torch

from omegaconf import ListConfig
from typing import Any, Dict, Optional

from .. import builder
from .. import metrics
from .datamaker import get_train_val_datasets


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):

    @classmethod
    def _format_checkpoint_name(
        cls,
        filename: Optional[str],
        epoch: int,
        step: int,
        metrics: Dict[str, Any],
        prefix: str = "",
    ) -> str:
        if not filename:
            # filename is not set, use default name
            filename = "{epoch}" + cls.CHECKPOINT_JOIN_CHAR + "{step}"

        # check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)
        if len(groups) >= 0:
            metrics.update({"epoch": epoch, 'step': step})
            for group in groups:
                name = group[1:]
                filename = filename.replace(group, name + "{" + name)
                if name not in metrics:
                    metrics[name] = 0
            filename = filename.format(**metrics)

        if prefix:
            filename = cls.CHECKPOINT_JOIN_CHAR.join([prefix, filename])

        return filename


def build_elements(cfg): 
    # Create model
    model = builder.build_model(cfg)
    if cfg.model.load_pretrained:
        print(f'Loading pretrained model from {cfg.model.load_pretrained} ...')
        weights = torch.load(cfg.model.load_pretrained, map_location=lambda storage, loc: storage)['state_dict']
        weights = {re.sub(r'^model.', '', k) : v for k,v in weights.items()}
        model.load_state_dict(weights, strict=False)

    # Create loss
    criterion = builder.build_loss(cfg, model.config)
    model.loss_fn = criterion
    # Create optimizer
    optimizer = builder.build_optimizer(cfg, model.parameters())
    # Create learning rate scheduler
    scheduler = builder.build_scheduler(cfg, optimizer)
    return model,optimizer,scheduler


def build_trainer(cfg, args, task, snapshot=-1): 
    version = f'snapshot_{snapshot}' if snapshot >= 0 else ''
    trainer = pl.Trainer.from_argparse_args(args,
        max_epochs=cfg.train.num_epochs,
        callbacks=[ModelCheckpoint(
            monitor='vm',
            filename='{epoch:03d}-{vm:.4f}',
            save_last=True,
            save_weights_only=True,
            mode=cfg.evaluate.mode,
            save_top_k=cfg.evaluate.save_top_k or 1
        ), pl.callbacks.LearningRateMonitor(logging_interval='step')],
        logger=pl.loggers.CSVLogger(
            save_dir=cfg.experiment.save_dir,
            name=cfg.experiment.name,
            version=version
        ),
        replace_sampler_ddp=False,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches or 1,
        profiler='simple'
    )
    
    return trainer


def define_task(cfg, args):
    train_dataset, valid_dataset = get_train_val_datasets(cfg)
    model,optimizer,scheduler = build_elements(cfg)
    evaluation_metrics = [getattr(metrics, m['name'])(**m['params']) for m in cfg.evaluate.metrics]
    valid_metric = list(cfg.evaluate.monitor) \
        if isinstance(cfg.evaluate.monitor, ListConfig) \
        else cfg.evaluate.monitor

    task = builder.build_task(cfg, model)
    
    task.set('optimizer', optimizer)
    task.set('scheduler', scheduler)
    task.set('metrics', evaluation_metrics)
    task.set('valid_metric', valid_metric)

    task.set('train_dataset', train_dataset)
    task.set('valid_dataset', valid_dataset)

    return task


def train(cfg, args):
    task = define_task(cfg, args)
    if args.find_lr:
        cfg.optimizer.params.lr = find_lr(cfg, args)
        task = define_task(cfg, args)
    num_snapshots = cfg.train.num_snapshots or 1
    if num_snapshots == 1:
        trainer = build_trainer(cfg, args, task)
        trainer.fit(task)
    else:
        for i in range(num_snapshots):
            if i > 0:
                # Need to rebuild optimizer and scheduler
                task.set('optimizer', builder.build_optimizer(cfg, task.model.parameters()))
                task.set('scheduler', builder.build_scheduler(cfg, task.optimizer))
            trainer = build_trainer(cfg, args, task, snapshot=i)
            trainer.fit(task)


def suggest_lr(results, skip_begin=10):
    try:
        # Ensure that selected LR is less than LR at loss minima
        # Otherwise, can sometimes be tricked by downward slopes at higher LRs
        loss = np.array(results['loss'])
        loss = loss[skip_begin:loss.argmin()]
        loss = loss[np.isfinite(loss)]
        min_grad = np.gradient(loss).argmin()
        return results['lr'][min_grad + skip_begin]
    except:
        raise Exception('Failed to compute suggestion for `lr`. There might not be enough points.')


def find_lr(cfg, args):
    task = define_task(cfg, args)
    trainer = build_trainer(cfg, args, task)
    lr_finder = trainer.tuner.lr_find(task)
    suggested_lr = suggest_lr(lr_finder.results)
    print(f'SUGGESTED LEARNING RATE : {suggested_lr:.4e}')
    if not osp.exists('lr_finder'): os.makedirs('lr_finder')
    fig = lr_finder.plot(suggest=False)
    fig.suptitle(f'Suggested LR : {suggested_lr:.4e}')
    fig.savefig(osp.join('lr_finder', f'{osp.basename(args.config).replace(".yaml", "")}_lr-{suggested_lr:.2e}.png'))
    return suggested_lr
