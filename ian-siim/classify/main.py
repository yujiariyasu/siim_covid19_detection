import argparse
import os, os.path as osp
import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf

from skp.controls import datamaker, training, inference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    parser.add_argument('config', type=str)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--kfold', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--cpu-inference', action='store_true')
    parser.add_argument('--checkpoint', type=lambda s: s.split(','))
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--imgfiles', type=str)
    parser.add_argument('--act-fn', type=str)
    parser.add_argument('--find-lr', action='store_true')
    parser.add_argument('--save-features-dir', type=str)
    parser.add_argument('--save-preds-file', type=str)
    parser.add_argument('--save-pseudo-segs', type=str)
    parser.add_argument('--rank-average', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def setup_dist(rank):
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl', init_method='env://')
    return torch.distributed.get_world_size()


def main(args):

    # Print some info ...
    if args.local_rank == 0:
        print('PyTorch environment ...')
        print(f'  torch.__version__              = {torch.__version__}')
        print(f'  torch.version.cuda             = {torch.version.cuda}')
        print(f'  torch.backends.cudnn.version() = {torch.backends.cudnn.version()}')
        print('\n')

    # Load config
    cfg = OmegaConf.load(args.config)
    if args.seed >= 0:
        cfg.experiment.seed = args.seed

    # Set accelerator in config
    cfg.accelerator = args.accelerator

    if args.mode not in ['train','find_lr'] and args.accelerator == 'ddp':
        # Using custom distributed inference
        cfg.local_rank = args.local_rank
        cfg.world_size = setup_dist(args.local_rank)
        
    # Handle experiment name
    cfg.experiment.name = osp.basename(args.config).replace('.yaml', '')
    if args.local_rank == 0: print(f'Running experiment {cfg.experiment.name} ...')
    if args.sync_batchnorm:
        cfg.experiment.name = osp.join(cfg.experiment.name, 'sbn')
        
    # If running K-fold, change seed and save directory
    # Also need to edit folds in data
    if args.kfold >= 0:
        cfg.experiment.seed = int(f'{cfg.experiment.seed}{args.kfold}')
        cfg.experiment.name = osp.join(cfg.experiment.name, f'fold{args.kfold}')
        if isinstance(cfg.data.inner_fold, (int,float)): 
            cfg.data.inner_fold = None
        cfg.data.outer_fold = args.kfold

    if args.local_rank == 0: print(f'Saving checkpoints and logs to {cfg.experiment.save_dir} ...')

    # Set number of workers
    if cfg.data:
        cfg.data.num_workers = args.num_workers 
    
    # Set seed
    assert hasattr(cfg.experiment, 'seed'), \
        'Please specify `seed` under `experiment` in config file'
    seed = pl.seed_everything(cfg.experiment.seed)

    if args.mode in ['train', 'find_lr']:
        getattr(training, args.mode)(cfg, args)
    else:
        getattr(inference, args.mode)(cfg, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)