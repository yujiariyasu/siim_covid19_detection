import argparse
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_config')
    parser.add_argument('--gpu', default='rtx6000')
    parser.add_argument('--kfold', type=int, default=0)
    parser.add_argument('--single-node', action='store_true')
    return parser.parse_args()


def make_batch_script(cfg, gpu, kfold, single_node):
    prefix = cfg.split('/')[-1].replace('.py', '')
    folder = cfg.split('/')[-2]
    name = f'{prefix}_{gpu}'
    with open(f'batch-scripts/{name}.sh', 'w') as f:
        f.write('#!/bin/bash -l\n')
        f.write('#SBATCH -p gpu\n')
        f.write('#SBATCH --qos=derek.merck\n')
        ngpus = '#SBATCH --gres=gpu:4\n' if single_node else '#SBATCH --gres=gpu:1\n'
        f.write(ngpus)
        nnodes = '#SBATCH --nodes=1\n' if single_node else '#SBATCH --nodes=4\n'
        f.write(nnodes)
        f.write('#SBATCH --cpus-per-gpu=4\n')
        f.write('#SBATCH --mem-per-cpu=4g\n')
        f.write('#SBATCH -t 6:00:00\n')
        f.write(f'#SBATCH --constraint={gpu}\n')
        f.write(f'#SBATCH --out=outfiles/{name}.out\n\n')
        f.write('export MASTER_PORT=$((12000 + RANDOM % 20000))\n')
        f.write('module load gcc/5 cuda/11.1.0\n')
        f.write('export OMP_NUM_THREADS=2\n')
        f.write('cd /blue/derek.merck/ianpan/cov2/mmdetection/\n')
        srunline = f'srun bash tools/dist_train.sh configs/{folder}/{prefix}.py 4'
        if kfold > 0:
            srunline = [srunline] * kfold
            for i,srun in enumerate(srunline):
                srun = srun + f' --kfold {i}'
                f.write(f'{srun}\n\n')
        else:
            f.write(f'{srunline}\n')
    return f'{name}.sh'


args = parse_args()
make_batch_script(args.base_config, args.gpu, args.kfold, args.single_node)
