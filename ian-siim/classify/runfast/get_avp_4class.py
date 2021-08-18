import argparse
import glob
import os, os.path as osp
import pandas as pd

from numpy import sort, mean


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = sort(glob.glob(osp.join(args.folder, 'fold*/metrics.csv')))
    metrics_list = []
    for i,m in enumerate(metrics):
        df = pd.read_csv(m)
        met = df[['avp0','avp1','avp2','avp3']].mean(axis=1).max()
        print(f'FOLD{i}: {met:0.4f}')
        metrics_list.append(met)
    print(f'CVAVG: {mean(metrics_list):0.4f}\n')


if __name__ == '__main__':
    main()

