import argparse
import glob
import numpy as np
import os.path as osp
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    return parser.parse_args()


def main():
    args = parse_args()
    folder = args.folder
    folds = glob.glob(osp.join(folder, 'fold*'))
    folds = np.sort(folds)
    cv_list = []
    for i, fo in enumerate(folds):
        logs = np.sort(glob.glob(osp.join(fo, '*.log')))
        # Get most recent log
        log = logs[-1]
        with open(log) as f:
            log = [l.strip() for l in f.readlines()]
        metric_list = []
        for line in log:
            if re.search('mAP:', line):
                metric = line
                metric = float(metric.split('mAP:')[-1])
                metric_list.append(metric)
        max_metric = np.max(metric_list)
        cv_list.append(max_metric)
        print(f'FOLD {i}: {max_metric:0.4f}')

    print(f'CV AVG: {np.mean(cv_list):0.4f}')


if __name__ == '__main__':
    main()