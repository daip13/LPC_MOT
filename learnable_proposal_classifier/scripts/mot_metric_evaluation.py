import numpy as np
import pandas as pd

import os
import os.path as osp
import pickle
from pathlib import Path
import argparse

import motmetrics as mm

from copy import deepcopy
from collections import OrderedDict

#import torch
#from torch_scatter import scatter_add
#from pytorch_lightning import Callback

import re

import time
#from mot_neural_solver.path_cfg import DATA_PATH
#from mot_neural_solver.utils.misc import load_pickle, save_pickle

###########################################################################
# MOT Metrics
###########################################################################

# Formatting for MOT Metrics reporting
mh = mm.metrics.create()
MOT_METRICS_FORMATERS = mh.formatters
MOT_METRICS_NAMEMAP = mm.io.motchallenge_metric_names
MOT_METRICS_NAMEMAP.update({'norm_' + key: 'norm_' + val for key, val in MOT_METRICS_NAMEMAP.items()})
MOT_METRICS_FORMATERS.update({'norm_' + key: val for key, val in MOT_METRICS_FORMATERS.items()})
MOT_METRICS_FORMATERS.update({'constr_sr': MOT_METRICS_FORMATERS['mota']})


def compute_mot_metrics(gt_path, out_mot_files_path, seqs, print_results = True):
    """
    The following code is adapted from
    https://github.com/cheind/py-motmetrics/blob/develop/motmetrics/apps/eval_motchallenge.py
    It computes all MOT metrics from a set of output tracking files in MOTChallenge format
    Args:
        gt_path: path where MOT ground truth files are stored. Each gt file must be stored as
        <SEQ NAME>/gt/gt.txt
        out_mot_files_path: path where output files are stored. Each file must be named <SEQ NAME>.txt
        seqs: Names of sequences to be evalluated

    Returns:
        Individual and overall MOTmetrics for all sequeces
    """
    def _compare_dataframes(gts, ts):
        """Builds accumulator for each sequence."""
        accs = []
        names = []
        for k, tsacc in ts.items():
            if k in gts:
                accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
                names.append(k)

        return accs, names

    mm.lap.default_solver = 'lapsolver'
    gtfiles = [os.path.join(gt_path, i, 'gt/gt.txt') for i in seqs]
    tsfiles = [os.path.join(out_mot_files_path, '%s.txt' % i) for i in seqs]
    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D')) for f in tsfiles])

    mh = mm.metrics.create()
    accs, names = _compare_dataframes(gt, ts)

    # We will need additional metrics to compute IDF1, etc. from different splits inf CrossValidationEvaluator
    summary = mh.compute_many(accs, names=names,
                              metrics=mm.metrics.motchallenge_metrics + ['num_objects',
                                                                         'idtp', 'idfn', 'idfp', 'num_predictions'],
                              generate_overall=True)
    if print_results:
        print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

    return summary


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, help='', required=True)
    parser.add_argument('--out_mot_files_path', type=str, help='', required=True)
    return parser.parse_args()

def main():
    args = parse_arguments()
    seqs = ['MOT17-01-DPM', 'MOT17-03-DPM', 'MOT17-06-DPM', 'MOT17-07-DPM', 'MOT17-08-DPM', 'MOT17-12-DPM', 'MOT17-14-DPM', 'MOT17-01-SDP', 'MOT17-03-SDP', 'MOT17-06-SDP', 'MOT17-07-SDP', 'MOT17-08-SDP', 'MOT17-12-SDP', 'MOT17-14-SDP', 'MOT17-01-FRCNN', 'MOT17-03-FRCNN', 'MOT17-06-FRCNN', 'MOT17-07-FRCNN', 'MOT17-08-FRCNN', 'MOT17-12-FRCNN', 'MOT17-14-FRCNN']
    summary = compute_mot_metrics(args.gt_path, args.out_mot_files_path, seqs, print_results = True)
if __name__ == "__main__":
    main()
