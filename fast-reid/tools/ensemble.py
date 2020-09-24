'''
Ensemble multiple model:
1. concat
--------------------------------------
Author: Yang Qian
Email: yqian@aibee.com
'''
import argparse
import logging
import sys
import os
import atexit
import bisect

import numpy as np
import cv2
import torch
import tqdm
from torch.backends import cudnn
from collections import deque
from collections import OrderedDict

sys.path.append('.')

from fastreid.evaluation import evaluate_rank, ReidEvaluator, print_csv_format
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from fastreid.utils.visualizer import Visualizer
from fastreid.utils.events import TensorboardXWriter
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import DefaultPredictor

cudnn.benchmark = True
logger = logging.getLogger('fastreid.ensembel')

def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="fastreid model ensembel")
    parser.add_argument("--config-file", default=["configs/Aibee/Store-CTF-body.yml"], metavar="FILE", nargs='+', help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

class FeatureExtraction(object):
    def __init__(self, cfgs):
        """
        Args:
            cfg (CfgNode):
        """
        self.cfgs = cfgs
        self.predictors = []

        for cfg in cfgs:
            self.predictors.append(DefaultPredictor(cfg))

    def run_on_loader(self, data_loader):
        for batch in data_loader:
            predictions = []
            for predictor in self.predictors:
                predictions.append(predictor(batch["images"]))
            yield torch.cat(predictions, dim=-1), batch


if __name__ == "__main__":
    args = get_parser().parse_args()
    logger = setup_logger()
    cfgs = []
    for config_file in args.config_file:
        cfg = setup_cfg(config_file, args.opts)
        cfgs.append(cfg)
    results = OrderedDict()
    for dataset_name in cfgs[0].DATASETS.TESTS:
        test_loader, num_query = build_reid_test_loader(cfgs[0], dataset_name)
        evaluator = ReidEvaluator(cfgs[0], num_query)
        feat_extract = FeatureExtraction(cfgs)
        for (feat, batch) in tqdm.tqdm(feat_extract.run_on_loader(test_loader), total=len(test_loader)):
            evaluator.process(batch, feat)
        result = evaluator.evaluate()
        results[dataset_name] = result
    print_csv_format(results)
    