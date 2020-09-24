from __future__ import division
import sys
import os
import glob
import torch
import torch.nn as nn
import argparse
import matplotlib
sys.path.append('/root/LPC_MOT/learnable_proposal_classifier/gcn_based_purity_network/')
from dsgcn.models.dsgcn import dsgcn
import json
import numpy as np
import random
from datasets import build_dataset_pipeline, build_processor, build_dataloader
from models import build_model

def parse_args():
    parser = argparse.ArgumentParser(description='Purity Inference Network')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--stage', choices=['pipeline'], default='pipeline')
    parser.add_argument('--input_dir', help='input data path')
    parser.add_argument('--output_dir', help='the dir to save inference results')
    parser.add_argument('--load_from1', default=None, help='the checkpoint file to load from')
    parser.add_argument('--gpus', type=int, default=1,
            help='number of gpus(only applicable to non-distributed training)')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    args = parser.parse_args()
    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cfg = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    cfg.test_data = {}
    cfg.test_data['feat_path'] = os.path.join(cfg.input_dir, 'features_total.b')
    cfg.test_data['trk_vid_path'] = os.path.join(cfg.input_dir, 'tracklet_video_name.json')
    cfg.test_data['trk_frm_num'] = os.path.join(cfg.input_dir, 'tracklet_frame_num.json')
    cfg.test_data['spatem_path'] = os.path.join(cfg.input_dir,'tracklet_temporal_spatial.json')
    cfg.test_data['proposal_folders'] = os.path.join(cfg.input_dir, 'eval1')
    cfg.workers_per_gpu = 1
    cfg.batch_size_per_gpu = 128
    cfg.test_data['phase'] = 'test'
    fn_node_pattern = '*_node.json'
    fn_nodes = glob.glob(os.path.join(cfg.test_data['proposal_folders'], fn_node_pattern))
    if len(fn_nodes) > 0:
        # set cuda
        cfg.cuda = not cfg.no_cuda and torch.cuda.is_available()
        # set random seeds
        if cfg.seed is not None:
            set_random_seed(cfg.seed)
        model = torch.load(cfg.load_from1)
        dataset = build_dataset_pipeline(cfg.test_data)
        processor = build_processor(cfg.stage)

        output_probs = []
        if cfg.gpus == 1:
            data_loader = build_dataloader(
                    dataset,
                    processor,
                    cfg.batch_size_per_gpu,
                    cfg.workers_per_gpu,
                    train=False)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            for i, data in enumerate(data_loader):
                with torch.no_grad():
                    output = model(data, return_loss=False)
                    output = output[:,1]
                    output = output.view(-1)
                    output_probs.append(output.tolist())
        else:
            raise NotImplementedError
        output_probs1 = [iop for item in output_probs for iop in item]
        output_probs1 = output_probs1[:len(fn_nodes)]
        assert len(output_probs1) == len(fn_nodes)
        estimated_iop_dict = {}
        for i, node in enumerate(dataset.lst):
            node_name = node.split('/')[-1]
            estimated_iop = output_probs1[i]
            estimated_iop_dict[node_name] = estimated_iop
        with open(cfg.output_dir + '/Estimated_IoP_eval.json', 'w') as f:
            json.dump(estimated_iop_dict, f)


if __name__ == '__main__':
    main()
