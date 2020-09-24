import os
from struct import Struct
import time
import glob
import json
import numpy as np
import sys

class Timer():
     def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

     def __enter__(self):
        self.start = time.time()
        return self

     def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(self.name, time.time() - self.start))
        return exc_type is None


def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


class ClusterDataset_pipeline(object):

    def __init__(self, cfg, is_test=False):
        feat_paths = cfg['feat_path']
        trk_vid_names = cfg['trk_vid_path']
        trk_frm_nums = cfg['trk_frm_num']
        spatem_paths = cfg['spatem_path']
        proposal_folders = cfg['proposal_folders']
        self.wo_weight = cfg.get('wo_weight', True)
        self.phase = cfg['phase']
        self._read(feat_paths, spatem_paths, proposal_folders, trk_vid_names, trk_frm_nums)

        print('#cluster: {}, wo_weight: {}'.format(self.size, self.wo_weight))
    
        
    def convert_keys_to_int(self, json_res):
        new_json_res = {}
        for key in json_res:
            key_res = json_res[key]
            key = int(key)
            new_json_res[key] = key_res
        return new_json_res

    
    def unpack_records(self, format, data):
        record_struct = Struct(format)
        return (record_struct.unpack_from(data, offset)
                for offset in range(0, len(data), record_struct.size))


    def _read(self, feat_paths, spatem_paths, proposal_folders, trk_vid_names, trk_frm_nums):
        if feat_paths.endswith('json'):
            with open(feat_paths, 'r') as f:
                features = json.load(f)
        elif feat_paths.endswith('b'):
            with open(feat_paths,'rb') as f:
                data = f.read()
            features = []
            for rec in self.unpack_records('f'*2048, data):
                features.append(rec)
        features = np.array(features)
        features = l2norm(features)
        self.features = features
        with open(spatem_paths, 'r') as f:
            spatem = json.load(f)
        self.spatem = np.array(spatem)
        with open(trk_vid_names, 'r') as f:
            tt1 = json.load(f)
            tt1 = self.convert_keys_to_int(tt1)
            self.trk_vid_name = tt1
        with open(trk_frm_nums, 'r') as f:
            tt2 = json.load(f)
            tt2 = self.convert_keys_to_int(tt2)
            self.trk_frm_num = tt2
        fn_node_pattern = '*_node.json'
        with Timer('read proposal list'):
            self.lst = []
            print('read proposals from folder: ', proposal_folders)
            fn_nodes = sorted(glob.glob(os.path.join(proposal_folders, fn_node_pattern)))
            assert len(fn_nodes) > 0, 'files under {} is 0'.format(proposal_folders)
            num_nodes = 0
            for fn_node in fn_nodes:
                node_name = fn_node.split('/')[-1].split('_')[0]
                self.lst.append(fn_node)
                num_nodes += 1
            self.size = len(self.lst)
    
    def __len__(self):
        return self.size

