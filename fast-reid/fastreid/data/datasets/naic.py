# encoding: utf-8
"""
@author:  Yang Qian
@contact: yqian@aibee.com
"""

import glob
import os.path as osp
import re
import warnings
import numpy as np

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class NAIC(ImageDataset):
    """NAIC.

    URL: `<https://naic.pcl.ac.cn/frame/3>`_

    """
    _junk_pids = [-1]
    dataset_dir = ''
    dataset_name = "NAIC"

    def __init__(self, root='datasets', stage='1A', split_file="", extra_data_cfg=None, **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'NAIC')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "1/A" under '
                          '"NAIC".')

        stage_dir = osp.join(stage[0], stage[1])
        self.train_dir = osp.join(self.data_dir, stage_dir, 'train/images')
        self.query_dir = osp.join(self.data_dir, stage_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, stage_dir, 'gallery')
        self.train_label = osp.join(self.data_dir, stage_dir, 'train/label.txt')
        self.split_file = osp.join(self.data_dir, stage_dir, split_file) if split_file else None

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, label=self.train_label, split_file=self.split_file, train_or_val='train')
        if self.split_file:
            query = self.process_dir(self.train_dir, label=self.train_label, split_file=self.split_file, train_or_val='query')
            gallery = self.process_dir(self.train_dir, label=self.train_label, split_file=self.split_file, train_or_val='gallery')
        else:
            query = self.process_dir(self.query_dir)
            gallery = self.process_dir(self.gallery_dir)

        pids_cur = set(((np.array(train)[:, 1]).astype(np.int)).tolist())
        pids_start = len(pids_cur)
        print('PIDs of original datasets: {}'.format(len(pids_cur)))
        camid = len(train) + 1
        if extra_data_cfg is not None and extra_data_cfg.ROOTS[0]:
            for i in range(len(extra_data_cfg.ROOTS)):
                extra_data_root = extra_data_cfg.ROOTS[i]
                extra_data_list = extra_data_cfg.LISTS[i]
                print('loading data from {}'.format(osp.join(extra_data_root, extra_data_list)))
                pids_start += 1e4
                pids = []
                with open(osp.join(extra_data_root, extra_data_list), 'r') as f:
                    for line in f.readlines():
                        im_name, pid = line.strip().split(' ')
                        if osp.exists(osp.join(extra_data_root, im_name)):
                            train.append([osp.join(extra_data_root, im_name), int(pid)+pids_start, camid])
                        else:
                            print('{} do not exists'.format(osp.join(extra_data_root, im_name)))
                        camid += 1
                        pids.append(int(pid)+pids_start)
                print('PIDs of {}: {}'.format(osp.join(extra_data_root, extra_data_list), len(set(pids))))
                assert len(set(pids) & pids_cur) == 0, 'wrong pid assignment: {}'.format(set(pids) & set(pids_cur))
                pids_cur |= set(pids)


        super(NAIC, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, label=None, split_file=None, train_or_val='train'):
        data = []
        if split_file:
            with open(split_file, 'r') as f:
                camid = 0
                train_val_map = {'0': 'train', '1': 'query', '2': 'gallery'}
                for line in f.readlines():
                    im_name, pid, train_val = line.strip().split(' ')
                    if train_val_map[train_val] == train_or_val:
                        data.append((osp.join(dir_path, im_name), pid, camid))
                        camid += 1
        elif label:
            with open(label, 'r') as f:
                camid = 0
                for line in f.readlines():
                    im_name, pid = line.strip().split(':')
                    data.append((osp.join(dir_path, im_name), pid, camid))
                    camid += 1
        else:
            img_paths = glob.glob(osp.join(dir_path, '*.png'))
            for img_path in img_paths:
                data.append((img_path, '-1', -1))
        return data