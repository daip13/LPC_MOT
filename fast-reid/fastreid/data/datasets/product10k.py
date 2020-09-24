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
class Product10k(ImageDataset):
    _junk_pids = [-1]
    dataset_dir = 'Product10k'
    dataset_name = "Product10k"

    def __init__(self, root='./datasets', split_file="", extra_data_cfg=None, **kwargs):
 
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.data_dir = self.dataset_dir
        self.train_dir = osp.join(self.data_dir,  'train')
        self.query_dir = osp.join(self.data_dir,  'test')
        self.gallery_dir = osp.join(self.data_dir, 'test')
        self.train_label = osp.join(self.data_dir, 'train.csv')
        self.test_label = osp.join(self.data_dir, 'test.csv')
        self.split_file = osp.join(self.data_dir,  split_file) if split_file else None
        self.camid = 0
        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, label=self.train_label, split_file=self.split_file, train_or_val='train')
        if self.split_file:
            query = self.process_dir(self.train_dir, split_file=self.split_file, train_or_val='query')
            gallery = self.process_dir(self.train_dir, split_file=self.split_file, train_or_val='gallery')
        else:
            query = self.process_dir(self.query_dir)
            gallery = self.process_dir(self.gallery_dir)
        super(Product10k, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, label=None, split_file=None, train_or_val='train'):
        data = []
        if split_file:
            with open(split_file, 'r') as f:
                train_val_map = {'0': 'train', '1': 'query', '2': 'gallery'}
                for line in f.readlines():
                    im_name, sku, train_val = line.strip().split(' ')
                    if train_val_map[train_val] == train_or_val:
                        data.append((osp.join(dir_path, im_name), sku, self.camid))
                        self.camid += 1
        elif label:
            with open(label, 'r') as f:
                for idx, line in enumerate(f.readlines()):
                    if idx == 0:
                        continue
                    im_name, sku, group = line.strip().split(',')
                    data.append((osp.join(dir_path, im_name), sku, self.camid))
                    self.camid += 1
        else:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
            for img_path in img_paths:
                data.append((img_path, '-1', -1))
        return data