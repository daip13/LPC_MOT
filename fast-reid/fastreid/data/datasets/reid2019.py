# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class REID2019(ImageDataset):
    _junk_pids = [0, -1]
    data_dir = 'NAIC/1/A'
    dataset_name = "REID2019"
    def __init__(self, root='', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.data_dir, self.dataset_name)
        if not osp.isdir(self.dataset_dir):
            warnings.warn('The current data dir:%s is not a dir.' % self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir)
        required_files = [
            self.train_dir,
        ]
        train = self.process_dir(self.train_dir)
        query = []
        gallery = []
        super(REID2019, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        data = []
        num = 0
        for pid in os.listdir(dir_path):
            for img in glob.glob(osp.join(dir_path, pid, '*.png')):
                data.append([img, pid, num])
                num += 1
        return data
