# encoding: utf-8
"""
@author:  He Zhangping
@contact: zphe@aibee.cn
"""

import glob
import os
import os.path as osp
import re
import random
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MOT17(ImageDataset):
    dataset_name = "MOT17"

    def __init__(self, root='/ssd/zphe/data/reid/MOT17_reid',  **kwargs):
        self.root = root
        self.data_dir = root

        self.train_dir = osp.join(self.data_dir, 'train')
        self.test_dir = osp.join(self.data_dir, 'test')

        required_files = [
            self.train_dir,
            self.test_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, train_or_val='train')
        query, gallery = self.process_dir(self.test_dir, train_or_val='test')
        super(MOT17, self).__init__(train, query, gallery, **kwargs)
        
    def process_dir(self, dir_path, train_or_val='train'):
        if train_or_val == 'train':
            train = []
            total_num = 0
            for pid in os.listdir(dir_path):
                if not osp.isdir(osp.join(dir_path, pid)):
                    continue
                imgs = glob.glob(osp.join(dir_path, pid, '*.jpg'))
                for img in imgs:
                    total_num += 1
                    cam_id = total_num
                    train.append([img, str(pid), cam_id])
            return train
        else:
            query = []
            gallery = []
            total_num = 0
            for pid in os.listdir(dir_path):
                if not osp.isdir(osp.join(dir_path, pid)):
                    continue
                imgs = glob.glob(osp.join(dir_path, pid, '*.jpg'))
                num = len(imgs)
                random.shuffle(imgs)
                for idx, img in enumerate(imgs):
                    total_num += 1
                    cam_id = total_num
                    if idx < (num//2):
                        query.append([img, str(pid), cam_id])
                    else:
                        gallery.append([img, str(pid), cam_id])
            return query, gallery