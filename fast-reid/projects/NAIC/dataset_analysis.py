'''
analysis color distribution of NAIC datasets
---------------
Author: Yang Qian
Email: yqian@aibee.com
'''

import numpy as np
import cv2
import os

if __name__ == "__main__":
    dataset_dir = 'D:\\document\\NAIC\\NAIC2019\\1\\train_set'

    im_names = os.listdir(dataset_dir)
    axis = [0, 1, 2]
    for im_name in im_names:
        im = cv2.imread(os.path.join(dataset_dir, im_name))
        np.random.shuffle(axis)
        im_new = im[:, :, axis]
        cv2.imshow('original', im)
        cv2.imshow('new', im_new)
        cv2.waitKey()