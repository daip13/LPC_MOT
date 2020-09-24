# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import glob
import os
import os.path as osp
import sys
import json
from multiprocessing import Pool, cpu_count
import pdb
# pdb.set_trace()

import cv2
import numpy as np
from tqdm import tqdm
from torch.backends import cudnn

# sys.path.append('..')
sys.path.insert(0, '../')
import fastreid
from fastreid.config import get_cfg
from fastreid.utils.file_io import PathManager
from predictor import FeatureExtractionDemo

# import some modules added in project like this below
# from projects.PartialReID.partialreid import *

cudnn.benchmark = True

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def load_json(filename):
    import json
    with open(filename, 'r') as rf:
        data = json.load(rf)
    return data

def mkdir(dirname):
    if not osp.exists(dirname):
        os.makedirs(dirname)

def save_json(filename, data):
    mkdir(osp.dirname(filename))
    with open(filename, 'w') as wf:
        json.dump(data, wf)
    print('save json to ' + filename)

def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
        default='../configs/MOT-Strongerbaseline.yml'
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input_video_dir", default='/ssd/zphe/data/MOT17_videos//'
    )
    parser.add_argument(
        "--input_det_dir", default='/ssd/zphe/data/MOT17_det_tracktor_json/'
    )
    parser.add_argument(
        "--det_type", default='DPM'
    )
    parser.add_argument(
        "--output",
        default='/ssd/zphe/data/MOT20_R50-ibn_results',
        help='path to save features'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=8
    )
    return parser


# if __name__ == '__main__':
args = get_parser().parse_args()
cfg = setup_cfg(args)
videos = glob.glob(osp.join(args.input_video_dir, '*.mp4'))
demo = FeatureExtractionDemo(cfg, parallel=args.parallel)
# def process_one_video(params):
for video in tqdm(videos):
    # device = torch.device("cuda:%d" % gpu_id) if gpu_id>=0 else torch.device("cpu")
    # demo.predictor.model.to(device)
    videoname = osp.basename(video)
    print('processing ' + videoname)
    jsonfile = osp.join(args.input_det_dir, args.det_type, videoname + '.final.reduced.json')
    if not osp.exists(jsonfile):
        print(jsonfile + ' not exists!!!')
        continue
        # return
    det_results = load_json(jsonfile)
    vidcap = cv2.VideoCapture(video)
    save_results = {}
    frames = list(det_results.keys())
    frames = set(sorted([int(x) for x in frames]))
    max_frame = max(list(frames))
    frame_id = 1
    ret = True
    if vidcap.isOpened():
        while ret:
            ret, frame = vidcap.read()
            if frame_id in frames:
                bboxes = det_results[str(frame_id)]
                save_results[str(frame_id)] = []
                for bbox in bboxes:
                    box = bbox[0]
                    pid = bbox[1]
                    box_i = [int(x) for x in box]
                    x,y,w,h = box_i
                    img = frame[y:y+h, x:x+w:, ]
                    # img = img.to(device)
                    feat = demo.run_on_image(img)[0].numpy().tolist()
                    save_results[str(frame_id)].append([box, pid, feat])
            frame_id += 1
            if frame_id > max_frame:
                break
    #save_json(osp.join(args.output, args.det_type, videoname + '.final.reduced.json'), save_results)

# cores = args.cores
# if cores <= 0:
#     cores = cpu_count()
# num_of_process = int(min(cores, len(videos)))
# mp_pool = Pool(cores)
# multi_inputs = []
# for i in range(num_of_process):
#     param = (videos[i], i)
#     multi_inputs.append(param)
# mp_pool.map(process_one_video, param)
