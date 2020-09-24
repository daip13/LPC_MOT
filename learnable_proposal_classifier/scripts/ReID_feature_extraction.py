import argparse
import glob
import os
import os.path as osp
import sys
import json
from multiprocessing import Pool, cpu_count
import pdb
import re
import cv2
import numpy as np
from tqdm import tqdm
from torch.backends import cudnn

# sys.path.append('..')
sys.path.append(os.path.join(os.path.dirname(__file__), "../proto/"))
import detection_results_pb2

sys.path.insert(0, '/root/LPC_MOT/fast-reid/')
import fastreid
from fastreid.config import get_cfg
from fastreid.utils.file_io import PathManager
from demo.predictor import FeatureExtractionDemo

cudnn.benchmark = True

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def load_det_txt(txt_file):
    res = {}
    with open(txt_file, 'r+') as f1:
        for line in f1:
            infos = line.rstrip()
            infos = infos.split(',')
            frame = int(infos[0])
            bbx = [float(infos[2]), float(infos[3]), float(infos[4]), float(infos[5])]
            det_score = float(infos[6])
            res.setdefault(frame, []).append([bbx, det_score])
    return res

def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
        default='/root/LPC_MOT/fast-reid/configs/MOT-Strongerbaseline.yml'
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--dataset_dir", default='/root/LPC_MOT/dataset/MOT17/'
    )
    parser.add_argument(
        "--input_video_dir", default='/root/LPC_MOT/dataset/MOT17/videos/'
    )
    parser.add_argument(
        "--output_dir",
        default='/root/LPC_MOT/dataset/MOT17/detection_reid_with_traindata/',
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

phases = ['train', 'test']
detectors = ['SDP', 'DPM', 'FRCNN']
train_sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
test_sequences = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']

if __name__ == '__main__':
    args = get_parser().parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    sequences = []
    videos = glob.glob(osp.join(args.input_video_dir, '*.mp4'))
    for phase in phases:
        if phase == 'train':
            for sequence in train_sequences:
                for detector in detectors:
                    sequences.append(os.path.join(args.dataset_dir, phase, sequence + '-' + detector))
        elif phase == 'test':
            for sequence in test_sequences:
                for detector in detectors:
                    sequences.append(os.path.join(args.dataset_dir, phase, sequence + '-' + detector))
    cfg = setup_cfg(args)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)
    for sequence in sequences:
        print('processing ' + os.path.basename(sequence))
        det_num = 1
        save_results = []
        det_file = os.path.join(sequence, 'det', 'tracktor_prepr_det.txt')
        det_res = load_det_txt(det_file)
        videoname = os.path.basename(sequence).split('-')[0] + '-' + os.path.basename(sequence).split('-')[1]
        video = glob.glob(os.path.join(args.input_video_dir, videoname + "*.mp4"))[0]
        vidcap = cv2.VideoCapture(video)
        ret = True
        frames = list(det_res.keys())
        frames = set(sorted([int(x) for x in frames]))
        max_frame = max(list(frames))
        frame_id = 1
        ret = True
        if vidcap.isOpened():
            while ret:
                ret, frame = vidcap.read()
                if frame_id in frames:
                    print('processing frame' + str(frame_id))
                    bboxes = det_res[frame_id]
                    for bbox in bboxes:
                        box = bbox[0]
                        det_score = bbox[1]
                        box_i = [int(x) for x in box]
                        x,y,w,h = box_i
                        # skip the case that the box is not in the image
                        if x + w < 0 or x > frame.shape[1] or y > frame.shape[0] or y + h < 0:
                            continue
                        img = frame[max(y, 0):min(y+h, frame.shape[0]), max(x, 0):min(x+w, frame.shape[1]):, ]
                        img = frame[y:y+h, x:x+w:, ]
                        # img = img.to(device)
                        feat = demo.run_on_image(img)[0].numpy().tolist()
                        save_results.append([det_num, frame_id, box, det_score, feat])
                        det_num += 1
                frame_id += 1
                if frame_id > max_frame:
                    break
        output_file = os.path.join(args.output_dir, os.path.basename(sequence) + '.pb')
        detections_pb = detection_results_pb2.Detections()
        for detection in save_results:
            det_num, frame, box, det_score, feat = detection
            _detection = detections_pb.tracked_detections.add()
            _detection.frame_index = frame
            _detection.detection_id = det_num
            _detection.box_x = int(box[0])
            _detection.box_y = int(box[1])
            _detection.box_width = int(box[2])
            _detection.box_height = int(box[3])
            tf = _detection.features.features.add()
            for d in feat:
                tf.feats.append(d)
        with open(output_file, 'wb') as f:
            f.write(detections_pb.SerializeToString())
    print('finished')
