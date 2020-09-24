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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_file_path', type=str, help='video detection pb file',
        required=True)
    parser.add_argument('--output_dir', type=str, required=True,
            help='the path to save tracking results')
    parser.add_argument('--det_type', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments() 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    sequences = glob.glob(os.path.join(args.detection_file_path, '*json'))
    for sequence in sequences:
        print('processing ' + os.path.basename(sequence))
        det_res = json.load(open(sequence, 'r'))
        sequence_id = os.path.basename(sequence).split('_')[0][-2:]

        output_file = os.path.join(args.output_dir, 'MOT17-' + sequence_id + '-' + args.det_type + '.pb')
        detections_pb = detection_results_pb2.Detections()
        det_num = 1
        for frame in det_res.keys():
            bboxes = det_res[frame]
            for bbox in bboxes:
                box, tt, feat = bbox
                _detection = detections_pb.tracked_detections.add()
                _detection.frame_index = int(frame)
                _detection.detection_id = det_num
                det_num += 1
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
