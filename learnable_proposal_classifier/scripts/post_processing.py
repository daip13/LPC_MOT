from multiprocessing import Pool
import heapq
import numpy as np
import yaml, glob
import argparse
import random 
import os
import sys
import json
import base64
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), "../proposal_generation/"))
from core_function import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, help='output path', default ='')
    parser.add_argument('--input_path', type=str, help='input path', default ='')
    parser.add_argument('--min_track_length_filter', type=int, default=2)
    args = parser.parse_args()
    run(args)

def run(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    body_pb_files = glob.glob(args.input_path + '/*pb')
    body_pb_files = sorted(body_pb_files)
    for pb_file in body_pb_files:
        track_res, _, _, _ = load_sv_pb_result_from_sv_pb_file(pb_file, 0)
        track_res_smoothness = post_processing_trajectory_smoothness(track_res, args.min_track_length_filter)
        sequence_name = os.path.basename(pb_file).split('.')[0]
        output_json_file = os.path.join(args.output_path, sequence_name + '.mp4.cut.mp4.final.reduced.json')
        output_txt_file = os.path.join(args.output_path, sequence_name + '.txt')
        with open(output_json_file, 'w') as f:
            json.dump(track_res_smoothness, f)
        track_res_smoothness_sorted = {k:track_res_smoothness[k] for k in sorted(track_res_smoothness.keys())}
        for frame, bbxs in track_res_smoothness_sorted.items():
            bbxs = sorted(bbxs, key=lambda x:x[1])
            for bbx in bbxs:
                track_id = bbx[1]
                bb_left = bbx[0][0]
                bb_top = bbx[0][1]
                bb_width = bbx[0][2]
                bb_height = bbx[0][3]
                track_info = [frame, track_id, bb_left, bb_top, bb_width, bb_height, -1, -1, -1]
                with open(output_txt_file, 'a') as f1:
                    for info in track_info:
                        f1.write(str(info) + ",")
                    f1.write(str(-1) + '\n')

if __name__ == "__main__":
    main()
