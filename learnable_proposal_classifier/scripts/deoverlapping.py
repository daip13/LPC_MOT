from multiprocessing import Pool
import heapq
from bisect import bisect_left
import numpy as np
import yaml, glob
import argparse
import random 
import os
import time
import sys
import json
import base64
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), "../proposal_generation/"))
from core_function import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='input sv path', default = '')
    parser.add_argument('--output_path', type=str, help='output path', default ='')
    parser.add_argument('--proposal_path', type=str, help='proposal path', default ='')
    parser.add_argument('--GCN_output_file', type=str, help='purity classification results', default ='')
    parser.add_argument('--tracklet_id_file', type=str, help='', default ='')
    parser.add_argument('--prefix', type=str, help='prefix of input pb files', default='')
    parser.add_argument('--num_proc', type=int, help='number of processors', default=8)
    parser.add_argument('--deoverlapping_threshold', type=float, help='the threshold for deoverlapping', default =0.50)
    parser.add_argument('--weight_param', type=float, help='the weighting parameter for purity score', default =1)
    args = parser.parse_args()
    body_pb_files = glob.glob(args.input_path + '/' + args.prefix + '*.pb')
    if len(body_pb_files) > 0:
        fn_node_pattern = '*.json'
        fn_nodes_eval = sorted(glob.glob(os.path.join(args.proposal_path, fn_node_pattern)))
        proposals_output = {}
        with open(args.GCN_output_file, 'r') as f:
            estimated_IoP_dict = json.load(f)
        with open(args.tracklet_id_file, 'r') as f:
            tracklet_id_transfer = json.load(f)
        tracklet_id_ts = {v:k for k, v in tracklet_id_transfer.items()}
        assert len(fn_nodes_eval) == len(estimated_IoP_dict)
        for fn_node in fn_nodes_eval:
            with open(fn_node, 'r') as f:
                proposal_i = json.load(f)
            video_name_all = [tracklet_id_ts[trk].split('_')[0] for trk in proposal_i]
            prop_name = [int(tracklet_id_ts[trk].split('_')[1]) for trk in proposal_i]
            assert len(set(video_name_all)) == 1
            video_name = video_name_all[0]
            proposals_output.setdefault(video_name, {})
            proposals_output[video_name].setdefault("proposals", [])
            proposals_output[video_name].setdefault("IoU", [])
            proposals_output[video_name].setdefault("IoP", [])
            proposals_output[video_name]["proposals"].append(prop_name)
            proposals_output[video_name]["IoP"].append(estimated_IoP_dict[fn_node.split('/')[-1]])
        
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        if args.num_proc > 1:
            p = Pool(args.num_proc)
            p.map(deoverlap_one_video, [ (video, prop_i, body_pb_files, args) for video, prop_i in proposals_output.items()])
            p.close()
            p.join()
        else:
            for video, prop_i in proposals_output.items():
                deoverlap_one_video((video, prop_i, body_pb_files, args))
               

    print('finished')

def deoverlap_one_video(context):
    try:
        video, prop_i, body_pb_files, args = context
        print('proceesing ' + video)
        input_pb_files = glob.glob(args.input_path + '/' + args.prefix + '*.pb')
        pb_files_all = []
        for pb_file in body_pb_files:
            vid_name = os.path.basename(pb_file).split('.')[0]
            if vid_name == video:
                pb_files_all.append(pb_file)
        _, tracklet_all, _, _  = load_sv_pb_result_from_sv_pb_file(pb_files_all[0])
        for tracklet_id in tracklet_all.keys():
            prop_i['proposals'].append([tracklet_id])
            prop_i['IoP'].append(1.0)
        for proposal in prop_i['proposals']:
            frame_all = []
            for tid in proposal:
                frame_all += tracklet_all[tid][1]
            prop_i['IoU'].append(float(max(frame_all) - min(frame_all)) / 200.0)
        cameraId = video.split('_')[0]
        tracklet_label_output_ini, proposal_out, tracklet_IoP_perfect = deoverlap_IoU_IoP_combine_threshold(prop_i['proposals'],
                                                                                                  prop_i['IoU'],
                                                                                                  prop_i['IoP'], args.weight_param,
                                                                                                  args.deoverlapping_threshold,
                                                                                                  tracklet_all)
        merge_id_map = {}
        for proposals in proposal_out:
            track_id_assign = 'single_view_track_' + str(proposals[0])
            tracklet_id_assign = proposals[0]
            for tid in proposals:
                track_id = 'single_view_track_' + str(tid)
                merge_id_map[track_id] = [tracklet_id_assign, track_id_assign]
        for input_pb_file in input_pb_files:
            video_name = input_pb_file.split('/')[-1].split('.')[0]
            if video_name == video:
                cur_track_path = input_pb_file
                output_path = os.path.join(args.output_path, os.path.basename(cur_track_path))
                save_merge(cur_track_path, merge_id_map, output_path)
    except Exception as e:
        print(e)
        sys.exit()


if __name__ == "__main__":
    main()
