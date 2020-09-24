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
import copy
from collections import Counter
sys.path.append(os.path.join(os.path.dirname(__file__), "../proposal_generation/"))
from core_function import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fe_body_track_folder', type=str, help='fe_body_track_folder', default='')
    parser.add_argument('--gth_label_folder', type=str, help='gth_label_folder', default='')
    parser.add_argument('--output_path', type=str, help='output path', default='')
    parser.add_argument('--iter_max', type=int, help='maximum iteration number', default=10)
    parser.add_argument('--th_step', type=float, help='th_step', default=0.05)
    parser.add_argument('--num_proc', type=int, help='number of processor', default=8)
    parser.add_argument('--max_size', type=int, help='maximum cluster size', default=2)
    parser.add_argument('--knum', type=int, help='k number for graph construction', default=3)
    parser.add_argument('--time_thd', type=int, help='start time threshold', default=20)
    parser.add_argument('--time_thd_max', type=int, help='end time threshold', default=80)
    parser.add_argument('--time_sigma', type=int, help='time sigma', default=40)
    parser.add_argument('--distance_thd_2d', type=int, help='start distance threshold', default=30.0)
    parser.add_argument('--distance_thd_2d_max', type=int, help='end distance threshold', default=200.0)
    parser.add_argument('--distance_sigma', type=int, help='distance sigma', default=100.0)
    parser.add_argument('--app_thd', type=float, help='start appearance threshold', default=0.90)
    parser.add_argument('--app_thd_min', type=float, help='end appearance threshold', default=0.00)
    parser.add_argument('--prefix', type=str, help='prefix of input pb files', default='')
    parser.add_argument('--use_gt', action="store_true", help="")
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    body_pb_files = glob.glob(args.fe_body_track_folder + '/' + args.prefix + '*.pb')
    mlog.info('total {} number of files for processing'.format(len(body_pb_files))) 
    start_time_all = time.time()
    if args.num_proc > 1:
        p = Pool(args.num_proc)
        p.map(SV_one_video, [(body_pb_file, args) for i, body_pb_file in enumerate(body_pb_files)])
        p.close()
        p.join()
    else:
        for i in range(len(body_pb_files)):
            body_pb_file = body_pb_files[i]
            SV_one_video((body_pb_file, args))
    end_time_all = time.time()
    mlog.info('Total time cost for proposal generation is {}'.format(end_time_all-start_time_all))
    print('finished')


def SV_one_video(context):
    try:
        body_pb_file, args = context
        mlog.info('processing file {}'.format(os.path.basename(body_pb_file)))
        moving_cameras = ["MOT17-05", "MOT17-06", "MOT17-07", "MOT17-10", "MOT17-11", "MOT17-12", "MOT17-13", "MOT17-14"]
        sequence_name = os.path.basename(body_pb_file).split('-')[0] + '-' + os.path.basename(body_pb_file).split('-')[1]
        if sequence_name in moving_cameras:
            args.time_thd = 10
            args.time_thd_max = 40
            args.distance_thd_2d =  50
            args.distance_thd_2d_max = 250
        else: 
            args.time_thd = 30
            args.time_thd_max = 120
            args.distance_thd_2d =  50
            args.distance_thd_2d_max = 350
        # this is to load the sv tracking inputs
        track_res, tracklet_all, _, _ = load_sv_pb_result_from_sv_pb_file(body_pb_file, 0)
        # load the gt files
        sv_tid_to_gth = None
        if args.use_gt:
            gth_json_files = glob.glob(args.gth_label_folder + '/' + sequence_name + '*json')
            try:
                assert len(gth_json_files) == 1
                gth_json_file = gth_json_files[0]
            except:
                mlog.info('something wrong with gt label files of {}: {}'.format(sequence_name, gth_json_files))
                return
            if gth_json_file.endswith('json'):
                with open(gth_json_file, 'r') as f:
                    gt_res = json.load(f)
                    gt_res = {int(k):v for k, v in gt_res.items()}
                sv_tid_to_gth, sv_gth_ret, _ = assign_matched_labels(track_res, gt_res)
            else:
                raise NotImplementedError
        # do proposal generation: initialization
        cluster_indoor = []
        proposal_ini = []
        for key1, value1 in tracklet_all.items():
            key = copy.deepcopy(key1)
            value = copy.deepcopy(value1)
            tt1 = set()
            tt1.add(key)
            proposal_ini.append(tt1)
            node1 = clusterNode(key, value)
            cluster_indoor.append(node1)
        # do proposal generation
        start_time = time.time()
        proposals_all, indoor_candidates = [], []
        proposals_all.append(proposal_ini)
        mlog.info('The number of initial clusters is {}'.format(len(cluster_indoor)))
        proposals_all, indoor_candidates = Super_vertex_based_proposal_generation(cluster_indoor, args, proposals_all, sv_tid_to_gth=sv_tid_to_gth)
        end_time = time.time()
        mlog.info('time cost for proposal generation is {}'.format(end_time-start_time))
        
        proposals_all1 = []
        for i in range(len(proposals_all)):
            for j in range(len(proposals_all[i])):
                propo_temp = sorted(list(proposals_all[i][j]))
                if propo_temp not in proposals_all1:
                    proposals_all1.append(list(propo_temp))
        proposals_all, IoP_all, IoU_all = [], [], []
        start = time.time()
        for proposal in proposals_all1:
            frame_index_all, bbx_all = [], []
            for tid in proposal:
                frame_index_all += tracklet_all[tid][1]
                bbx_all += tracklet_all[tid][2]
            frame_index_all, bbx_all = zip(*sorted(zip(frame_index_all, bbx_all)))
            if sv_tid_to_gth is not None:
                id_sets = set()
                id_lists = []
                for tid in proposal:
                    if tid in sv_tid_to_gth and sv_tid_to_gth[tid] != -1:
                        id_sets.add(sv_tid_to_gth[tid])
                        id_lists += sv_gth_ret[tid] 
                if id_sets:
                    if len(id_sets) == 1:
                        iop_label1 = 1
                    else:
                        id_count = Counter(id_lists)
                        gt_id, count = id_count.most_common()[0] 
                        iop_label1 = evaluate_purity_for_proposal(frame_index_all, bbx_all, gt_res, gt_id) 
                else:
                    iop_label1 = 1
            else:
                iop_label1 = -1
            proposals_all.append(proposal)
            IoP_all.append(iop_label1)
            IoU_all.append(float(len(frame_index_all)) / 2500.0)
        end = time.time()
        vid_validation = os.path.basename(body_pb_file).split('.')[0]
        if sv_tid_to_gth is not None:
            gth_to_tid = list(set([pid for tid, pid in sv_tid_to_gth.items() if pid != -1]))
            tids_with_gt = [tid for tid, pid in sv_tid_to_gth.items() if pid != -1]
            tracklet_label_output, proposal_out_eval, _ = deoverlap_IoU_IoP_combine_threshold(proposals_all, IoU_all, IoP_all, 100)
            pure_proposals, all_proposals, pt_proposals = 0, 0, 0
            track_id_merge = {}
            for i, proposal_i in enumerate(proposal_out_eval):
                id_sets = set()
                for tid in proposal_i:
                    track_id_merge[tid] = i
                    if tid in sv_tid_to_gth and sv_tid_to_gth[tid] != -1:
                        id_sets.add(sv_tid_to_gth[tid])
                if len(id_sets) <= 1:
                    pure_proposals += 1
                if len(id_sets) == 1:
                    pt_proposals += 1
                all_proposals += 1
            
            mlog.info('[Video: {}] [initial] NT is {}, PT is {}'.format(sequence_name, float(len(sv_tid_to_gth))/len(gth_to_tid), float(len(tids_with_gt))/len(gth_to_tid) )) 
            mlog.info('[Video: {}] precision is {} {}/{}, NT is {} {}/{}, PT is {} {}/{}'.format(sequence_name, float(pure_proposals)/all_proposals, pure_proposals, all_proposals, float(len(proposal_out_eval))/len(gth_to_tid), len(proposal_out_eval), len(gth_to_tid), float(pt_proposals)/len(gth_to_tid), pt_proposals, len(gth_to_tid)))
            track_res_merge = {}
            for frame, bbxs in track_res.items():
                track_res_merge.setdefault(frame, [])
                for bbx in bbxs:
                    box = bbx[0]
                    idx = bbx[1]
                    if idx in track_id_merge:
                        track_res_merge[frame].append([box, track_id_merge[idx]])
                    else:
                        mlog.info('[warning] {} is not in id map'.format(idx))
                        track_res_merge[frame].append(bbx)
            track_res_merge_file = os.path.join(args.output_path, vid_validation + '.mp4.cut.mp4.final.reduced.json')
            with open(track_res_merge_file, 'w') as f:
                json.dump(track_res_merge, f)
        # generate output
        output_json_file = args.output_path + vid_validation + '_proposals.json'
        proposals_output, num_proposals = {}, 0
        for ii in range(len(proposals_all)):
            proposals_output[num_proposals] = {}
            proposals_output[num_proposals]['proposals'] = list(proposals_all[ii])
            proposals_output[num_proposals]['IoP'] = IoP_all[ii]
            num_proposals += 1
        with open(output_json_file, 'w') as f1:
            json.dump(proposals_output, f1)
    except Exception as e:
        mlog.info(e)
        sys.exit()


if __name__ == "__main__":
    main()
