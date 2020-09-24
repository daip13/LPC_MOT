from multiprocessing import Pool
from struct import Struct
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
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), "../proposal_generation/"))
from core_function import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fe_body_track_folder', type=str, help='fe_body_track_folder', default = '')
    parser.add_argument('--proposal_file', type=str, help='proposal path', default ='')
    parser.add_argument('--output_path', type=str, help='video path', default ='')
    parser.add_argument('--prefix', type=str, help='prefix of input pb files', default='')
    args = parser.parse_args()
    proposal_files = glob.glob(args.proposal_file + '/' + args.prefix + '*proposals.json')
    vide_name_all = [os.path.basename(proposal_file).split('_proposals.json')[0] for proposal_file in proposal_files]
    vide_name_all = sorted(vide_name_all)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(args.output_path + 'eval1/'):
        os.makedirs(args.output_path+'eval1/')
    generate_one_part((vide_name_all, args))


def generate_one_part(context):
    try:
        vide_name_all, args = context
        body_pb_files = glob.glob(args.fe_body_track_folder + '/' + args.prefix + '*pb')
        tracklet_id_transfer, tracklet_features, tracklet_temporal_spatial, tracklet_frame_num, tracklet_video_name = {}, [], [], {}, {}
        for i, video_name in enumerate(tqdm(vide_name_all)):
            tracklet_id_transfer, tracklet_features, tracklet_temporal_spatial, tracklet_frame_num, tracklet_video_name = SV_one_video((video_name, body_pb_files, args), tracklet_id_transfer, tracklet_features, tracklet_temporal_spatial, tracklet_frame_num, tracklet_video_name)
        output_json_file = os.path.join(args.output_path, 'tracklet_id_transfer.json')
        with open(output_json_file, 'w') as f1:
            json.dump(tracklet_id_transfer, f1)
        output_app_file = os.path.join(args.output_path, 'features_total.b')
        with open(output_app_file, 'wb') as f:
            write_records(tracklet_features, 'f'*len(tracklet_features[0]), f)
        
        output_knns_file = os.path.join(args.output_path, 'tracklet_temporal_spatial.json')
        with open(output_knns_file, 'w') as f3:
            json.dump(tracklet_temporal_spatial, f3)
        
        opath_frnm = os.path.join(args.output_path, 'tracklet_frame_num.json')
        opath_vdnm = os.path.join(args.output_path, 'tracklet_video_name.json')
        with open(opath_frnm, 'w') as f:
            json.dump(tracklet_frame_num,f)
        with open(opath_vdnm, 'w') as f:
            json.dump(tracklet_video_name, f)

     
        num_proposals = 0
        GT_IoP_all = {}
        for video_name in vide_name_all:
            proposal_files = glob.glob(args.proposal_file + video_name + '*proposals.json')
            assert len(proposal_files) == 1
            proposal_file = proposal_files[0]
            with open(proposal_file, 'r') as f:
                proposal_video = json.load(f)
            used_dp = []
            for proposal in proposal_video.values():
                tracklet_total = proposal["proposals"]
                #iop = proposal["IoP"]
                iop = -1
                nodes = []
                for tracklet_id in tracklet_total:
                    new_tracklet_id = video_name + '_' + str(tracklet_id)
                    nodes.append(tracklet_id_transfer[new_tracklet_id])
                if nodes not in used_dp:
                    if len(nodes) > 1:
                        assert iop == -1
                        ofolder = args.output_path + 'eval1/'
                        opath_node = os.path.join(ofolder, '{}_node.json'.format(num_proposals))
                        GT_IoP_all[num_proposals] = iop
                        with open(opath_node, 'w') as f:
                            json.dump(nodes,f)
                        num_proposals += 1
                        used_dp.append(nodes)
                    else:
                        used_dp.append(nodes)
        iop_path = args.output_path + '/GT_IoP.json'
        with open(iop_path, 'w') as f:
            json.dump(GT_IoP_all,f)
        print('finished')
    except Exception as e:
        print(e)
        return

def write_records(records, format, f):
    '''
        Write a sequence of tuples to a binary file of structures.
    '''
    record_struct = Struct(format)
    for r in records:
        f.write(record_struct.pack(*r))

def read_records(format, f):
    record_struct = Struct(format)
    chunks = iter(lambda: f.read(record_struct.size), b'')
    return (record_struct.unpack(chunk) for chunk in chunks)

def unpack_records(format, data):
    record_struct = Struct(format)
    return (record_struct.unpack_from(data, offset)
            for offset in range(0, len(data), record_struct.size))


def SV_one_video(context, tracklet_id_transfer, tracklet_features, tracklet_temporal_spatial, tracklet_frame_num, tracklet_video_name):
    try:
        vid_validation, body_pb_files, args = context
        pb_files_all = []
        for pb_file in body_pb_files:
            vid_name = os.path.basename(pb_file).split('.')[0]
            if vid_name == vid_validation:
                pb_files_all.append(pb_file)
        assert len(pb_files_all) == 1
        _, tracklet_all, _, _ = load_sv_pb_result_from_sv_pb_file(pb_files_all[0])
        tracklet_order = []
        for key, value in tracklet_all.items():
            tracklet_order.append(key)
        num_old = len(tracklet_id_transfer)
        for i in range(len(tracklet_order)):
            key = tracklet_order[i]
            new_tracklet_id = vid_validation + '_' + str(key)
            assert new_tracklet_id not in tracklet_id_transfer
            tracklet_id_transfer[new_tracklet_id] = num_old + i
            tracklet_features.append(tracklet_all[key][0])
            frame_index = tracklet_all[key][1]
            assert sorted(frame_index) == frame_index
            start_frame_index = frame_index[0]
            end_frame_index = frame_index[-1]
            start_bbx = tracklet_all[key][2][0]
            end_bbx = tracklet_all[key][2][-1]
            temporal_spatial = [start_frame_index, end_frame_index, start_bbx[0], start_bbx[1], end_bbx[0], end_bbx[1], start_bbx[2], start_bbx[3], end_bbx[2], end_bbx[3]]
            tracklet_temporal_spatial.append(temporal_spatial)
            tracklet_frame_num[num_old + i] = int(len(tracklet_all[key][1]))
            tracklet_video_name[num_old + i] = vid_validation
    

        return tracklet_id_transfer, tracklet_features, tracklet_temporal_spatial, tracklet_frame_num, tracklet_video_name
    except Exception as e:
        print(e)
        sys.exit()


if __name__ == "__main__":
    main()
