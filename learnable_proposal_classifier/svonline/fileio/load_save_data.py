import numpy as np
import os
import sys
import cv2
import json
import random
import copy
# set relative paths
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from core.utils import *
from core.node import *
from core.data_types import FeatureTypes, IDContainer
sys.path.append('/root/LPC_MOT/learnable_proposal_classifier/proto/')
import online_tracking_results_pb2

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)

def load_detections_from_json(detection_file, camera_name, confidence_threshold,
                    start_frame=-1, end_frame=100000000, do_augmentation=False, skip_frames = 1, models=None):
    mlog.info("Loading detection file {}".format(detection_file))
    det_nodes = []

    _, ext = os.path.splitext(detection_file)
    if ext == '.pb' or ext == '.det' or ext == '.filtering':
        mlog.info("Please implement JSON parser")
        raise NotImplementedError
    elif ext == '.json':
        detections_from_json = json.load(open(detection_file, 'r'))
        detections_from_json = {int(k):v for k, v in detections_from_json.items()}
        if do_augmentation:
            detections_all = [[k, v] for k, vs in detections_from_json.items() for v in vs]
            remove_ratio = random.uniform(5, 20)/ 100.0
            remove_num = int(len(detections_all) * remove_ratio)
            remove_index = [random.randint(0, len(detections_all)-1) for _ in range(remove_num)]
            detections_left = [detections_all[ii] for ii in range(len(detections_all)) if ii not in remove_index]
            mlog.info("detection num reduce from {} -> {} after randomly drop".format(len(detections_all), len(detections_left)))
            detections_from_json = {}
            for det in detections_left:
                frame, box = det[0], det[1]
                detections_from_json.setdefault(frame, []).append(box)

        for frame_index in sorted(detections_from_json.keys()):
            if start_frame <= frame_index < end_frame and frame_index % skip_frames == 0:
                humans = detections_from_json[frame_index]
                for i, human in enumerate(humans):
                    node = Node(len(det_nodes))
                    box = human[0]
                    # in case box dimension is too small
                    if box[2] < 4.0 or box[3] < 4.0:
                        continue

                    node.init(camera_name, int(frame_index / skip_frames), {
                        FeatureTypes.ID: IDContainer(),
                        FeatureTypes.Box2D: box,
                        FeatureTypes.OrigBox2D: box,
                        FeatureTypes.DetectionScore: 1.0,
                        FeatureTypes.ReID: human[2],
                    }, models=None)
                    det_nodes.append(node)
    else:
        raise Exception('unknown detection format {0}'.format(ext))
    mlog.info("loaded {} number of detections".format(len(det_nodes)))
    return det_nodes

def save_nodes_to_json(nodes, camera_view, output_file, skipped_frames=1):
    data = {}
    for node in nodes:
        tid = node.tid
        frames_ver = node[camera_view].timestamps
        frames = [int(frame) for frame in frames_ver]
        if len(frames) < 2:
            continue
        boxes = node[camera_view].feature(FeatureTypes.Box2D)
        for frame, box in zip(frames, boxes):
            data.setdefault(frame, [])
            data[frame].append([ box, tid, 0 ])
            # fill the skipped frames for viz
            for fill in range(max(0, frame - skipped_frames + 1), frame):
                data.setdefault(fill, [])
                data[fill].append([ box, tid, 0 ])
    with open(output_file, 'w') as fp:
        fp.write(json.dumps(data, indent=2))

def save_nodes_online_pbs(nodes, camera_view, tracking_file, result_id_type='tracklet_index'):
    tracks_from_pb = online_tracking_results_pb2.Tracks() 
    det_num_totals = 0
    for idx, node in enumerate(sorted(nodes, key = lambda x: x.mv_start)):
        tid = node.tid

        frames = node[camera_view].timestamps
        if len(frames) < 2:
            continue
        
        boxes = node[camera_view].feature(FeatureTypes.Box2D)
        app_feats = list(np.array(node[camera_view].feature(FeatureTypes.ReID), dtype=np.float32))
        track = tracks_from_pb.tracks.add()
        if app_feats:
            feat = np.mean(app_feats, axis = 0)
            del track.features.features[:]
            tf = track.features.features.add()
            for d in feat:
                tf.feats.append(d)

        track.tracklet_id = int(node.tid)
        track.track_id = 'single_view_track_' + str(node.tid)
        for frame, box, app in zip(frames, boxes, app_feats):
            detection = track.tracked_detections.add()
            detection.frame_index = frame
            detection.box_x = int(box[0])
            detection.box_y = int(box[1])
            detection.box_width = int(box[2])
            detection.box_height = int(box[3])
            det_num_totals += 1
    #cos_simi = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    print("det_num_totals is {}".format(det_num_totals))
    with open(tracking_file, 'wb') as f:
        f.write(tracks_from_pb.SerializeToString())

