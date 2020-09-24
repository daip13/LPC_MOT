from multiprocessing import Pool
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../svonline/"))
import json
import glob
import time
from core.utils import *
from core.node import *
from core.data_types import FeatureTypes, IDContainer
from core.graph import Graph
from core.creator import *
sys.path.append(os.path.join(os.path.dirname(__file__), "../proto/"))
import online_tracking_results_pb2, detection_results_pb2
import argparse

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='configuration file', required=True)
    parser.add_argument('--detection_file_path', type=str, help='video detection pb file',
        required=True)
    parser.add_argument('--prefix', type=str, default='MOT')
    parser.add_argument('--output_path', type=str, required=True,
            help='the path to save tracking results')
    parser.add_argument('--start_frame', type=int, default=0, help='the number of frames to process')
    parser.add_argument('--num_proc', type=int, default=1, help='number of processors')
    parser.add_argument('--num_frames_process', type=int, default=10000000, help='the number of frames to process')
    parser.add_argument('--do_augmentation', action="store_true", help="if do_augmentation, filter detection randomly")
    return parser.parse_args()

def load_detections_to_nodes(detection_file, camera_name, confidence_threshold,
                    start_frame=-1, end_frame=100000000, do_augmentation=False, skip_frames = 1, models=None):
    mlog.info("Loading detection file {}".format(detection_file))
    det_nodes = []

    _, ext = os.path.splitext(detection_file)
    if ext == '.pb':
        detections_pb = detection_results_pb2.Detections()
        with open(detection_file, 'rb') as f:
            detections_pb.ParseFromString(f.read())
        detections_from_file = {}
        for detection in detections_pb.tracked_detections:
            frame = detection.frame_index
            detection_id = detection.detection_id
            box = [detection.box_x, detection.box_y, detection.box_width, detection.box_height]
            feat = [ d for d in detection.features.features[0].feats ]
            assert len(feat) == 2048
            detections_from_file.setdefault(frame, []).append([box, detection_id, feat])
    elif ext == '.json':
        detections_from_file = json.load(open(detection_file, 'r'))
        detections_from_file = {int(k):v for k, v in detections_from_file.items()}
    else:
        raise Exception('unknown detection format {0}'.format(ext))
    if do_augmentation:
        detections_all = [[k, v] for k, vs in detections_from_file.items() for v in vs]
        remove_ratio = random.uniform(5, 20)/ 100.0
        remove_num = int(len(detections_all) * remove_ratio)
        remove_index = [random.randint(0, len(detections_all)-1) for _ in range(remove_num)]
        detections_left = [detections_all[ii] for ii in range(len(detections_all)) if ii not in remove_index]
        mlog.info("detection num reduce from {} -> {} after randomly drop".format(len(detections_all), len(detections_left)))
        detections_from_file = {}
        for det in detections_left:
            frame, box = det[0], det[1]
            detections_from_file.setdefault(frame, []).append(box)

    for frame_index in sorted(detections_from_file.keys()):
        if start_frame <= frame_index < end_frame and frame_index % skip_frames == 0:
            humans = detections_from_file[frame_index]
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

def generate_sv_single_video(context):
    try:
        detection_file, args = context
        output_pb_file = os.path.join(args.output_path, os.path.basename(detection_file).split('.')[0] + '.mp4.cut.pb')
        output_json_file = os.path.join(args.output_path, os.path.basename(detection_file).split('.')[0] + '.mp4.cut.mp4.final.reduced.json')
        cname = "single_view"
        moving_cameras = ["MOT17-05", "MOT17-06", "MOT17-07", "MOT17-10", "MOT17-11", "MOT17-12", "MOT17-13", "MOT17-14"]
        ############################################################
        # load and set configs
        video_name = os.path.basename(detection_file).split('-')[0] +  '-' + os.path.basename(detection_file).split('-')[1]
        #video_name = os.path.basename(detection_file).split('_')[0]
        if video_name in moving_cameras:
            config_file = os.path.join(args.config_path, "single_view_online_moving.json")
        else:
            config_file = os.path.join(args.config_path, "single_view_online_static.json")
        with open(config_file, 'r') as f:
            configs = json.loads(f.read())
        ############################################################
        # load data
        nodes = load_detections_to_nodes(detection_file,
                cname,
                configs["detection_confidence"],
                args.start_frame,
                args.start_frame + args.num_frames_process - 1, args.do_augmentation)

        opts = {} 
        graph = Graph(nodes,
                create_affinities(configs["affinity"], opts),
                cname)
        
        engine = create_algorithm(configs["algorithm"])

        output = engine(graph, cname)
        save_nodes_online_pbs(output, cname, output_pb_file)
        save_nodes_to_json(output, cname, output_json_file)
    except Exception as e:
        mlog.info(e)
        sys.exit()


if __name__ == "__main__":
    args = parse_arguments()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    #detection_files = glob.glob(os.path.join(args.detection_file_path, args.prefix + "*.json"))
    detection_files = glob.glob(os.path.join(args.detection_file_path, args.prefix + "*.pb"))
    if args.num_proc == 1:
        for detection_file in detection_files:
            generate_sv_single_video((detection_file, args))
    else:
        p = Pool(args.num_proc)
        p.map(generate_sv_single_video, [(detection_file, args) for i, detection_file in enumerate(detection_files)])
        p.close()
        p.join()
    print('all finished')


