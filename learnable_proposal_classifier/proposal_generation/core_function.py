import numpy as np
import gc
import yaml, glob
import argparse
import random
import os
import time
import sys
from multiprocessing import Pool
import heapq
from bisect import bisect_left, bisect_right
from tqdm import tqdm
import json
import base64
import logging
import copy
from scipy.optimize import linear_sum_assignment
from collections import Counter
from cluster_nodes import *
from scipy.interpolate import interp1d
sys.path.append('/root/LPC_MOT/learnable_proposal_classifier/proto/')
import online_tracking_results_pb2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)


def load_pb(filename):
    mlog.info('loading {}'.format(filename))
    tracks_from_pb = online_tracking_results_pb2.Tracks()
    with open(filename, 'rb') as f:
        tracks_from_pb.ParseFromString(f.read())
    num_dets = sum( [ len(trk.tracked_detections) for trk in tracks_from_pb.tracks ] )
    mlog.info('there are {} number of detections'.format(num_dets))

    tracks = {}
    for track in tracks_from_pb.tracks:
        tracks.setdefault(track.track_id, []).append(track)
    return tracks


def iou(bbox, candidates, eps=0.00001):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:4]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:4].prod(axis=1)

    return area_intersection / (area_bbox + area_candidates - area_intersection+eps)

def evaluate_purity_for_proposal(frames, bbxes, gt_res, gt_id, overlap_threshold=0.1, max_impure_det=1):
    ret = []
    for ii, frame in enumerate(frames):
        if frame not in gt_res:
            ret.append(-1)
            continue
        gt_ids_this_frame = [bbx[1] for bbx in gt_res[frame]]
        if gt_id not in gt_ids_this_frame:
            ret.append(-1)
            continue
        lbox = np.array([bbx[0] for bbx in gt_res[frame] if bbx[1] == gt_id])
        tbox = bbxes[ii]
        ov = iou(np.array(tbox), lbox)
        if ov.max() >= overlap_threshold:
            ret.append(1)
        else:
            ret.append(-1)
    id_count = Counter(ret)
    indee, count = id_count.most_common()[0]
    if indee == 1 and (len(ret) - count) <= max_impure_det:
        return 1
    else:
        return 0

def generate_sv_data_augmentation(det_gt_group, detections, remove_ratio=0.05):
    sv_tid_to_gth, sv_gth_ret  = {}, {}
    max_frame_gap, iou_threshold, reid_threshold = 3, 0.4, 0.6
    detection_id_all = sorted(detections.keys())
    ## do data augmentation, randomly remove some detections
    remove_num = int(len(detections) * remove_ratio)
    remove_index = [random.randint(0, len(detection_id_all)-1) for _ in range(remove_num)]
    detection_id_all1 = [detection_id_all[ii] for ii in range(len(detection_id_all)) if ii not in remove_index]
    groups, pids = [], []
    for pid in det_gt_group:
        det_ids = sorted(det_gt_group[pid])
        det_ids_left = sorted([det for det in det_ids if det in detection_id_all1])
        group_cache = []
        for det in det_ids_left:
            if not group_cache:
                group_cache.append(det)
            else:
                det_old = group_cache[-1]
                frame_gap = abs(detections[det]['frame'] - detections[det_old]['frame'])
                iou_th = iou(np.array(detections[det_old]['bbx']), np.array([detections[det]['bbx']]))
                reid_dist = np.dot(np.array(detections[det_old]['reid']), np.array(detections[det]['reid'])) / (np.linalg.norm(np.array(detections[det_old]['reid'])) * np.linalg.norm(np.array(detections[det]['reid'])))
                if frame_gap < max_frame_gap and iou_th.max() > iou_threshold and reid_dist > reid_threshold:
                    group_cache.append(det)
                else:
                    groups.append(group_cache)
                    pids.append(pid)
                    group_cache = []
                    group_cache.append(det)
        if group_cache:
            groups.append(group_cache)
            pids.append(pid)
    assert len(groups) == len(pids)
    tracklet_all = {}
    for ii in range(len(groups)):
        group = sorted(groups[ii])
        if len(group) < 2:
            continue
        tid = ii
        sv_tid_to_gth[tid] = pids[ii]
        sv_gth_ret[tid] = [pids[ii] for _ in range(len(group))]
        features, frame_index, bounding_box = [], [], []
        for detid in group:
            features.append(detections[detid]['reid'])
            frame_index.append(detections[detid]['frame'])
            bounding_box.append(detections[detid]['bbx'])
        app_feature = np.mean(features, axis = 0).tolist()
        timestamps = []
        face_feature = []
        body_img = None
        tracklet_all[tid] = [app_feature, timestamps, frame_index, bounding_box, face_feature, body_img]
    return tracklet_all, sv_tid_to_gth, sv_gth_ret


def assign_matched_labels(track_nodes, label_nodes, overlap_threshold = 0.4):
    '''
    this function assigns labels for track_nodes. each track_node represents 1 track.
    input:
        track_nodes: each track_node represents 1 track. 
        track_nodes = [track_node1, track_node2, ...]
        track_node[frame] = [bbx, track_id]

        label_nodes: each label_node represents 1 label track.
        label_node[frame] = [bbx, label]
    output:
        ret_output[track_id] = label
    '''
    ret = {}
    GT_tid_num = {}
    for frame in sorted(track_nodes.keys()):
        if not track_nodes[frame]:
            continue
        for trk in track_nodes[frame]:
            tid = trk[1]
            ret.setdefault(tid, [])
            ret[tid].append(-1)
        if frame not in label_nodes or not label_nodes[frame]:
            continue
        for tt1 in label_nodes[frame]:
            GT_tid_num[tt1[1]] = GT_tid_num.get(tt1[1], 0) + 1
        tboxes = np.array([ d[0] for d in track_nodes[frame] ])
        lboxes = np.array([ d[0] for d in label_nodes[frame] ])
        ov = np.zeros((tboxes.shape[0], lboxes.shape[0]), dtype=np.float32)
        for i, box in enumerate(tboxes):
            ov[i, :] = iou(box, lboxes)
        #indices = linear_assignment(1.0 - ov)
        indices = linear_sum_assignment(1.0 - ov)
        indices = np.array(list(zip(*indices)))
        tbbx_to_GT = np.argmax(ov, axis=1)
        for ij in indices:
            i,j = ij
            pid = label_nodes[frame][j][1]
            if ov[i, j] > overlap_threshold and pid > 0:
                tid = track_nodes[frame][i][1]
                ret[tid][-1] = pid
    impure_list = {}
    ret_output = {}
    for key, val in ret.items():
        dict_tid = {}
        for tt in val:
            dict_tid[tt] = dict_tid.get(tt, 0)+1
        dict_tid1 = sorted(dict_tid.items(), key = lambda item:item[1], reverse = True)
        ret_output[key] = dict_tid1[0][0]
        if dict_tid1[0][0] == -1 and len(dict_tid1) > 1:
            ret_output[key] = dict_tid1[1][0]
        if len(dict_tid1) > 1 and dict_tid1[1][1] > 5:
            impure_list[key] = val
    return ret_output, ret, impure_list


def load_sv_pb_result_from_sv_pb_file(pb_file, threshold_filter=0):
    track_total = {}
    num_frames_before_merge = 0
    num_frames_after_merge = 0
    tracking_res = online_tracking_results_pb2.Tracks()
    with open(pb_file, 'rb') as f:
        tracking_res.ParseFromString(f.read())
    track_frame_res = {}
    for track in tracking_res.tracks:
        num_frames_before_merge += len(track.tracked_detections)
        if len(track.tracked_detections) > threshold_filter:
            num_frames_after_merge += len(track.tracked_detections)
            tid = track.tracklet_id
            app_feature = list(track.features.features[0].feats)
            tracked_detections = track.tracked_detections
            frame_index, bounding_box = [], []
            for det in tracked_detections:
                frame_idx = det.frame_index
                box_x = det.box_x
                box_y = det.box_y
                box_width = det.box_width
                box_height = det.box_height
                bbx = [box_x, box_y, box_width, box_height]
                frame_index.append(frame_idx)
                bounding_box.append(bbx)
                track_frame_res.setdefault(frame_idx, []).append([bbx, tid])
            sort_index = np.array(frame_index).argsort()
            frame_index = np.array(frame_index)[sort_index]
            frame_index = frame_index.tolist()
            bounding_box = np.array(bounding_box)[sort_index, :]
            bounding_box = bounding_box.tolist()
            if tid not in track_total:
                track_total[tid] = [app_feature, frame_index, bounding_box]
            else:
                tt1 = track_total[tid]
                track_total[tid] = [app_feature, tt1[2]+frame_index, tt1[3]+bounding_box]
    return track_frame_res, track_total, num_frames_after_merge, num_frames_before_merge


def compute_box_conflict(boxes1, boxes2, iou_th=0.8):
    '''
    Count the number of duplicate boxes
    :param boxes1:
    :param boxes2:
    :return:
    '''

    ts = set(boxes1.keys()) & set(boxes2.keys())
    ret = 0
    for t in ts:
        box1 = boxes1[t][0]
        box2 = boxes2[t][0]
        #_flow_util = of.FLOWtracks()
        #ov = _flow_util.iou(box1, [box2])[0]
        ov = iou(np.array(box1), np.array([box2]))[0]
        if ov < iou_th:
            ret += 1
        if ret > 2:
            break
    return ret

def poly_fit_boxes(all_timestamps, all_boxes, target_frame, degree = 1):
    all_xywh = all_boxes.copy().astype(np.float32)
    all_xywh[:, :2] += (all_boxes[:, 2:] / 2.0)
    # do poly fit and  get the fitted boxes
    fitted_boxes = []
    coeffs = []
    for i in range(4):
        c = np.polyfit(all_timestamps, all_xywh[:, i], degree)
        coeffs.append(c)
        func = np.poly1d(c)
        fitted = func(target_frame)
        fitted_boxes.append(fitted.tolist())
    fitted_boxes = np.array(fitted_boxes).T
    fitted_boxes[:, :2] -= (fitted_boxes[:, 2:] / 2.0)
    return fitted_boxes.tolist()

def compare_temporal_conflict(frame_bbx_i, frame_bbx_j, max_overlap=0):
    frame_i = [key for key in frame_bbx_i.keys()]
    frame_j = [key for key in frame_bbx_j.keys()]
    frame_ij = list(set(frame_i).intersection(set(frame_j)))
    if len(frame_ij) > max_overlap:
        bbx_i, bbx_j = frame_bbx_i[frame_ij[0]][0], frame_bbx_j[frame_ij[0]][0]
        bbx_ii = [bbx_i[0] + bbx_i[2]/2.0, bbx_i[1] + bbx_i[3]]
        bbx_jj = [bbx_j[0] + bbx_j[2]/2.0, bbx_j[1] + bbx_j[3]]
        spatial_distance = np.linalg.norm(np.array(bbx_ii) - np.array(bbx_jj)) 
        sv_conflict_num = compute_box_conflict(frame_bbx_i, frame_bbx_j)
        if sv_conflict_num > max_overlap:
            return False, [], spatial_distance
        else:
            return True, 0.0, spatial_distance
    else:
        frame_i.sort()
        frame_j.sort()
        frame_left_j = frame_j[0]
        frame_right_j = frame_j[-1]
        indee1 = bisect_left(frame_i, frame_left_j)
        indee2 = bisect_left(frame_i, frame_right_j)
        if indee1==indee2 and indee2==0:
            frame_couple = [frame_i[0], frame_j[-1]]
        elif indee1==indee2 and indee2==len(frame_i):
            frame_couple =[frame_i[-1], frame_j[0]]
        elif 0 < indee1 and indee1 < len(frame_i):
            frame_couple = [frame_i[indee1-1], frame_j[0]]
        else:
            frame_couple = [frame_i[indee2-1], frame_j[-1]]
        # compute the distance between the predicted box and the start box
        frame_ii = frame_i[-10:]
        predicted_box = poly_fit_boxes(np.array(frame_ii), np.array([frame_bbx_i[frame][0] for frame in frame_ii]), np.array([frame_left_j]))
        predicted_box = predicted_box[0]
        starting_box = frame_bbx_j[frame_left_j][0]
        bbx_ii = [predicted_box[0] + predicted_box[2]/2.0, predicted_box[1] + predicted_box[3]]
        bbx_jj = [starting_box[0] + starting_box[2]/2.0, starting_box[1] + starting_box[3]]
        spatial_distance = np.linalg.norm(np.array(bbx_ii) - np.array(bbx_jj))
        return True, abs(frame_couple[1] - frame_couple[0]), spatial_distance


def graph_construction(tracklet_input, knum, time_thd, time_sigma, distance_thd, distance_sigma, app_thd, time_conflict=0):
    #mlog.info('knum {}, time_thd {}, distance_thd {}, distance_sigma {}, app_thd {}, time_conflict {}'.format(knum, time_thd, distance_thd, distance_sigma, app_thd, time_conflict))
    num_total = len(tracklet_input)
    app_all = np.array([fea for tracklet in tracklet_input for fea in tracklet.features])
    feature_idx = np.array([idx for idx in range(len(tracklet_input)) for _ in tracklet_input[idx].features])
    app_matrix = np.dot(app_all, app_all.transpose())
    app_matrix_norm = np.sqrt(np.multiply(app_all, app_all).sum(axis=1))
    app_matrix_norm = app_matrix_norm[:, np.newaxis]
    app_simi_all = np.divide(app_matrix, np.dot(app_matrix_norm, app_matrix_norm.transpose()))
    #app_simi_all = (app_simi_all + 1) / 2.0
    del app_all, app_matrix, app_matrix_norm
    gc.collect()
    assert app_simi_all.shape[0] == len(feature_idx)
    assert app_simi_all.shape[1] == len(feature_idx)
    for i in range(num_total):
        app_simi_all[i][i] = -float('inf')
    start_frame_list = [tracklet.start_frame for tracklet in tracklet_input] 
    assert sorted(start_frame_list) == start_frame_list
    ners = np.zeros([num_total, knum], dtype=np.int32) - 1
    simi = np.zeros([num_total, knum]) - 1
    for i in range(num_total-1):
        max_start_frame_i = tracklet_input[i].end_frame + time_thd + 1
        max_idx = bisect_right(start_frame_list, max_start_frame_i)
        if max_idx <= i+1:
            continue
        indee = [idx for idx in range(i+1, max_idx)]
        sim_all = []
        for j in indee:
            frame_bbx_i = tracklet_input[i].frame_bbx
            frame_bbx_j = tracklet_input[j].frame_bbx
            indee_tem, frame_dis, distance = compare_temporal_conflict(frame_bbx_i, frame_bbx_j, time_conflict)
            if indee_tem:
                if frame_dis > time_thd:
                    sim_all.append(-float('inf'))
                else:
                    time_simi = np.exp(-1*(float(frame_dis)/time_sigma))
                    if (distance) > distance_thd:
                        sim_all.append(-float('inf'))
                    else:
                        distance_simi = np.exp(-1*(distance/distance_sigma))
                        app_idx_i = np.where(feature_idx==i)
                        app_idx_j = np.where(feature_idx==j)
                        app_simi_i = app_simi_all[app_idx_i]
                        app_simi_ij = app_simi_i.T[app_idx_j]
                        app_simi = app_simi_ij.mean()
                        if app_simi >= app_thd:
                            sim_all.append(float(app_simi+distance_simi+time_simi)/3)
                        else:
                            sim_all.append(-float('inf'))
            else:
                sim_all.append(-float('inf'))
        available_index = np.where(np.array(sim_all) > -10)
        sim_all = np.array(sim_all)[available_index]
        sim_all = sim_all.tolist()
        indee = np.array(indee)[available_index]
        indee = indee.tolist()
        if len(sim_all) > 0:
            simi_dict = {}
            for kk in range(len(sim_all)):
                sim = sim_all[kk]
                simi_dict.setdefault(sim, []).append(kk)
            knum_list = heapq.nlargest(min(knum, len(sim_all)), sim_all)
            ind_list, ind_used = [], {}
            for knum_i in knum_list:
                ind_tt = simi_dict[knum_i]
                if knum_i not in ind_used:
                    ind = ind_tt[0]
                    ind_used[knum_i] = 0
                else:
                    ind_new = ind_used[knum_i] + 1
                    ind = ind_tt[ind_new]
                    ind_used[knum_i] = ind_new
                ind_list.append(ind)
            ners_i, sim_i = [], []
            for ind in ind_list:
                ners_i.append(indee[ind])
                sim_i.append(sim_all[ind])
            ners[i, :len(ners_i)] = ners_i
            simi[i, :len(sim_i)] = sim_i
    anchor = np.tile(np.arange(num_total).reshape(num_total, 1), (1, knum))
    selidx = np.where((simi >= 0) & (ners != -1) & (ners != anchor))
    pairs = np.hstack((anchor[selidx].reshape(-1, 1), ners[selidx].reshape(-1, 1)))
    scores = simi[selidx]
    if len(pairs) > 0:
        pairs = np.sort(pairs, axis=1)
        pairs, unique_idx = np.unique(pairs, return_index=True, axis=0)
        scores = scores[unique_idx]
    return pairs, scores


def Super_vertex_based_proposal_generation(cluster_input, args, proposals_all, sv_tid_to_gth=None):
    proposals_all1 = copy.deepcopy(proposals_all)
    #cluster_ini = copy.deepcopy(cluster_input)
    cluster_candidates = []
    time_thd = args.time_thd
    time_thd_max = args.time_thd_max
    distance_thd_max = args.distance_thd_2d_max
    distance_thd = args.distance_thd_2d
    app_thd = args.app_thd
    app_thd_min = args.app_thd_min
    iter_max = args.iter_max
    th_step = args.th_step
    max_size = args.max_size
    distance_sigma = args.distance_sigma
    time_sigma = args.time_sigma
    time_conflict, time_conflict_max = 0, 0
    for itr in range(iter_max):
        while True:
            cluster_nums_input = len(cluster_input)
            cluster_input = sorted(cluster_input,  key=lambda x: x.start_frame)
            pairs, scores = graph_construction(cluster_input, args.knum, min(time_thd, time_thd_max), time_sigma, min(distance_thd, distance_thd_max), distance_sigma, max(app_thd, app_thd_min), min(time_conflict, time_conflict_max))
            if len(pairs) > 0:
                clusters = cluster_nodes(pairs, scores, th_step, max_size)
                cluster_output, proposals_tt = update_cluster(cluster_input, clusters, time_conflict)
                for cluster_tmp in cluster_output:
                    if cluster_tmp not in cluster_candidates:
                        cluster_candidates.append(cluster_tmp)
                proposals_all1.append(proposals_tt)
                cluster_input = cluster_output
                if sv_tid_to_gth is not None:
                    gth_to_tid = list(set([pid for tid, pid in sv_tid_to_gth.items() if pid != -1]))
                    pure_proposals, all_proposals, pt_proposals = 0, 0, 0
                    for proposal_i in proposals_tt:
                        id_sets = set()
                        for tid in proposal_i:
                            if tid in sv_tid_to_gth and sv_tid_to_gth[tid] != -1:
                                id_sets.add(sv_tid_to_gth[tid])
                        if len(id_sets) <= 1:
                            pure_proposals += 1
                        if len(id_sets) == 1:
                            pt_proposals += 1
                        all_proposals += 1
                    mlog.info('[Iter: {}] precision is {} {}/{}, NT is {} {}/{}, PT is {} {}/{}'.format(itr, float(pure_proposals)/all_proposals, pure_proposals, all_proposals, float(len(proposals_tt))/len(gth_to_tid), len(proposals_tt), len(gth_to_tid), float(pt_proposals)/len(gth_to_tid), pt_proposals, len(gth_to_tid)))
            cluster_nums_output = len(cluster_input)
            if cluster_nums_input == cluster_nums_output:
                break
        time_thd += float(time_thd_max)/iter_max
        distance_thd += float(distance_thd_max)/iter_max
        app_thd -= float(app_thd - app_thd_min)/iter_max
        time_conflict += float(time_conflict_max)/iter_max
        if len(cluster_input) <= 1:
            break
    return proposals_all1, cluster_candidates


class clusterNode:
    '''
    a node of cluster (proposal)
    '''
    def __init__(self, tracklet_name, tracklet_value):
        self.node_name = set()
        self.node_name.add(tracklet_name)
        self.features = []
        self.features.append(tracklet_value[0])
        self.frame_index = tracklet_value[1]
        points = np.array([[bbx[0] + bbx[2]/2, bbx[1] + bbx[3]] for bbx in tracklet_value[2]])
        self.bbxfloor = {}
        self.bbx = points
        frame_bbx, frame_point = {}, {}
        for i in range(len(tracklet_value[1])):
            frame = tracklet_value[1][i]
            frame_bbx[frame] = [tracklet_value[2][i]]
            frame_point[frame] = [points[i]]
        self.frame_bbx = frame_bbx
        self.frame_point = frame_point
        self.start_frame = min(tracklet_value[1])
        self.end_frame = max(tracklet_value[1])
        self.start_bbx = points[0]
        self.end_bbx = points[-1]
    
    def addNode(self, new_node):
        name1 = self.node_name | new_node.node_name
        self.node_name = name1
        fea1 = self.features + new_node.features
        self.features = fea1
        self.frame_index += new_node.frame_index
        bbx1 = np.concatenate((self.bbx, new_node.bbx), axis=0)
        self.bbx = bbx1
        if len(self.bbxfloor) > 0 and len(new_node.bbxfloor) > 0:
            bbxfloor1 = np.concatenate((self.bbxfloor, new_node.bbxfloor), axis=0)
            self.bbxfloor = bbxfloor1
        self.start_frame = min(self.start_frame, new_node.start_frame)
        self.end_frame = max(self.end_frame, new_node.end_frame)
        if self.start_frame == new_node.start_frame:
            self.start_bbx = new_node.start_bbx
        if self.end_frame == new_node.end_frame:
            self.end_bbx = new_node.end_bbx
        frame_bbx1 = self.frame_bbx
        frame_bbx2 = new_node.frame_bbx
        for key in frame_bbx2.keys():
            if key not in frame_bbx1:
                frame_bbx1[key] = []
            frame_bbx1[key] += frame_bbx2[key]
        self.frame_bbx = frame_bbx1
        frame_point1 = self.frame_point
        frame_point2 = new_node.frame_point
        for key in frame_point2.keys():
            if key not in frame_point1:
                frame_point1[key] = []
            frame_point1[key] += frame_point2[key]
        self.frame_point = frame_point1


def update_cluster(cluster_input, clusters, max_overlap=0):
    cluster_output = []
    node_used = []
    for clus in clusters:
        indee_tem = []
        for i in range(len(clus)):
            for j in range(i+1, len(clus)):
                frame_bbx_i = cluster_input[clus[i]].frame_bbx
                frame_bbx_j = cluster_input[clus[j]].frame_bbx
                inde1, _, _ = compare_temporal_conflict(frame_bbx_i, frame_bbx_j, max_overlap)
                indee_tem.append(inde1)
        if all(indee_tem):
            ii = 0
            for numi in clus:
                node_used.append(numi)
                if ii == 0:
                    node_ini = copy.deepcopy(cluster_input[numi])
                    ii += 1
                else:
                    node_ini.addNode(cluster_input[numi])
            cluster_output.append(node_ini)
    node_all = [i for i in range(len(cluster_input))]
    node_left = [i for i in node_all if i not in node_used]
    for tt in node_left:
        cluster_output.append(cluster_input[tt])
    proposals_all = [clu.node_name for clu in cluster_output]
    return cluster_output, proposals_all


def deoverlap_IoU_IoP_combine_threshold(proposals, IoU, IoP, weight, threhold=0.50, tracklet_all = {}):
    proposals_new, IoU_new, IoP_new, IoU_IoP_new = [], [], [], []
    for i in range(len(proposals)):
        proposals_new.append(proposals[i])
        iop = IoP[i]
        if iop >= threhold:
            iop = 1
        else:
            iop = 0
        iou = IoU[i]
        IoU_new.append(iou)
        IoP_new.append(iop)
        IoU_IoP_new.append(weight*iop+iou)
    sorted_ind = sorted(range(len(IoU_IoP_new)), key=lambda k: IoU_IoP_new[k], reverse=True)
    proposals_new = [proposals_new[i] for i in sorted_ind]
    IoU_new = [IoU_new[i] for i in sorted_ind]
    IoP_new = [IoP_new[i] for i in sorted_ind]
    IoU_IoP_new = [IoU_IoP_new[i] for i in sorted_ind]
    tracklet_label_output = {}
    num_ii = [i for i in IoP_new if i>0]
    num_pro = len(num_ii)
    tracklet_IoP_perfect = set()
    for i, cluster in enumerate(proposals_new):
        for v in cluster:
            if v not in tracklet_label_output:
                tracklet_label_output[v] = []
            tracklet_label_output[v].append(i)
            tracklet_IoP_perfect.add(v)
    tracklet_label_output1 = {}
    for idx, lbs in tracklet_label_output.items():
        tracklet_label_output1[idx] = lbs[0]
    proposal_final = {}
    for key, val in tracklet_label_output1.items():
        if val not in proposal_final:
            proposal_final[val] = set()
        proposal_final[val].add(key)
    proposal_out = [list(val) for val in proposal_final.values()]
    return tracklet_label_output1, proposal_out, tracklet_IoP_perfect


def merge_sv_tracks(tracks, vid_id = None, feat_merge_type = 'avg'):
    # early stopping if a single track
    if len(tracks) == 1:
        return tracks[0]

    track_id = tracks[0].track_id
    for t in tracks:
        assert t.track_id == track_id
    features = []
    for trk in tracks:
        if trk.features and len(trk.features.features) > 0:
            feat = [ d for d in trk.features.features[0].feats ]
            features.append(feat)
    if features:
        if feat_merge_type == 'avg':
            feat = np.mean(features, axis = 0)
            del tracks[0].features.features[:]
            tf = tracks[0].features.features.add()
            for d in feat:
                tf.feats.append(d)
        elif feat_merge_type == 'all':
            del tracks[0].features.features[:]
            for feat in features:
                tf = tracks[0].features.features.add()
                for d in feat:
                    tf.feats.append(d)
        elif feat_merge_type == 'conf':
            best_q, num_q = 0, 0
            best_idx1, best_idx2 = -1, -1
            for i, trk in enumerate(tracks):
                if trk.face_quality > best_q:
                    best_q = trk.face_quality
                    best_idx1 = i
                if len(trk.tracked_detections) > num_q:
                    num_q = len(trk.tracked_detections)
                    best_idx2 = i
            if best_idx1 >= 0:
                feat = [ d for d in tracks[best_idx1].features.features[0].feats ]
            else:
                assert best_idx2 >= 0
                feat = [ d for d in tracks[best_idx2].features.features[0].feats ]
            del tracks[0].features.features[:]
            tf = tracks[0].features.features.add()
            for d in feat:
                tf.feats.append(d)
        #assert len(tracks[0].features.features[0].feats) == tracks[0].features.dim
    # merge detections
    detections = []
    for t in tracks:
        for det in t.tracked_detections:
            detections.append(det)

    detections = sorted(detections, key=lambda x: x.frame_index)
    del tracks[0].tracked_detections[:]

    last_frame_index = -1
    for det in detections:
        if last_frame_index == det.frame_index:
            continue
        buf = tracks[0].tracked_detections.add()
        buf.CopyFrom(det)
        last_frame_index = det.frame_index
    
    return tracks[0]


def save_merge(track_path, merge_id_map, output_path):
    tracks = load_pb(track_path)
    for tid in tracks:
        try:
            tracks[tid] = merge_sv_tracks(tracks[tid])
        except Exception as e:
            print("Error: failed to merge for" + track_path + " exception:" + str(e))
            raise e
    # use this round merging information
    change_count = 0
    tt_index = len(merge_id_map) + 1
    grouped = {}
    for tid in tracks:
        trk = tracks[tid]
        if trk.track_id in merge_id_map:
            tracklet_id, track_id = merge_id_map[trk.track_id]
            trk.tracklet_id = tracklet_id
            trk.track_id = track_id
            change_count += 1
            grouped.setdefault(track_id, [])
            grouped[track_id].append(trk)
    for tid in grouped:
        grouped[tid] = merge_sv_tracks(grouped[tid])

    mlog.info("There are {} number of tracks with modified trackid out of {}".format(
        change_count, len(tracks)))
    mlog.info("{} => {}".format(len(tracks), len(grouped) ))
    
    # now save to pb file
    tracks_from_pb = online_tracking_results_pb2.Tracks()
    with open(track_path, 'rb') as f:
        tracks_from_pb.ParseFromString(f.read())
    del tracks_from_pb.tracks[:]
    for tid in grouped:
        trk = grouped[tid]
        buf = tracks_from_pb.tracks.add()
        buf.CopyFrom(trk)
    with open(output_path, 'wb') as f:
        f.write(tracks_from_pb.SerializeToString())

def post_processing_trajectory_smoothness(track_res, min_track_length_filter=2):
    def interpolate_boxes(frames, boxes, target_frames):
        uniq_idx = [0] + [ i for i in range(1, len(frames)) if frames[i] > frames[i - 1] ]
        uniq_frames = [ frames[i] for i in uniq_idx ]
        if len(uniq_frames) == 1:
            return target_frames, boxes[uniq_idx]
        
        ret = []
        for i in range(boxes.shape[1]):
            f = interp1d(uniq_frames, boxes[uniq_idx, i], assume_sorted=True)
            ret.append(f(target_frames))
        return target_frames, np.array(ret).T
    
    track_res_sort = sorted(track_res.items(), key=lambda d: d[0])
    track_res_stats = {}
    track_id_mapping, start_id = {}, 1
    for item in track_res_sort:
        frame, bbxs = item
        for bbx in bbxs:
            track_id = bbx[1]
            if track_id not in track_id_mapping:
                track_id_mapping[track_id] = start_id
                start_id += 1
            track_res_stats.setdefault(track_id_mapping[track_id], {})
            track_res_stats[track_id_mapping[track_id]].setdefault('frames', []).append(frame)
            track_res_stats[track_id_mapping[track_id]].setdefault('boxes', []).append(bbx[0])
    track_res_smoothness = {}
    for track_id in track_res_stats:
        frames = track_res_stats[track_id]['frames']
        assert sorted(frames) == frames
        if len(frames) < min_track_length_filter:
            continue
        boxes = track_res_stats[track_id]['boxes']
        sampled_frames = [ii for ii in range(min(frames), max(frames)+1)]
        tframes, tboxes = interpolate_boxes(np.array(frames), np.array(boxes), sampled_frames)
        valid = [ i for i, box in enumerate(tboxes) if not np.any(np.isnan(box)) ]
        tframes = [tframes[i] for i in valid]
        tboxes = [tboxes[i].tolist() for i in valid]
        if len(tframes) < min_track_length_filter:
            continue
        for ii, frame in enumerate(tframes):
            track_res_smoothness.setdefault(frame, []).append([tboxes[ii], track_id])
    return track_res_smoothness
