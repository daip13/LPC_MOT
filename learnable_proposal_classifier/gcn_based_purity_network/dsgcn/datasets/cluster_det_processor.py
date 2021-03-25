import numpy as np
import random
import time
import sys
import json
import torch


def load_npz(fn):
    return np.load(fn)['data']

def load_json(fn):
    return json.load(open(fn, 'r'))


def load_data(ofn):
    if ofn.endswith('.json'):
        return load_json(ofn)
    else:
        return load_npz(ofn)

def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


class ClusterDetProcessor_pipeline(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.dtype = np.float32

    def __len__(self):
        return self.dataset.size
   
    def build_graph(self, fn_node):
        node_name = fn_node.split('/')[-1].split('_')[0]
        features = self.dataset.features
        spatem = self.dataset.spatem
        trk_vid_name = self.dataset.trk_vid_name
        trk_frm_num = self.dataset.trk_frm_num
        node_old = load_data(fn_node)
        video_name = [trk_vid_name[trk] for trk in node_old]
        assert len(set(video_name)) == 1
        # notice: we set the label to be 1.0 in inference. Actually, it can be arbitrary value because we donot use this label in inference.
        label_output = 1.0
        node = list(node_old) 
        features_node = features[node, :]
        spatem_node = spatem[node, :]
        # sort the nodes according to the start time
        node = np.array(node)
        sort_index = spatem_node[:,0].argsort()
        spatem_node = spatem_node[sort_index, :]
        assert sorted(spatem_node[:,0].tolist()) == spatem_node[:,0].tolist() 
        features_node = features_node[sort_index, :]
        node = node[sort_index]
        # compute the similarity
        edge = []
        for ii in range(len(node)):
            for jj in range(ii,len(node)):
                idx1, idx2 = node[ii], node[jj]
                fea1, fea2 = features_node[ii, :], features_node[jj, :]
                spatem1, spatem2 = spatem_node[ii, :], spatem_node[jj, :]
                if jj == ii:
                    dist = 1.0
                else:
                    reid_simi = (1.0 + np.dot(fea1,fea2)/(np.linalg.norm(fea1)*(np.linalg.norm(fea1)))) / 2.0
                    temporal_dist = np.exp(-1*(float(abs(spatem2[0] - spatem1[1])/100)))
                    spatial_dist = np.exp(-1*np.linalg.norm(np.array(spatem2[2:4]) - np.array(spatem1[4:6])) / 200.0) 
                    dist = (temporal_dist + spatial_dist + reid_simi)/3.0
                #if dist > 0:
                edge.append([idx1, idx2, dist])
        # build adjcent matrix
        abs2rel = {}
        for i, n in enumerate(node):
            abs2rel[n] = i
        size = len(node)
        adj = np.eye(size)
        for e in edge:
            w = 1.
            if len(e) == 2:
                e1, e2 = e
            elif len(e) == 3:
                e1, e2, dist = e
                if not self.dataset.wo_weight:
                    w = 1. - dist
            else:
                raise ValueError('Unknown length of e: {}'.format(e))
            v1 = abs2rel[e1]
            v2 = abs2rel[e2]
            adj[v1][v2] = dist
            adj[v2][v1] = dist
        # add the temporal spatial information into the feature vectors
        if True:
            vertices1, vertices2 = [], []
            vertices1.append(features_node[0,:].tolist())
            vertices2.append([1, 0, 0, 0, 0])
            for ii in range(1, len(node)):
                spatem_old = spatem_node[ii-1,:].tolist()
                app_feature = features_node[ii,:].tolist()
                spatem_feature = spatem_node[ii,:].tolist()
                fps = 10
                time_diff = float(spatem_feature[0] - spatem_old[1])/fps
                u_diff = float(spatem_feature[2] - spatem_old[4])/((spatem_feature[6] + spatem_old[8])/2.0)
                v_diff = float(spatem_feature[3] - spatem_old[5])/((spatem_feature[7] + spatem_old[9])/2.0)
                w_diff = np.log(float(spatem_feature[6]) / spatem_old[8])
                h_diff = np.log(float(spatem_feature[7]) / spatem_old[9])
                vertices1.append(app_feature)
                vertices2.append([time_diff, u_diff, v_diff, w_diff, h_diff])
            vertices1 = l2norm(np.array(vertices1))
            vertices2 = np.array(vertices2)
        return vertices1, vertices2, adj, float(label_output) 

    def __getitem__(self, idx):
        if idx is None or idx > self.dataset.size:
            raise ValueError('idx({}) is not in the range of {}'.format(idx, self.dataset.size))
        fn_node = self.dataset.lst[idx]
        vertices1, vertices2, adj, binary_label = self.build_graph(fn_node)
        return vertices1.astype(self.dtype), \
               vertices2.astype(self.dtype), \
               adj.astype(self.dtype), \
               np.array(binary_label, dtype=self.dtype)
