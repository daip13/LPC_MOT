import os
import sys
import json
import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.optimize import linear_sum_assignment
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from core.graph import Graph, BipartiteMatchingGraph
from core.node import nodes_in_time, split_nodes
from core.creator import *
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)

MATCH_THRESHOLD = 1e3

class BaseAssociation(object):
    __metaclass__ = ABCMeta
    def __init__(self, verbose=0):
        self._verbose = verbose

    @abstractmethod
    def __call__(self, graph):
        raise NotImplementedError

class OnlineHungarianMethod(BaseAssociation):
    def __init__(self, verbose = 0, track_alive = 1, relative_threshold = 0.0, cap = None, out_dir = None, regular_batch_opt = None):
        super(OnlineHungarianMethod, self).__init__(verbose)
        self._track_alive = track_alive
        self._relative_threshold = relative_threshold
        self._cap = cap
        if self._cap is not None:
            self._dbg = True
            self._dbg_dir = out_dir
        else:
            self._dbg = False
            self._dbg_dir = None
        
        self._do_batch = False
        if regular_batch_opt is not None:
            self._do_batch = True
            self._batch_affinities, self._batch_engine, self._batch_configs =\
                    regular_batch_opt
    
    def update(self, target_nodes, det_nodes, affinity_metrics, camera_view):
        def check_competition(A, i, j):
            if self._relative_threshold == 0:
                return False
            if True:
                # two conditions.
                v = A[i, j]
                # 1. find all the other existing tracks competing for the same det j
                temp = A[:, j].copy()
                temp[i] = MATCH_THRESHOLD
                other_targets = np.where(temp - self._relative_threshold < v)[0]
                if other_targets.size > 0:
                    # 2. see if the competing track has any other obvious match
                    for ii in other_targets:
                        vii = A[ii, j]
                        if vii - self._relative_threshold < A[ii, :].min():
                            return True
                # 1. find all the other existing detections competing for the same track i
                temp = A[i, :].copy()
                temp[j] = MATCH_THRESHOLD
                other_dets = np.where(temp - self._relative_threshold < v)[0]
                if other_dets.size > 0:
                    # 2. see if the competing detection has any other obvious match
                    for jj in other_dets:
                        vjj = A[i, jj]
                        if vjj - self._relative_threshold < A[:, jj].min():
                            return True
                return False
            else:
                # check relative threshold
                # will not allow to associate when there is a strong competition
                temp1 = A[i, :].copy() # one track having two confusing candidate detections
                temp1[j] = MATCH_THRESHOLD
                temp2 = A[:, j].copy() # one detection having two competing tracks
                temp2[i] = MATCH_THRESHOLD
                return A[i, j] > \
                        min(temp1.min(), temp2.min()) - self._relative_threshold

        ret = [ -1 for _ in det_nodes ]
        kill = []
        if target_nodes and det_nodes:
            # solve the association with gating
            graph = BipartiteMatchingGraph(target_nodes, 
                        det_nodes, affinity_metrics, camera_view)
            A = graph.edges
            A[A > MATCH_THRESHOLD] = MATCH_THRESHOLD # numerical stability
            indices = linear_sum_assignment(A.astype(np.float64))
            indices = np.array(list(zip(*indices)))
            for ij in indices:
                i, j = ij
                if check_competition(A, i, j):
                    kill.append(i)
                    continue
                if A[i, j] < MATCH_THRESHOLD:
                    ret[j] = i
        return ret, kill

    def __call__(self, graph, camera_view):
        output_nodes = []
        alive_target_nodes = []
        input_nodes = split_nodes(graph.nodes, 1, camera_view)
        for frame, det_nodes in enumerate(input_nodes):
            if self._verbose > 0:
                if frame % 10 == 0:
                    mlog.info(
                            "processed {0} frames of {1}. There are {2} tracks.".format(
                                frame, graph.end, len(output_nodes)))
            # filter out old ones
            _, alive_target_nodes = nodes_in_time(
                    alive_target_nodes, frame - self._track_alive, 
                    frame - 1, camera_view)

            matches, kill = self.update(
                    alive_target_nodes, det_nodes,
                    graph.affinity_metrics, camera_view)

            for i, m in enumerate(matches):
                if m >= 0:
                    alive_target_nodes[m].extend(det_nodes[i])
                else:
                    output_nodes.append(det_nodes[i].clone())
                    alive_target_nodes.append(output_nodes[-1])
        
            # removed killed ones
            alive_target_nodes = [ node for i, node in enumerate(alive_target_nodes) if i not in kill ]

        mlog.info('tracking done with {0} tracklets'.format(len(output_nodes)))
        return output_nodes
