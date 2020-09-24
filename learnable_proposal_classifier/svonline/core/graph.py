import numpy as np
from data_types import *
from node import *
from affinity import compute_matches_dense

class Graph(object):
    def __init__(self, nodes, affinity_metrics, camera_view=None):
        self._nodes = nodes
        self._affinity_metrics = affinity_metrics
        self._camera_view = camera_view
        self._edge_cache = None
        self._start = None
        self._end = None

    @property
    def affinity_metrics(self):
        return self._affinity_metrics
    
    @property
    def start(self):
        if self._start is None:
            if self._camera_view is None:
                self._start = np.inf
                for n in self._nodes:
                    if self._start > n.start:
                        self._start = n.start
            else:
                self._start = np.inf
                for n in self._nodes:
                    if self._start > n[self._camera_view].start:
                        self._start = n[self._camera_view].start
        return int(self._start)

    @property
    def end(self):
        if self._end is None:
            if self._camera_view is None:
                self._end = 0
                for n in self._nodes:
                    if self._end < n.end:
                        self._end = n.end
            else:
                self._end = 0
                for n in self._nodes:
                    if self._end < n[self._camera_view].end:
                        self._end = n[self._camera_view].end
        return int(self._end)
       
    @property
    def camera_view(self):
        return self._camera_view

    @property
    def size(self):
        return len(self._nodes)

    @property
    def nodes(self):
        return self._nodes

class BipartiteMatchingGraph(object):
    def __init__(self, nodes_from, nodes_to, affinity_metrics, camera_view):
        self._affinity_metrics = affinity_metrics
        self._camera_view = camera_view
        self._edge_cache = None
        self._nodes_from = nodes_from
        self._nodes_to = nodes_to

    @property
    def camera_view(self):
        return self._camera_view

    @property
    def edges(self):
        if self._edge_cache is None:
            self._edge_cache = compute_matches_dense(
                        self._nodes_from, self._nodes_to, 
                        self._affinity_metrics, self._camera_view)
        return self._edge_cache
