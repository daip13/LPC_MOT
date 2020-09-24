#coding=utf-8
import os, sys
import random
import os.path as osp
import glob
import time
import cv2
import numpy as np
import scipy
from scipy.spatial.distance import cdist
import logging
import networkx as nx
import sklearn
from sklearn.cluster import DBSCAN
import heapq

def NsmallestIndex(arr, n):
    arr_index = [ (idx, v) for idx, v in enumerate(arr)]
    indexes = heapq.nsmallest(n, arr_index, key=lambda v:v[1])
    return [v[0] for v  in indexes]

class Cluster:
  def __init__(self, idxes, feats):
    self.idxes = idxes
    self.feats = feats
    self.center = self.feats.mean(axis=0)
    self.center /= np.linalg.norm(self.center)

  def merge(self, cls):
    self.idxes = list(set(self.idxes.extend(cls.idxes) ) )
    self.feats = np.concatenate([self.feats, cls.feats], axis=0) 

  def update(self):
    self.center = self.feats.mean(axis=0)
    self.center /= np.linalg.norm(self.center)

class HierarchicalCluster:
  def __init__(self, max_cluster_size = 500, thresh=0.6, policy='PERCENTILE', percentile=0.3): # policy in ['PERCENTILE', 'mean']
    self.thresh = thresh
    self.policy = policy
    self.percentile = percentile
    self.max_cluster_size = max_cluster_size

  def predict(self, feats):
    dists = cdist(feats, feats)
    num = dists.shape[0]
    mean_dist = np.mean(dists[dists > (1e-5)])
    min_dist = np.min(dists[dists > (1e-5)])
    logger.info('mean dists:{}, min dists:{}'.format(mean_dist, min_dist))
    thresh_step = 0.1
    cur_thresh = min_dist
    #initialization
    self._clusters = []
    for i in range(num):
      self._clusters.append(Cluster([i], np.reshape(feats[i], (1, -1)) ) )
    step = 0 
    while True:
      cur_thresh += thresh_step
      step += 1
      if step >= 6:
        break
      G = nx.Graph()
      for i in range(len(self._clusters)):
        G.add_node(i)
        for j in range(i+1, len(self._clusters)):
          if self.policy == 'mean':
            dist = 1 - np.dot(self._clusters[i].center, self._clusters[j].center)
          elif self.policy == 'PERCENTILE':
            dists = cdist(self._clusters[i].feats, self._clusters[j].feats)
            dist = np.percentile(dists, self.percentile)
          elif self.policy == 'max':
            dists = cdist(self._clusters[i].feats, self._clusters[j].feats)
            dist = np.max(dists)
          if dist < cur_thresh:
            G.add_edge(i,j)

      graphs = list(nx.connected_components(G))
      clusters = []
      for idx, g in enumerate(graphs):
        nodes = list(g)
        clusters.append(nodes)
      clusters = sorted(clusters, key=lambda k: -len(k)) #
      new_clusters = []
      for idx, cluster in enumerate(clusters):
        idxes = []
        feats = []
        for i in cluster:
          idxes.extend(self._clusters[i].idxes)
          feats.append(self._clusters[i].feats)
        feats = np.concatenate(feats, axis=0)
        new_clusters.append(Cluster(idxes, feats))
      self._clusters = new_clusters
      self._clusters = sorted(self._clusters, key=lambda k: -len(k.idxes))
      if len(self._clusters) <= 3:
        break
      dists = cdist(self._clusters[0].feats, self._clusters[0].feats)
      dists = dists[dists > 1e-5]
      if np.max(dists) > self.thresh or len(feats) > self.max_cluster_size:
        break
    return self._clusters[0]

class GraphCluster:
    def __init__(self, topk=2, thresh=0.2, metric='cosine'):
        self.topk = topk
        self.thresh = thresh
        assert metric in ['cosine', 'Euclidean']
        self.metric = metric
        
    def predict(self, feats, logger):
        dists = cdist(feats, feats, metric=self.metric)
        num = dists.shape[0]
        mean_dist = np.mean(dists[dists > (1e-5)])
        logger.info('mean dists:{}'.format(mean_dist))
        G = nx.Graph()
        logger.info('Construct graph...')
        for i in range(num):
            # idxes = np.argsort(dists[i])[1:]
            idxes = NsmallestIndex(dists[i], self.topk+1)
            idxes = [j for j in idxes if j != i]
            G.add_node(i)
            for j in range(self.topk):
                idx = idxes[j]
                #if dists[i, idx] < mean_dist:
                if dists[i, idx] < self.thresh:
                    G.add_edge(i,idx)

        logger.info('Get the connected componenets...')
        graphs = list(nx.connected_components(G))
        logger.info('Sort the clusters...')
        clusters = []
        for idx, g in enumerate(graphs):
            nodes = list(g)
            nodes.sort()
            clusters.append(nodes)
        clusters = sorted(clusters, key=lambda k: -len(k)) #
        return clusters

class DBSCANCluster:
  def __init__(self, min_samples=3, thresh=0.3):
    self.dbscan = DBSCAN(eps=thresh, min_samples=min_samples, metric='cosine')
  def predict(self, feats, logger=None):
    num = feats.shape[0]
    labels = self.dbscan.fit_predict(feats)
    clusters = []
    label_set = set(labels)
    for label in label_set:
      if label == -1:
        continue
      cluster = [i for i in range(num) if labels[i] == label]
      clusters.append(cluster)
    clusters = sorted(clusters, key=lambda k: -len(k)) #
    return clusters

class CenterCluster:
  def __init__(self, thresh=0.3):
    self.thresh = thresh
  def predict(self, feats, logger=None):
    center = feats.sum(axis=0)
    center /= np.linalg.norm(center)
    cluster = []
    num = feats.shape[0]
    for i in range(num):
      dist = 1 - np.dot(feats[i], center)
      if dist < self.thresh:
        cluster.append(i)
    cls = [cluster]
    return cls

def create_cluster(args):
  cluster_dict = {'GraphCluster': GraphCluster, 'DBSCANCluster':DBSCANCluster, 'CenterCluster': CenterCluster}
  if args.cluster_type ==  'GraphCluster':
    return cluster_dict[args.cluster_type](topk=args.topk, thresh=args.thresh)
  elif args.cluster_type ==  'DBSCANCluster':
    return cluster_dict[args.cluster_type](min_samples=args.min_samples, thresh=args.thresh)
  elif args.cluster_type ==  'CenterCluster':
    return cluster_dict[args.cluster_type](thresh=args.thresh)
  else:
    raise NotImplemented