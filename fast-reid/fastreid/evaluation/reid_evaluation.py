# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
from collections import OrderedDict
from sklearn import metrics
import json

import numpy as np
import torch
import torch.nn.functional as F

from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank import evaluate_rank
from .rerank import re_ranking
from .roc import evaluate_roc
from fastreid.utils import comm

logger = logging.getLogger(__name__)


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.features = []
        self.pids = []
        self.camids = []
        self.im_names = []

    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []

    def process(self, inputs, outputs):
        self.pids.extend(inputs["targets"])
        self.camids.extend(inputs["camid"])
        self.im_names.extend(inputs["img_path"])
        self.features.append(outputs.cpu())

    @staticmethod
    def cal_dist(metric: str, query_feat: torch.tensor, gallery_feat: torch.tensor):
        assert metric in ["cosine", "euclidean"], "must choose from [cosine, euclidean], but got {}".format(metric)
        if metric == "cosine":
            dist = 1 - torch.mm(query_feat, gallery_feat.t())
        else:
            m, n = query_feat.size(0), gallery_feat.size(0)
            xx = torch.pow(query_feat, 2).sum(1, keepdim=True).expand(m, n)
            yy = torch.pow(gallery_feat, 2).sum(1, keepdim=True).expand(n, m).t()
            dist = xx + yy
            dist.addmm_(query_feat, gallery_feat.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist.cpu().numpy()

    def gen_json(self, dist, query_names, gallery_names):
        gallay_length = int(200)
        result = {}
        idx = np.argsort(dist, axis=1)
        for i in range(len(query_names)):
            query = query_names[i]
            result[query] = gallery_names[idx[i]][:gallay_length].tolist()
        with open(self.cfg.TEST.JSON_FILE, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result))
        print('Finish generating json file!')

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            pids = comm.gather(self.pids)
            pids = sum(pids, [])

            camids = comm.gather(self.camids)
            camids = sum(camids, [])

            im_names = comm.gather(self.im_names)
            im_names = sum(im_names, [])
            if not comm.is_main_process():
                return {}
        else:
            features = self.features
            pids = self.pids
            camids = self.camids
            im_names = self.im_names

        features = torch.cat(features, dim=0)
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(pids[:self._num_query])
        query_camids = np.asarray(camids[:self._num_query])
        query_names = np.asarray(im_names[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(pids[self._num_query:])
        gallery_camids = np.asarray(camids[self._num_query:])
        gallery_names = np.asarray(im_names[self._num_query:])

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        if self.cfg.TEST.METRIC == "cosine":
            query_features = F.normalize(query_features, dim=1)
            gallery_features = F.normalize(gallery_features, dim=1)

        dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, gallery_features)

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA
            q_q_dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, query_features)
            g_g_dist = self.cal_dist(self.cfg.TEST.METRIC, gallery_features, gallery_features)
            re_dist = re_ranking(dist, q_q_dist, g_g_dist, k1, k2, lambda_value)
            if self.cfg.TEST.JSON_FILE:
                self.gen_json(re_dist, query_names, gallery_names)
                return {}
            else:
                query_features = query_features.numpy()
                gallery_features = gallery_features.numpy()
                cmc, all_AP, all_INP = evaluate_rank(re_dist, query_features, gallery_features,
                                                    query_pids, gallery_pids, query_camids,
                                                    gallery_camids, use_distmat=True)
        else:
            if self.cfg.TEST.JSON_FILE:
                self.gen_json(dist, query_names, gallery_names)
                return {}
            else:
                query_features = query_features.numpy()
                gallery_features = gallery_features.numpy()
                cmc, all_AP, all_INP = evaluate_rank(dist, query_features, gallery_features,
                                                    query_pids, gallery_pids, query_camids, gallery_camids,
                                                    use_distmat=False)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP

        if self.cfg.TEST.ROC_ENABLED:
            scores, labels = evaluate_roc(dist, query_features, gallery_features,
                                          query_pids, gallery_pids, query_camids, gallery_camids, use_cython=False)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)
