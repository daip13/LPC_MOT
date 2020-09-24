# encoding: utf-8
"""
<<<<<<< HEAD
@author:  HeZhangping
@contact: zphe@aibee.com
"""
import random
import torch
import torch.nn.functional as F

from fastreid.utils import comm
from .utils import concat_all_gather, euclidean_dist, normalize
"""
@author:  Zhangping He, Yang Qian
@contact: {zphe, yqian}@aibee.com
"""

import torch
import random
import torch.nn.functional as F

from fastreid.utils import comm
from .utils import concat_all_gather

class NpairLoss(object):
    """
    N-Pair loss
    Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
    Processing Systems. 2016.
    http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective
    """

    def __init__(self, cfg):
        
        super(NpairLoss, self).__init__()
        self.margin = cfg.MODEL.LOSSES.NPAIR.MARGIN
        self.l2_reg = cfg.MODEL.LOSSES.NPAIR.L2_REG
        self.scale = cfg.MODEL.LOSSES.NPAIR.SCALE
        self.hard_mining = cfg.MODEL.LOSSES.NPAIR.HARD_MINING
        
    def __call__(self, feats, labels):
        feats = F.normalize(feats, dim=1)
        all_feats, all_labels = None, None
        if comm.get_world_size() > 1:
            all_feats = concat_all_gather(feats)
            all_labels = concat_all_gather(labels)
        else:
            all_feats = feats
            all_labels = labels

        batch_size = feats.size(0)
        sim_mat = torch.mm(feats, all_feats.t())
        losses = []
        rank = comm.get_rank()
        for i in range(batch_size):
            pos_idxs = (all_labels == labels[i])
            pos_idxs[rank*batch_size + i] = False
            pos_pair_ = sim_mat[i][pos_idxs]
            neg_pair_ = sim_mat[i][all_labels != labels[i]]
            if self.hard_mining:
                neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
                pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]
            else:
                neg_pair = neg_pair_
                pos_pair = pos_pair_

            if len(pos_pair) < 1 or len(neg_pair) < 1:
                continue
            if len(pos_pair) > 1:
                pos_pair = pos_pair[random.randint(0, len(pos_pair) - 1)]
            else:
                pos_pair = pos_pair[0]
            # print('neg_pair:', neg_pair)
            # print('pos_pair:', pos_pair)
            loss = torch.log(1 + torch.sum(torch.exp(self.scale * (neg_pair - pos_pair)))) 
            losses.append(loss)
        return sum(losses) / len(losses)
