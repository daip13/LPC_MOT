'''
loss function library for knowledge distillation
--------
Author: Yang Qian
Email: yqian@aibee.com
'''
import torch
import torch.nn.functional as F
import math
import numpy as np

from fastreid.utils import comm

class FeatDistanceMiningLoss(torch.nn.modules.loss._Loss):
    '''
    Feature distance mining model of feature KD
    '''
    def forward(self, output, target, label, sim_mat=None, l2_norm=False):
        '''
        Input:
            output: embedding of student model, [B, 256]
            target: embedding of teacher model, [B, 256]
            label: Tensor[B], label of batch samples
            l2_norm: bool, whether apply l2 normalization to feature
        Return:
            feat_loss: Tensor[B]
            delta: Tensor[B]
        '''
        n_batch = output.size()[0]
        pos_pair_idx = self.get_positive_pairs(label)   # [N, P]
        pos_pair_target = target[pos_pair_idx]  # [N, P, 256]
        pos_pair_output = output[pos_pair_idx]  # [N, P, 256]
        if l2_norm:
            output = torch.nn.functional.normalize(output) # [B, 256]
            target = torch.nn.functional.normalize(target) # [B, 256]
        if sim_mat is not None:
            delta = self.cal_delta_by_logits(sim_mat, label, pos_pair_idx)
        else:
            delta = self.cal_delta(target[pos_pair_idx])    # [N, P]
        feat_loss = delta * torch.sum((pos_pair_output - pos_pair_target) * (pos_pair_output - pos_pair_target), dim=-1)
        return {'loss': torch.mean(torch.sum(feat_loss, dim=-1)), 'weights': delta.view(n_batch)}
    
    def get_positive_pairs(self, labels):
        '''
        get index of positive pairs
        --------------------------
        input:
            labels: Tensor[B]
        return:
            Tensor[N, P], N*P=B
        '''
        labels = labels.cpu().data.numpy()
        n_pairs = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            n_pairs.append(label_indices)

        n_pairs = np.array(n_pairs)
        return torch.LongTensor(n_pairs).cuda()

    def cal_delta(self, pos_pair, scale=10):
        '''
        measure uncertainty of samples by calculating distance between sample and class center.
        ---------------------------
        input:
            pos_pair: Tensor[N, P, D], N is the number of pairs, P is the number of positive samples in a pair, D is number of feature dimensions
        return:
            Tensor[N, P], element ranges in (0, 1)
        '''
        # 1. calculate class center
        center = torch.mean(pos_pair, dim=1, keepdim=True)  # [N, 1, D]
        # 2. calculate distance to center
        dist = torch.sum((pos_pair - center) * (pos_pair - center), dim=-1) # [N, P]
        dist = torch.exp(scale * (dist - torch.max(dist, dim=1, keepdim=True)[0]))
        # 3. normalizarion
        dist = dist / (torch.sum(dist, dim=1, keepdim=True) + 1e-10)
        return dist

    def cal_delta_by_logits(self, logits, label, pos_pair_idx, scale=10):
        '''
        measure sample quality based on logits
        '''
        n_class = logits.size()[-1]
        target = F.one_hot(label, num_classes=n_class)
        dist = torch.sum(target * logits, dim=-1)   # [B], easy mining
        dist = dist[pos_pair_idx]   # [N, P]
        dist = torch.exp(scale * (dist - torch.max(dist, dim=1, keepdim=True)[0]))
        # normalization
        dist = dist / (torch.sum(dist, dim=1, keepdim=True) + 1e-10)
        return dist

class SimMatrixLoss(object):
    """
    compare similarity matrix of embedding between teacher and student
    Well, it is same as Relational knowledge distillation
    """
    def __init__(self, cfg):
        self.PKT = cfg.MODEL.LOSSES.SM.PKT # whether to use PKT strategy
        self.DWS = cfg.MODEL.LOSSES.SM.DWS
        self.eps = cfg.MODEL.LOSSES.SM.EPS
    def __call__(self, output, target, label=None):
        '''
        output: tensor, embedding or intermediate feature map of student model
        target: list of tensor, embedding or intermediate feature map of multiple teacher model
        PKT: bool, if True, use PKT strategy; otherwise, no.
        '''
        n_batch = output.size()[0]
        output = torch.nn.functional.normalize(output.view(n_batch, -1)) # [B, 256]
        target = torch.cat(target, dim=-1)
        target = torch.nn.functional.normalize(target.view(n_batch, -1)) # [B, 256]

        # if comm.get_world_size() > 1:
        #     output = concat_all_gather(output)
        #     target = concat_all_gather(target)

        # euclidean distance
        sim_mat_output = output @ output.t()
        sim_mat_target = target @ target.t()
        if self.PKT:
            """learning deep representations by probbilistic knowledge transfer
            Code from author: https://github.com/passalis/probabilistic_kt"""
            # Scale cosine similarity to 0..1
            sim_mat_output = (sim_mat_output + 1.0) / 2.0
            sim_mat_target = (sim_mat_target + 1.0) / 2.0

            # Transform them into probabilities
            sim_mat_output = sim_mat_output / torch.sum(sim_mat_output, dim=1, keepdim=True)
            sim_mat_target = sim_mat_target / torch.sum(sim_mat_target, dim=1, keepdim=True)
            loss = sim_mat_target * torch.log((sim_mat_target + self.eps) / (sim_mat_output + self.eps))
        else:
            loss = (sim_mat_output - sim_mat_target) * (sim_mat_output - sim_mat_target)
            loss =  torch.triu(loss, diagonal=1).view(-1)
        return loss


class MaxL2Loss(object):
    r'''
    Instead of calculating sum of l2 loss of all samples, only select sample with maximum l2 loss.
    This can be regarded as a way of hard sample mining.
    max \sim LogSumExp
    ----------
    output: embedding of student model, [B, 256]
    target: embedding of teacher model, [B, 256]
    weight: importance of each dim of embedding calculated by PCA, [256]
    s: scale, influence convergence
    '''
    def __init__(self, cfg):
        self.s = cfg.MODEL.LOSSES.MAXL2.scale   # s=10
        self.weight = None

    def __call__(self, output, target, label=None):
        output = torch.nn.functional.normalize(output) # [B, 256]
        target = torch.nn.functional.normalize(target) # [B, 256]

        # if comm.get_world_size() > 1:
        #     output = concat_all_gather(output)
        #     target = concat_all_gather(target)

        n_batch = output.size()[0]

        if self.weight is None:
            loss = torch.log(torch.sum(torch.exp(self.s*torch.sum((output - target) * (output - target), dim=1)), dim=0))
        else:
            loss = torch.log(torch.sum(torch.exp(self.s*torch.sum((output - target) * (output - target) * self.weight * 256, dim=1)), dim=0)) - np.log(n_batch)
        return {'loss': loss}

class FeatUncertainLoss(object):
    r'''
    Model sample uncertainty by Gaussian distribution
    '''
    def __init__(self, cfg):
        self.l2_norm = cfg.MODEL.LOSSES.FEATL2.L2_NORM  # whether to apply l2 normalization to feature 
        self.DWS = cfg.MODEL.LOSSES.FEATL2.DWS # whether to use distance weighted sampling method
        self.transform_matrix = cfg.MODEL.LOSSES.FEATL2.TRANSFORM
        self.transform_matrix = np.loadtxt(self.transform_matrix).T if self.transform_matrix else None
        self.weight = cfg.MODEL.LOSSES.FEATL2.WEIGHT  # whether to use PCA dimension weights
        self.weight = np.loadtxt(self.weight) if self.weight else None
    def __call__(self, output, target, sigma):
        if self.l2_norm:
            output = torch.nn.functional.normalize(output) # [B, 256]
            target = torch.nn.functional.normalize(target) # [B, 256]
        # if comm.get_world_size() > 1:
        #     output = concat_all_gather(output)
        #     target = concat_all_gather(target)
        
        n_dim = output.size()[-1]
        sigma = sigma.squeeze()
        if self.weight is None:
            if len(sigma.size()) == 1:
                loss = torch.sum((output - target) * (output - target), dim=1) * torch.exp(-sigma) + sigma + np.log(2*np.pi)
            elif len(sigma.size()) == 2:
                loss = torch.sum((output - target) * (output - target) * torch.exp(-sigma), dim=1) + torch.sum(sigma, dim=1)
                sigma = torch.sum(sigma, dim=-1)
        else:
            if not isinstance(self.weight, torch.FloatTensor):
                self.weight = torch.FloatTensor(self.weight).to(output.device)
                self.transform_matrix = torch.FloatTensor(self.transform_matrix).to(output.device)
            if len(sigma.size()) == 1:  # dimension weighting
                output = output.mm(self.transform_matrix)
                target = target.mm(self.transform_matrix)
                loss = n_dim * torch.sum((output - target) * (output - target) * self.weight.unsqueeze(0), dim=1) * torch.exp(-sigma) + sigma + np.log(2*np.pi)
            elif len(sigma.size()) == 2:    # dimension weighting is valid in this situation
                loss = torch.sum((output - target) * (output - target) * torch.exp(-sigma), dim=1) + torch.sum(sigma, dim=1)
                sigma = torch.sum(sigma, dim=-1)
        return {'loss': torch.mean(loss), 'weights': torch.sqrt(torch.exp(sigma))}