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
from .build import LOSS_REGISTRY

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return torch.mean(cross_entropy_loss)

class KLKDLoss(torch.nn.modules.loss._Loss):
    ''' distillation loss for classification '''
    def forward(self, student, teacher):
        return 4*4*torch.nn.KLDivLoss()(F.log_softmax(student / 4, dim=1), F.softmax(teacher / 4, dim=1))

def KL(temperature):
    def kl_loss(student_outputs, teacher_outputs):
        loss = torch.nn.KLDivLoss(size_average=False, reduce=False)(
            F.log_softmax(student_outputs / temperature, dim=1),
            F.softmax(teacher_outputs.detach() / temperature, dim=1)) \
                * (temperature * temperature)
        return torch.mean(torch.sum(loss, dim=-1))
    return kl_loss

def DistanceWeightedSampling(distance, n_dim, cutoff=0.5, normalize=True):
    '''
    paper: Sampling Matters in Deep Embedding Learning
    ref: https://github.com/suruoxi/DistanceWeightedSampling
    ------
    distance: tensor, embeddings, [B, 256]
    n_dim: scale, dimension of feature
    cutoff: scale, threshold for minimum distance
    normalize: bool, whether to normalize weights
    '''
    distance_ = distance.detach()
    n = distance_.shape
    d = n_dim
    log_weights = ((2.0 - float(d)) * distance_.log() - (float(d-3)/2)*torch.log(torch.clamp(1.0 - 0.25*(distance_*distance_), min=1e-8)))
    if normalize:
        log_weights = (log_weights - log_weights.min()) / (log_weights.max() - log_weights.min() + 1e-8)
    weights = torch.exp(log_weights - torch.max(log_weights))
    weights_sum = torch.sum(weights)
    weights = weights / weights_sum
    np_weights = weights.cpu().numpy()
    return np.random.choice(n, n, p=np_weights)

@LOSS_REGISTRY.register()
class FeatL2Loss(object):
    """ L2 distance between teacher and student, with many variants """
    def __init__(self, cfg):
        self.l2_norm = cfg.MODEL.LOSSES.FEATL2.L2_NORM  # whether to apply l2 normalization to feature 
        self.DWS = cfg.MODEL.LOSSES.FEATL2.DWS # whether to use distance weighted sampling method
        self.DISTANCE_MINING = cfg.MODEL.LOSSES.FEATL2.DISTANCE_MINING # whether to use distance among positive pair to weight loss
        self.transform_matrix = cfg.MODEL.LOSSES.FEATL2.TRANSFORM
        self.transform_matrix = np.loadtxt(self.transform_matrix).T if self.transform_matrix else None
        self.weight = cfg.MODEL.LOSSES.FEATL2.WEIGHT  # whether to use PCA dimension weights
        self.weight = np.loadtxt(self.weight) if self.weight else None
    def __call__(self, output, target, label=None):
        r'''
        output: embedding of student model, [B, 256]
        target: embedding of teacher model, [B, 256]
        weight: tranformation matrix of PCA, [256, 256]
        l2_norm: bool, whether apply l2 normalization to feature
        DWS: bool, whether use distance weighted sampling
        '''
        assert len(target)==1, 'L2Loss do not support multi teacher now'
        target = target[0]
        if self.l2_norm:
            output = torch.nn.functional.normalize(output) # [B, 256]
            target = torch.nn.functional.normalize(target) # [B, 256]

        if self.DISTANCE_MINING:
            pos_pair_idx = self.get_positive_pairs(label)   # [N, P]
            target = target[pos_pair_idx]  # [N, P, 256]
            output = output[pos_pair_idx]

        # if comm.get_world_size() > 1:
        #     output = concat_all_gather(output)
        #     target = concat_all_gather(target)

        n_dim = output.size()[-1]
        # euclidean distance
        if self.weight is None:
            loss = torch.sum((output - target) * (output - target), dim=-1)
        else:
            if not isinstance(self.weight, torch.FloatTensor):
                self.weight = torch.FloatTensor(self.weight).to(output.device)
                self.transform_matrix = torch.FloatTensor(self.transform_matrix).to(output.device)
            output = output.mm(self.transform_matrix)
            target = target.mm(self.transform_matrix)
            loss = 256*torch.sum((output - target) * (output - target) * self.weight.unsqueeze(0), dim=-1)
        
        if self.DWS:
            index = DistanceWeightedSampling(loss, n_dim=n_dim)
            loss = loss[index]
        if self.DISTANCE_MINING:
            delta = self.cal_delta(target)
            loss = delta * loss
        return loss

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

    def cal_delta(self, pos_pair):
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
        dist = torch.exp(dist - torch.max(dist, dim=1, keepdim=True)[0])
        # 3. normalizarion
        dist = dist / (torch.sum(dist, dim=1, keepdim=True) + 1e-10)
        return dist

@LOSS_REGISTRY.register()
class FeatUncertainLoss(object):
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

        if self.DWS:
            index = DistanceWeightedSampling(loss, n_dim=n_dim)
            loss = loss[index]
        return loss
        

@LOSS_REGISTRY.register()
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
        return loss

@LOSS_REGISTRY.register()
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
        n_dim = output.size()[-1]
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
        
        if self.DWS:
            index = DistanceWeightedSampling(loss, n_dim=n_dim)
            loss = loss[index]
        return loss

@LOSS_REGISTRY.register()
class AttentionLoss(object):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""
    def __init__(self, cfg):
        self.p = cfg.MODEL.LOSSES.ATTENTION.P # p=2

    def __call__(self, g_s, g_t):
        return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        s_W, t_W = f_s.shape[3], f_t.shape[3]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_W))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_W))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))

class LikelihoodLoss(torch.nn.modules.loss._Loss):
    '''
    implement likelihood loss by modeling embedding through Gaussian distributions.
    reference paper: Data Uncertainty Learning in Face Recognition
    '''
    def forward(self, output, target, r):
        r'''
        output: embedding of student model, [B, 256]
        target: embedding of teacher model, [B, 256]
        r: logarithm of variance \delta, [B, 256]
        '''
        loss = torch.mean(torch.exp(-r) * ((output - target) * (output - target)) + r)
        return loss

class RankLoss(torch.nn.Module):
    def __init__(self):
        super(RankLoss, self).__init__()

    def forward(self, feat_t, feat_s, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if feat_t.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = feat_t[n_pairs[:, 0]]    # (n, embedding_size)
        positives = feat_t[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = feat_t[n_negatives]    # (n, n-1, embedding_size)

        prob_t = self.n_pair_loss(anchors, positives, negatives)

        anchors = feat_s[n_pairs[:, 0]]    # (n, embedding_size)
        positives = feat_s[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = feat_s[n_negatives]    # (n, n-1, embedding_size)

        prob_s = self.n_pair_loss(anchors, positives, negatives)
        return torch.sum(torch.nn.functional.kl_div(prob_s.log(), prob_t, reduction='none'))

    @staticmethod
    def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
                        and n_negatives (n, n-1)
        """
        labels = labels.cpu().data.numpy()
        n_pairs = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor, positive = np.random.choice(label_indices, 2, replace=False)
            # anchor, positive = label_indices[0], label_indices[1]
            n_pairs.append([anchor, positive])

        n_pairs = np.array(n_pairs)

        n_negatives = []
        for i in range(len(n_pairs)):
            negative = np.concatenate([n_pairs[:i, 1], n_pairs[i+1:, 1]])
            n_negatives.append(negative)

        n_negatives = np.array(n_negatives)

        return torch.LongTensor(n_pairs), torch.LongTensor(n_negatives)

    @staticmethod
    def n_pair_loss(anchors, positives, negatives, margin=None, binary=False):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        if margin is None:
            if binary is False:
                x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
                x *= 40
            else:
                x = torch.sum(anchors ^ negatives - anchors ^ positives, dim=1).unsqueeze(1) # (n, 1, n-1)
        else:
            # add ArcFace loss
            cos_m, sin_m = math.cos(margin), math.sin(margin)
            cos_theta = torch.matmul(anchors/200, positives.transpose(1,2)/200).clamp(-1, 1)
            sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
            x = torch.matmul(anchors, negatives.transpose(1,2))-200*200*cos_theta_m
            x *= 40
        # x *= 1e-3
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        return torch.nn.functional.softmax(torch.log(1+x).squeeze(), dim=0)

def pdist(e, squared=False, eps=1e-12):
    n_batch = e.size()[0]
    e = e.view(n_batch, -1)
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

class HardDarkRank(torch.nn.Module):
    def __init__(self, alpha=3, beta=3, permute_len=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * pdist(teacher, squared=False).pow(self.beta)
        score_student = -1 * self.alpha * pdist(student, squared=False).pow(self.beta)

        permute_idx = score_teacher.sort(dim=1, descending=True)[1][:, 1:(self.permute_len+1)]
        ordered_student = torch.gather(score_student, 1, permute_idx)

        log_prob = (ordered_student - torch.stack([torch.logsumexp(ordered_student[:, i:], dim=1) for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()

        return loss

class RkdDistance(torch.nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            # mean_td = t_d[t_d>0].mean()
            # t_d = t_d / mean_td
            t_d = t_d[t_d>0].mean()

        d = pdist(student, squared=False)
        # mean_d = d[d>0].mean()
        # d = d / mean_d
        d = d[d>0].mean()

        loss = torch.nn.functional.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss

class RKdAngle(torch.nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C
        n_batch = student.size()[0]
        student = student.view(n_batch, -1)
        teacher = teacher.view(n_batch, -1)
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = torch.nn.functional.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = torch.nn.functional.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = torch.nn.functional.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss
