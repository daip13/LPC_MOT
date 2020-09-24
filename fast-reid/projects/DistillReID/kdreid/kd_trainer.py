# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from fastreid.engine import DefaultTrainer
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer
from .config import update_model_teacher_config
from .loss_kd import SimMatrixLoss, FeatDistanceMiningLoss, MaxL2Loss, FeatUncertainLoss
from .modeling.sigma import Sigma


class KDTrainer(DefaultTrainer):
    """
    A knowledge distillation trainer for person reid of task.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)

        model_t = self.build_model_teacher(self.cfg)
        for param in model_t.parameters():
            param.requires_grad = False

        logger = logging.getLogger('fastreid.'+__name__)

        # Load pre-trained teacher model
        print('88888\n'*10)
        logger.info("Loading teacher model ...")
        Checkpointer(model_t).load(cfg.MODEL.TEACHER_WEIGHTS)

        # Load pre-trained student model
        # logger.info("Loading student model ...")
        # Checkpointer(self.model, self.data_loader.dataset).load(cfg.MODEL.STUDENT_WEIGHTS)

        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        model_t.apply(set_bn_eval)

        self.model_t = model_t
        self.model_t.eval()

        if 'uncertainty' in cfg.MODEL.LOSSES.NAME:
            self.sigma = Sigma(n_input=cfg.MODEL.LOSSES.UNCERTAIN.IN_FEAT, n_output=cfg.MODEL.LOSSES.UNCERTAIN.OUT_FEAT)

    def run_step(self):
        """
        Implement the moco training logic described above.
        """
        assert self.model.training, "[KDTrainer] base model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)

        data_time = time.perf_counter() - start

        outputs, targets = self.model(data)

        # Compute reid loss
        loss_dict = {}
        # if isinstance(self.model, DistributedDataParallel):
        #     loss_dict = self.model.module.losses(outputs, targets)
        # else:
        #     loss_dict = self.model.losses(outputs, targets)

        with torch.no_grad():
            outputs_t = self.model_t(data)

        # loss_dict['loss_kl'] = self.distill_loss(outputs[1], outputs_t[1].detach())
        # loss_dict['loss_pkt'] = 1e4 * self.pkt_loss(outputs[1], outputs_t[1].detach())
        # loss_dict['FeatL2Loss'] = self.FeatL2Loss(outputs['feat'], outputs_t['feat'].detach())
        # loss = FeatDistanceMiningLoss()(outputs['feat'], outputs_t['feat'].detach(), targets, l2_norm=True, sim_mat=outputs_t['logits'].detach())
        loss = FeatUncertainLoss(self.cfg)(outputs['feat'], outputs_t['feat'].detach(), sigma=outputs['sigma'])
        # loss_dict['SimMatLoss'] = SimMatrixLoss(self.cfg)(outputs['feat'], outputs_t['feat'].detach())
        # loss = MaxL2Loss(self.cfg)(outputs['feat'], outputs_t['feat'].detach())
        loss_dict['FeatUncertainLoss'] = loss['loss']

        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        if 'weights' in loss:
            metrics_dict["weights"] = loss['weights']
        metrics_dict['images'] = data['images']
        metrics_dict['labels'] = targets
        metrics_dict['l2_dist'] = torch.sum((outputs['feat'].detach() - outputs_t['feat'].detach())*(outputs['feat'].detach() - outputs_t['feat'].detach()), dim=-1)
        self._write_metrics(metrics_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()

    @classmethod
    def build_model_teacher(cls, cfg) -> nn.Module:
        cfg_t = update_model_teacher_config(cfg)
        model_t = build_model(cfg_t)
        return model_t

    @staticmethod
    def pkt_loss(output_net, target_net, eps=0.0000001):
        # Normalize each vector by its norm
        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + eps)
        output_net[output_net != output_net] = 0

        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + eps)
        target_net[target_net != target_net] = 0

        # Calculate the cosine similarity
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

        # Scale cosine similarity to 0..1
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        # Transform them into probabilities
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))
        return loss

    @staticmethod
    def distill_loss(y_s, y_t, t=4):
        p_s = F.log_softmax(y_s / t, dim=1)
        p_t = F.softmax(y_t / t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (t ** 2) / y_s.shape[0]
        return loss

    @staticmethod
    def FeatL2Loss(f_s, f_t):
        return torch.mean(torch.sum((f_s - f_t)*(f_s - f_t), dim=-1))

# class FeatUncertainLoss(torch.nn.modules.loss._Loss):
#     '''
#     uncertainty model of feature KD
#     '''
#     def forward(self, output, target, label, l2_norm=False):
#         '''
#         Input:
#             output: embedding of student model, [B, 256]
#             target: embedding of teacher model, [B, 256]
#             label: Tensor[B], label of batch samples
#             l2_norm: bool, whether apply l2 normalization to feature
#         Return:
#             feat_loss: Tensor[B]
#             delta: Tensor[B]
#         '''
#         if l2_norm:
#             output = torch.nn.functional.normalize(output) # [B, 256]
#             target = torch.nn.functional.normalize(target) # [B, 256]
#         n_batch = output.size()[0]
#         pos_pair_idx = self.get_positive_pairs(label)   # [N, P]
#         pos_pair_target = target[pos_pair_idx]  # [N, P, 256]
#         pos_pair_output = output[pos_pair_idx]  # [N, P, 256]
#         delta = self.cal_delta(pos_pair_target)    # [N, P]
#         feat_loss = delta * torch.sum((pos_pair_output - pos_pair_target) * (pos_pair_output - pos_pair_target), dim=-1)
#         return torch.mean(torch.sum(feat_loss, dim=-1))
    
#     def get_positive_pairs(self, labels):
#         '''
#         get index of positive pairs
#         --------------------------
#         input:
#             labels: Tensor[B]
#         return:
#             Tensor[N, P], N*P=B
#         '''
#         labels = labels.cpu().data.numpy()
#         n_pairs = []

#         for label in set(labels):
#             label_mask = (labels == label)
#             label_indices = np.where(label_mask)[0]
#             if len(label_indices) < 2:
#                 continue
#             n_pairs.append(label_indices)

#         n_pairs = np.array(n_pairs)
#         return torch.LongTensor(n_pairs).cuda()

#     def cal_delta(self, pos_pair):
#         '''
#         measure uncertainty of samples by calculating distance between sample and class center.
#         ---------------------------
#         input:
#             pos_pair: Tensor[N, P, D], N is the number of pairs, P is the number of positive samples in a pair, D is number of feature dimensions
#         return:
#             Tensor[N, P], element ranges in (0, 1)
#         '''
#         # 1. calculate class center
#         center = torch.mean(pos_pair, dim=1, keepdim=True)  # [N, 1, D]
#         # 2. calculate distance to center
#         dist = torch.sum((pos_pair - center) * (pos_pair - center), dim=-1) # [N, P]
#         dist = torch.exp(dist - torch.max(dist, dim=1, keepdim=True)[0])
#         # 3. normalizarion
#         dist = dist / (torch.sum(dist, dim=1, keepdim=True) + 1e-10)
#         return dist