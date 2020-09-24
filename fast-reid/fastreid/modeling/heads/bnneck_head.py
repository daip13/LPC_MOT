# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class BNneckHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer):
        super().__init__()
        self.neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        self.pool_layer = pool_layer
        extra_feat_dim = cfg.MODEL.HEADS.EXTRA_FEAT

        self.bnneck = get_norm(cfg.MODEL.HEADS.NORM, in_feat+extra_feat_dim, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True)
        self.bnneck.apply(weights_init_kaiming)

        # identity classification layer
        self.classifier = None
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        self.cls_type = cls_type
        if cls_type == 'linear':          self.classifier = nn.Linear(in_feat+extra_feat_dim, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':    self.classifier = ArcSoftmax(cfg, in_feat+extra_feat_dim, num_classes)
        elif cls_type == 'circleSoftmax': self.classifier = CircleSoftmax(cfg, in_feat+extra_feat_dim, num_classes)
        elif cls_type == 'amSoftmax':     self.classifier = AMSoftmax(cfg, in_feat+extra_feat_dim, num_classes)
        elif cls_type == '':
            print('no classifier')
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcSoftmax', 'amSoftmax' and 'circleSoftmax'.")

        if cls_type != '':
            self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None, extra_feat=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        if extra_feat is not None:
            global_feat = torch.cat((global_feat, extra_feat), dim=1)
        bn_feat = self.bnneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]

        # Evaluation
        if not self.training: return {'feat':bn_feat, 'logits':F.linear(F.normalize(bn_feat), F.normalize(self.classifier.weight)) if self.cls_type != '' else None}

        if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
        elif self.neck_feat == "after": feat = bn_feat
        else:
            raise KeyError("MODEL.HEADS.NECK_FEAT value is invalid, must choose from ('after' & 'before')")

        # Training
        if self.cls_type != '':
            try:              cls_outputs = self.classifier(bn_feat)
            except TypeError: cls_outputs = self.classifier(bn_feat, targets)
            pred_class_logits = F.linear(bn_feat, self.classifier.weight)

            return {'cls':cls_outputs, 'logits':pred_class_logits, 'feat':feat}
        else:
            return {'cls': None, 'logits': None, 'feat': feat}
