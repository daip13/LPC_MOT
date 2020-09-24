# encoding: utf-8
"""
@author:  Yang Qian
@contact: yqian@aibee.com
"""

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class SigmaHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer):
        super().__init__()
        self._cfg = cfg
        reduction_dim = cfg.MODEL.HEADS.REDUCTION_DIM
        self.neck_feat = cfg.MODEL.HEADS.NECK_FEAT

        self.pool_layer = pool_layer

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_feat, reduction_dim, 1, 1, bias=False),
            get_norm(cfg.MODEL.HEADS.NORM, reduction_dim, cfg.MODEL.HEADS.NORM_SPLIT),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.sigma = nn.Sequential(
            nn.Conv2d(in_feat, 1, 1, 1, bias=False),
            get_norm(cfg.MODEL.HEADS.NORM, 1, cfg.MODEL.HEADS.NORM_SPLIT),
        )

        self.bnneck = get_norm(cfg.MODEL.HEADS.NORM, reduction_dim, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True)

        self.bottleneck.apply(weights_init_kaiming)
        self.bnneck.apply(weights_init_kaiming)

        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        if cls_type != '':
            if cls_type == 'linear':          self.classifier = nn.Linear(reduction_dim, num_classes, bias=False)
            elif cls_type == 'arcSoftmax':    self.classifier = ArcSoftmax(cfg, reduction_dim, num_classes)
            elif cls_type == 'circleSoftmax': self.classifier = CircleSoftmax(cfg, reduction_dim, num_classes)
            elif cls_type == 'amSoftmax':     self.classifier = AMSoftmax(cfg, reduction_dim, num_classes)
            else:
                raise KeyError(f"{cls_type} is invalid, please choose from "
                            f"'linear', 'arcSoftmax', 'amSoftmax' and 'circleSoftmax'.")

            self.classifier.apply(weights_init_classifier)
        self.cls_type = cls_type

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        features = self.pool_layer(features)
        global_feat = self.bottleneck(features)
        sigma = self.sigma(features)
        bn_feat = self.bnneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]

        if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
        elif self.neck_feat == "after": feat = bn_feat
        else:
            raise KeyError("MODEL.HEADS.NECK_FEAT value is invalid, must choose from ('after' & 'before')")
        # Evaluation
        if not self.training: return {'feat':bn_feat, 'logits':F.linear(F.normalize(bn_feat), F.normalize(self.classifier.weight)) if self.cls_type != '' else None, 'sigma': sigma}

        # Training
        if self.cls_type != '':
            try:              cls_outputs = self.classifier(bn_feat)
            except TypeError: cls_outputs = self.classifier(bn_feat, targets)

            pred_class_logits = F.linear(bn_feat, self.classifier.weight)

            return {'cls': cls_outputs, 'logits': pred_class_logits, 'feat': feat, 'sigma': sigma}
        else:
            return {'feat': feat, 'sigma': sigma}

