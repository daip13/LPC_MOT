# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_classifier
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class LinearHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer):
        super().__init__()
        self.pool_layer = pool_layer

        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        self.classifier = None
        if cls_type == 'linear':          self.classifier = nn.Linear(in_feat, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':    self.classifier = ArcSoftmax(cfg, in_feat, num_classes)
        elif cls_type == 'circleSoftmax': self.classifier = CircleSoftmax(cfg, in_feat, num_classes)
        elif cls_type == 'amSoftmax':     self.classifier = AMSoftmax(cfg, in_feat, num_classes)
        else:
            print(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcSoftmax', 'amSoftmax' and 'circleSoftmax'.")
        if self.classifier:
            self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None, extra_feat=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        global_feat = global_feat.view(global_feat.size(0), -1)

        # Evaluation
        if not self.training: return {'feat': global_feat}
        if self.classifier:
            # Training
            try:              cls_outputs = self.classifier(global_feat)
            except TypeError: cls_outputs = self.classifier(global_feat, targets)

            pred_class_logits = F.linear(global_feat, self.classifier.weight)

            return {'cls': cls_outputs, 'logits': pred_class_logits, 'feat': global_feat}
        else:
            return {'cls': None, 'logits': None, 'feat': global_feat}