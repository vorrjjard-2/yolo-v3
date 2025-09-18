import torch
import torch.nn as nn

@MODEL.register()
class YOLOv3(nn.Module):
    def __init__(self, backbone, neck, head, num_classes, anchors, strides):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        ...
    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        preds = self.head(feats)
        return preds
