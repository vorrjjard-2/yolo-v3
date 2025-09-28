import torch
import torch.nn as nn

from yolov3.models.registry import MODEL

@MODEL.register()
class YOLOv3(nn.Module):
    def __init__(self, backbone, neck, head):
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
