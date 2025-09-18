import torch
import torch.nn as nn

from yolov3.models.registry import (
    MODEL,
    BACKBONE,
    NECK,
    HEAD,
    LOSS,
    DATASET,
    OPTIMIZER,
    SCHEDULER
)

def build_from_config(cfg):
    backbone_cls = BACKBONE[cfg["backbone"]["type"]]
    backbone = backbone_cls(**cfg["backbone"].get("kwargs", {}))

    neck_cls = NECK[cfg["neck"]["type"]]
    neck = neck_cls(**cfg["neck"].get("kwargs", {}))

    head_cls = HEAD[cfg["head"]["type"]]
    head = head_cls(**cfg["head"].get("kwargs", {}))

    model_cls = MODEL[cfg["model"]["type"]]
    model = model_cls(backbone=backbone, neck=neck, head=head, **cfg["model"].get("kwargs", {}))

    return model
