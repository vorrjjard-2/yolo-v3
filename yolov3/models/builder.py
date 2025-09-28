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
    # Backbone
    backbone_cls = BACKBONE[cfg["backbone"]["type"]]
    backbone_kwargs = cfg["backbone"].get("kwargs", {})
    print("[Backbone]", backbone_cls.__name__, backbone_kwargs)
    backbone = backbone_cls(**backbone_kwargs)

    # Neck
    neck_cls = NECK[cfg["neck"]["type"]]
    neck_kwargs = cfg["neck"].get("kwargs", {})
    print("[Neck]", neck_cls.__name__, neck_kwargs)
    neck = neck_cls(**neck_kwargs)

    # Head
    head_cls = HEAD[cfg["head"]["type"]]
    head_kwargs = cfg["head"].get("kwargs", {})
    print("[Head]", head_cls.__name__, head_kwargs)
    head = head_cls(**head_kwargs)

    # Model
    model_cls = MODEL[cfg["model"]["type"]]
    model_kwargs = cfg["model"].get("kwargs", {})
    print("[Model]", model_cls.__name__, model_kwargs)
    model = model_cls(
        backbone=backbone,
        neck=neck,
        head=head,
        **model_kwargs
    )

    return model
