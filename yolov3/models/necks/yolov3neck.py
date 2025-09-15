import torch
import torch.nn as nn

from yolov3.models.common import (
    CNNBlock,
    ResidualBlock
)

from yolov3.models.heads import (
    ScalePrediction
)

class YOLOv3VanillaNeck(nn.Module):
    def __init__(self, in_maps : tuple):
        super().__init__()
        self.c3, self.c4, self.c5 = in_maps
        self.config = [
            [(512, 1, 1), (1024, 3, 1), (512, 1, 1), (1024, 3, 1), (512, 1, 1), "S"],
            [(256, 1, 1), "U", (256, 1, 1), (512, 3, 1), (256, 1, 1), (512, 3, 1), (256, 1, 1), "S"],
            [(128, 1, 1), "U", (128, 1, 1), (256, 3, 1), (128, 1, 1), (256, 3, 1), (128, 1, 1), "S"]
        ]
        architecture = []
        for block in self.config:
            subnet = []
            for module in block:
                if module isinstande


        

    def _create_layers(self):
        net = []

        for block in self.config:


            else:
                net.append(nn.Upsample(scale_factor=2))



