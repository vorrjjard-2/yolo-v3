import torch
import torch.nn as nn

from yolov3.models.common import (
    CNNBlock,
    ResidualBlock
)

class Darknet53(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.stem = nn.Sequential(
            [CNNBlock(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=1, padding=1), 
             CNNBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)])

        self.block2 = _create_layers(64, )


    def _create_layers(self, type, in_channels, out_channels):
        in_channels = self.in_channels 
        


    def forward: