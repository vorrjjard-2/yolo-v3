import torch
import torch.nn as nn

from yolov3.models.common import (
    CNNBlock,
    ResidualBlock
)

from yolov3.models.registry import BACKBONE


@BACKBONE.register()
class Darknet53(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.config = [(64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)]

        self.in_channels = in_channels
        self.stem = CNNBlock(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)

        architecture = []
        in_channels = 32

        for out_channels, num_repeats in self.config:
            architecture.append(self._create_layer(
                in_channels=in_channels,
                out_channels=out_channels, 
                num_repeats=num_repeats
                ))
            in_channels = out_channels

        self.net = nn.ModuleList(architecture)

    def _create_layer(self, in_channels, out_channels, num_repeats):
        sublayers = []

        sublayers.append(
            CNNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ))

        sublayers.append(
            ResidualBlock(
                channels=out_channels,
                use_residual=True,
                num_repeats=num_repeats
            )
        )
        
        return nn.Sequential(*sublayers)

    def forward(self, x):
        features = []
        map_indices = (2, 3, 4)

        x = self.stem(x)
        
        for i, stage in enumerate(self.net):
            x = stage(x)
            if i in map_indices:
                features.append(x)
        
        return tuple(features)

def test():
    net = Darknet53()
    raw = torch.randn([1, 3, 416, 416])
    f1, f2, f3 = net(raw)

    assert f1.shape == torch.Size([1, 256, 52, 52])
    assert f2.shape == torch.Size([1, 512, 26, 26])
    assert f3.shape == torch.Size([1, 1024, 13, 13])

    print("Test - Feature map shapes are correct.")

if __name__ == "__main__":
    test()