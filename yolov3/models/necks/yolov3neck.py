import torch
import torch.nn as nn

from yolov3.models.common import (
    CNNBlock,
    ResidualBlock
)

from yolov3.models.backbones.darknet53 import Darknet53

class YOLOv3VanillaNeck(nn.Module):
    def __init__(self, in_channels=(1024, 768, 384)):
        super().__init__()
        self.in_channels = in_channels
        self.config = [
            [(512, 1, 1), (1024, 3, 1), (512, 1, 1), (1024, 3, 1), (512, 1, 1)],
            [(256, 1, 1), (512, 3, 1), (256, 1, 1), (512, 3, 1), (256, 1, 1)],
            [(128, 1, 1), (256, 3, 1), (128, 1, 1), (256, 3, 1), (128, 1, 1)]
        ]

        self.reduce54 = nn.Sequential(
            CNNBlock(
                in_channels=512,
                out_channels=256,
                kernel_size=1,
                stride=1
            ),
            nn.Upsample(scale_factor=2)
        )

        self.reduce43 = nn.Sequential(
            CNNBlock(
                in_channels=256,
                out_channels=128,
                kernel_size=1,
                stride=1
            ),
            nn.Upsample(scale_factor=2)
        )

        architecture = []

        for i, block in enumerate(self.config):
            in_channels = self.in_channels[i]
            subnet = []
            for module in block:
                out_channels, kernel_size, stride = module 
                subnet.append(
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0
                    )
                )
                in_channels = out_channels

            architecture.append(nn.Sequential(*subnet))
        
        self.p5 = architecture[0]
        self.p4 = architecture[1]
        self.p3 = architecture[2]

    def forward(self, maps : tuple):
        C3, C4, C5 = maps
    
        P5 = self.p5(C5)
        C4 = torch.cat((self.reduce54(P5), C4), dim=1)

        P4 = self.p4(C4)
        C3 = torch.cat((self.reduce43(P4), C3), dim=1)

        P3 = self.p3(C3)

        return (P3, P4, P5)
        

def test():
    net = Darknet53()
    raw = torch.randn([1, 3, 416, 416])
    x = net(raw)

    neck = YOLOv3VanillaNeck()
    P3, P4, P5 = neck(x)

    assert P3.shape == torch.Size([1, 128, 52, 52])
    assert P4.shape == torch.Size([1, 256, 26, 26])
    assert P5.shape == torch.Size([1, 512, 13, 13])

    print("Test - All sizes good.")

if __name__ == "__main__":
    test()
