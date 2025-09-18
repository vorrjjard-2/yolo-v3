import torch
import torch.nn as nn

from yolov3.models.common import(
    CNNBlock
)

from yolov3.models.registry import HEAD


@HEAD.register()
class YOLOv3Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

@HEAD.register()
class YOLOv3MultiScaleHead(nn.Module):
    def __init__(self, in_channels_list, num_classes):
        super().__init__()
        self.heads = nn.ModuleList([
            YOLOv3Head(in_ch, num_classes) for in_ch in in_channels_list
        ])

    def forward(self, features): 
        return [head(f) for head, f in zip(self.heads, features)]
