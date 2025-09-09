"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import config
import numpy as np
import os
import pandas as pd
import torch

import cv2

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    pre_index,
    normalize_bboxes,
    denormalize_bboxes,
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        split,
        anchors,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        self.split = split
        self.id_annotations, self.id_images, self.id_categories = pre_index(
            os.path.join('datasets', config.DATASET, self.split, '_annotations.coco.json')
        )
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.id_annotations)


    def __getitem__(self, index):
        # FROM HERE

        raw_path, W, H = self.id_images[index]
        raw_bboxes = self.id_annotations[index]

        bboxes = normalize_bboxes(raw_bboxes, W, H)
        image = np.array(cv2.cvtColor(cv2.imread(os.path.join(config.DATA_ROOT, config.DATASET, self.split, raw_path)), cv2.COLOR_BGR2RGB))
    
        class_labels = [box[0] for box in bboxes]
        coords_only = [box[1:] for box in bboxes]
        
        if self.transform:
            augmented = self.transform(image=image, bboxes=coords_only, class_labels=class_labels)
            image = augmented["image"]
            coords_only = augmented["bboxes"]
            class_labels = augmented["class_labels"]
            bboxes = [list(coord) + [cls] for cls, coord in zip(class_labels, coords_only)]

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)


def test():
    anchors = config.ANCHORS

    transform = config.test_transforms

    dataset = YOLODataset(
        'train',
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    x, y = next(iter(loader))
    boxes = []

    for i in range(y[0].shape[1]):
        anchor = scaled_anchors[i]
        boxes += cells_to_bboxes(
            y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
        )[0]



    boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
    plot_image(x[0], boxes)

if __name__ == "__main__":
    test()