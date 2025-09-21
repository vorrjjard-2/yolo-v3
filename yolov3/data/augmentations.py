import albumentations as A
import cv2

from albumentations.pytorch import ToTensorV2

def build_transforms(cfg : dict, split : str):
    IMAGE_SIZE = cfg["image_size"]
    scale = 1.1

    transforms = {}

    transforms["train"] = A.Compose(
        [
            A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
            A.PadIfNeeded(
                min_height=int(IMAGE_SIZE * scale),
                min_width=int(IMAGE_SIZE * scale),
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
            A.OneOf(
                [
                    A.ShiftScaleRotate(
                        rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                    ),
                    A.Affine(shear=15, p=0.5, mode="constant"),
                ],
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Blur(p=0.1),
            A.CLAHE(p=0.1),
            A.Posterize(p=0.1),
            A.ToGray(p=0.1),
            A.ChannelShuffle(p=0.05),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
    )
    transforms["valid"] = A.Compose(
            [
                A.LongestMaxSize(max_size=IMAGE_SIZE),
                A.PadIfNeeded(
                    min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
                ),
                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
        )

    transforms["test"] = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_SIZE),
            A.PadIfNeeded(
                min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
    )

    assert split in transforms

    return transforms[split]
