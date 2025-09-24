import torch
import torch.optim as optim
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

import os

from yolov3.models.backbones.darknet53 import Darknet53
from yolov3.models.necks.yolov3neck import YOLOv3VanillaNeck
from yolov3.models.heads.head import YOLOv3MultiScaleHead
from yolov3.models.models.yolov3 import YOLOv3

from yolov3.utils.utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    plot_couple_examples,
    read_config
)

from yolov3.models.builder import build_from_config
from yolov3.data.loader import get_loaders
from yolov3.utils.loss import YoloLoss

def train_fn(CONFIG, train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(CONFIG["device"])
        y0, y1, y2 = (
            y[0].to(CONFIG["device"]),
            y[1].to(CONFIG["device"]),
            y[2].to(CONFIG["device"]),
        )

        with torch.cuda.amp.autocast():
            P3, P4, P5 = model(x) # torch.Size([1, 3, 52, 52, 85]) torch.Size([1, 3, 26, 26, 85]) torch.Size([1, 3, 13, 13, 85])
            loss = (
                loss_fn(P5, y0, scaled_anchors[0])
                + loss_fn(P4, y1, scaled_anchors[1])
                + loss_fn(P3, y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

def main():

    cfg_name = "test.yaml" # !!!
    CONFIG = read_config(os.path.join('configs', cfg_name))

    if config["seed"]:
        seed_everything(42)

    model = build_from_config(CONFIG).to(CONFIG["device"])

    optimizer = optim.Adam(
        model.parameters(), lr=float(CONFIG["learning_rate"]), weight_decay=float(CONFIG["weight_decay"])
    )

    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, valid_loader, test_loader = get_loaders(CONFIG)

    if CONFIG["load_model"]:
        load_checkpoint(
            CONFIG["checkpoint_file"], model, optimizer, CONFIG["learning_rate"]
        )

    scaled_anchors = (
        torch.tensor(CONFIG["anchors"])
        * torch.tensor(CONFIG["S"]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(CONFIG["device"])

    for epoch in range(CONFIG["num_epochs"]):
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(CONFIG, train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        # if epoch > 0 and epoch % 3 == 0:
        #     check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
        #     pred_boxes, true_boxes = get_evaluation_bboxes(
        #         test_loader,
        #         model,
        #         iou_threshold=config.NMS_IOU_THRESH,
        #         anchors=config.ANCHORS,
        #         threshold=config.CONF_THRESHOLD,
        #     )
        #     mapval = mean_average_precision(
        #         pred_boxes,
        #         true_boxes,
        #         iou_threshold=config.MAP_IOU_THRESH,
        #         box_format="midpoint",
        #         num_classes=config.NUM_CLASSES,
        #     )
        #     print(f"MAP: {mapval.item()}")
        #     model.train()

if __name__ == "__main__":
    main()