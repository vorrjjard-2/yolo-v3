import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from yolov3.data.dataset import YOLODataset

def get_loaders(CONFIG):
    train = YOLODataset(CONFIG, split='train', transform=True)
    valid = YOLODataset(CONFIG, split='valid', transform=True)
    test = YOLODataset(CONFIG, split='test', transform=True)

    train_loader = DataLoader(dataset=train, batch_size=CONFIG["batch_size"], shuffle=True)
    valid_loader = DataLoader(dataset=valid, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=CONFIG["batch_size"], shuffle=True)

    return train_loader, valid_loader, test_loader