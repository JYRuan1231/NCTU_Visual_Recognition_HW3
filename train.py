import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import transforms, utils, models
import glob
import os
import pickle
from PIL import Image
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import csv
import time
import os
import math
import copy
from pycocotools.coco import COCO
from pycocotools.mask import encode
from pycocotools.cocoeval import COCOeval
from data_augment.autoaugment import ImageNetPolicy
from utils import binary_mask_to_rle
from data_transform import transform, draw_rect
import random
import config as cfg
from random import sample
from torch.utils.data import Dataset, DataLoader, random_split
from model import backboneWithFPN, backboneNet_efficient
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork,
    LastLevelMaxPool,
)
from torchvision.ops.misc import FrozenBatchNorm2d
from model import backboneWithBiFPN
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import misc as misc_nn_ops
from dataset import CocoDataset


data_transforms = {
    "train": transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
}


def train_model(model, optimizer, scheduler, num_epochs=25):
    iter_start = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_loss = 10000.0
    best_valid_loss = 10000.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                model_loader = train_loader
                dataset_size = n_train
            if phase == "val":
                model.train()  # Set model to validation mode
                model_loader = valid_loader
                dataset_size = n_valid

            train_loss = 0.0
            valid_loss = 0.0

            # Iterate over data.
            if phase == "train":
                for iter, (images, targets) in enumerate(model_loader):
                    images = list(image.to(device) for image in images)
                    targets = [
                        {k: v.to(device) for k, v in t.items()}
                        for t in targets
                    ]
                    # zero the parameter gradients
                    #                     optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    outputs = model(images, targets)
                    loss = sum(loss for loss in outputs.values())
                    loss = loss / accumulation_steps

                    train_loss += loss * accumulation_steps

                    # backward + optimize only if in training phase
                    loss.backward()

                    if (iter + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    if (iter + 1) % 50 == 0:
                        iter_end = time.time() - iter_start
                        print(
                            "{} iter:{} Loss: {:.4f} Time: {:.0f}m {:.0f}s".format(
                                phase,
                                iter,
                                loss * accumulation_steps,
                                iter_end // 60,
                                iter_end % 60,
                            )
                        )

                        iter_start = time.time()

                scheduler.step()
                epoch_loss = train_loss * batch_size / dataset_size
                print(
                    "{} Loss: {:.4f} total loss :{:.4f}".format(
                        phase, epoch_loss, train_loss
                    )
                )

                if best_train_loss > train_loss:
                    best_train_loss = train_loss

            if phase == "val":
                with torch.no_grad():
                    for iter, (images, targets) in enumerate(model_loader):
                        images = list(image.to(device) for image in images)
                        targets = [
                            {k: v.to(device) for k, v in t.items()}
                            for t in targets
                        ]
                        outputs = model(images, targets)
                        loss = sum(loss for loss in outputs.values())
                        valid_loss += loss
                        if (iter + 1) % 50 == 0:
                            iter_end = time.time() - iter_start
                            print(
                                "{} iter:{} Loss: {:.4f} Time: {:.0f}m {:.0f}s".format(
                                    phase,
                                    iter,
                                    loss,
                                    iter_end // 60,
                                    iter_end % 60,
                                )
                            )
                            iter_start = time.time()
                    epoch_loss = valid_loss * 1 / dataset_size
                    print(
                        "{} Loss: {:.4f} total loss :{:.4f}".format(
                            phase, epoch_loss, valid_loss
                        )
                    )

                    if best_valid_loss > valid_loss:
                        best_valid_loss = valid_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(
                            best_model_wts, cfg.model_folder + cfg.model_name
                        )
                        print("save best training weight,complete!")

        time_elapsed = time.time() - since
        print(
            "Complete one epoch in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

    print(
        "Best train loss: {:4f} Best val loss: {:4f}".format(
            best_train_loss, best_valid_loss
        )
    )
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":

    coco = COCO(cfg.coco_train_file)
    id_list = list(coco.imgs.keys())
    n_id = len(id_list)
    n_train = int(0.9 * n_id)
    n_valid = n_id - n_train
    train_id = sample(id_list, n_train)
    for i in train_id:
        id_list.remove(i)
    valid_id = id_list

    bifpn_mode = "Disable"
    eval_train_mode = "Disable"
    if cfg.bifpn:
        bifpn_mode = "Enable"
    if cfg.eval_train:
        eval_train_mode = "Enable"

    print()
    print()
    print(
        "Numbers of train images: ",
        n_train,
        "\nNumbers of validation images: ",
        n_valid,
    )
    print(
        "This training model:" "\nBiFPN:",
        bifpn_mode,
        "\nEvaluate traning model:",
        eval_train_mode,
    )
    print()
    print()
    print()

    train_dataset = CocoDataset(cfg.train_path, train_id, Is_Train=True)
    valid_dataset = CocoDataset(cfg.train_path, valid_id, Is_Train=False)

    batch_size = cfg.batch_size
    accumulation_steps = cfg.accumulation_steps
    workers = cfg.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=train_dataset.collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        collate_fn=valid_dataset.collate_fn,
    )
    train_size = len(train_loader)
    valid_size = len(valid_loader)

    # cuda setting

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = cfg.model_name
    model_path = cfg.model_folder + model_name
    bifpn = cfg.bifpn

    backbone = backboneNet_efficient()
    backboneFPN = backboneWithFPN(backbone)
    if bifpn == True:
        backboneFPN = backboneWithBiFPN(backbone)

    anchor_sizes = (32, 64, 128, 256, 512)
    aspect_ratios = ((0.5, 1, 2),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    model_ft = MaskRCNN(
        backboneFPN,
        num_classes=cfg.num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=cfg.min_size,
        max_size=cfg.max_size,
    )
    model_ft.to(device)

    optimizer_ft = optim.SGD(
        model_ft.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_ft, T_0=cfg.epochs, T_mult=cfg.T_mult, eta_min=cfg.eta_min
    )

    model_ft = train_model(
        model_ft, optimizer_ft, lr_scheduler, num_epochs=cfg.epochs
    )

    # Evaluate training model
    # # from utils import binary_mask_to_rle
    if cfg.eval_train == True:

        coco_dt = []
        cocoGt = COCO(cfg.coco_train_file)
        model_ft.eval().to(device)

        with torch.no_grad():
            for imgid in cocoGt.imgs:

                image = Image.open(
                    cfg.train_path + cocoGt.loadImgs(ids=imgid)[0]["file_name"]
                )
                image = image.convert("RGB")
                image = data_transforms["val"](image)
                image = image.unsqueeze(0)
                image = image.to(device)

                outputs = model_ft(image)  # run inference of your model
                scores = outputs[0]["scores"]
                masks = outputs[0]["masks"]
                labels = outputs[0]["labels"]
                boxes = outputs[0]["boxes"]
                masks = np.round(masks.cpu())

                n_instances = len(scores)
                if (
                    len(labels) > 0
                ):  # If any objects are detected in this image
                    for i in range(n_instances):  # Loop all instances
                        # save information of the instance in a dictionary then append on coco_dt list
                        pred = {}
                        pred[
                            "image_id"
                        ] = imgid  # this imgid must be same as the key of test.json
                        pred["score"] = float(scores[i])
                        pred["category_id"] = int(labels[i])
                        mask = masks[i][0].unsqueeze(2)
                        pred["segmentation"] = binary_mask_to_rle(
                            (masks[i][0]).detach().cpu().numpy()
                        )  # save binary mask to RLE, e.g. 512x512 -> rle
                        coco_dt.append(pred)

        with open(cfg.result_pth + cfg.train_result, "w") as f:
            json.dump(coco_dt, f)

        cocoGt = COCO(cfg.coco_train_file)
        cocoDt = cocoGt.loadRes(cfg.result_pth + cfg.train_result)
        cocoEval = COCOeval(cocoGt, cocoDt, "segm")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
    print("Done!")
