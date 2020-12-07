import torch
import torch.nn as nn
import torch.optim as optim
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


if __name__ == "__main__":

    coco_dt = []
    cocoGt = COCO("./data/" + "test.json")

    # model_path = "./models/"+'mask_rcnn_effb7_frozen_bifpn_60_v8_60'
    # model_ft = get_model_instance_segmentation(num_classes=21)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone = backboneNet_efficient()
    backboneFPN = backboneWithFPN(backbone)

    if cfg.bifpn == True:
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
    model_path = cfg.model_folder + cfg.model_name
    model_ft.load_state_dict(torch.load(model_path))

    with torch.cuda.device(0):
        model_ft.eval().to(device)
        with torch.no_grad():
            for iter, imgid in enumerate(cocoGt.imgs):

                image = Image.open(
                    cfg.test_path + cocoGt.loadImgs(ids=imgid)[0]["file_name"]
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
                        pred["segmentation"] = binary_mask_to_rle(
                            (masks[i][0]).detach().cpu().numpy()
                        )  # save binary mask to RLE, e.g. 512x512 -> rle
                        coco_dt.append(pred)

    with open(cfg.result_pth + cfg.json_name, "w") as f:
        json.dump(coco_dt, f)
    print("Done!")
