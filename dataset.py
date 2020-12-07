import torch
import torch.nn as nn
import torchvision
import csv
import timm
import time
import glob
import copy
import os
import json
import cv2
import numpy as np
import pickle
import config as cfg
import torch.optim as optim
from torch.optim import lr_scheduler
from timm.models import *
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, models, datasets
from PIL import Image
from PIL import Image, ImageEnhance, ImageOps
from data_augment.autoaugment import ImageNetPolicy
from pycocotools.coco import COCO
from pycocotools.mask import encode
from pycocotools.cocoeval import COCOeval
from data_transform import transform, draw_rect

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


class CocoDataset(Dataset):
    def __init__(self, data_folder, data_id, Is_Train):
        self.data_folder = data_folder
        self.coco_data = COCO("./data/" + "pascal_train.json")
        self.num_img = len((data_id))
        self.image_name = []
        self.Is_Train = Is_Train
        self.boxes = {}
        self.labels = {}
        self.masks = {}

        coco = self.coco_data

        for num in data_id:
            imgIds = num
            img_info = coco.loadImgs(ids=imgIds)[0]
            img_name = img_info["file_name"]

            # read box label mask
            annids = coco.getAnnIds(imgIds=imgIds)
            anns = coco.loadAnns(annids)

            height = coco.annToMask(anns[0]).shape[0]
            width = coco.annToMask(anns[0]).shape[1]

            boxes = np.zeros((len(anns), 4))
            labels = np.zeros(len(anns))
            masks = np.zeros((len(anns), height, width))

            for cnt in range(len(anns)):
                bbox = anns[cnt]["bbox"]
                boxes[cnt] = [
                    bbox[0],
                    bbox[1],
                    bbox[0] + bbox[2],
                    bbox[1] + bbox[3],
                ]
                labels[cnt] = anns[cnt]["category_id"]
                masks[cnt] = coco.annToMask(anns[cnt])

            self.image_name.append(img_name)
            self.boxes[img_name] = boxes
            self.labels[img_name] = labels
            self.masks[img_name] = masks

    def __getitem__(self, i):
        # Read image
        _img_name = self.image_name
        target_imgae = _img_name[i]
        #         print(target_imgae)

        #         image = Image.open(self.data_folder + target_imgae)
        #         image = image.convert('RGB')

        image = Image.open(self.data_folder + target_imgae).convert("RGB")
        boxes = self.boxes[target_imgae]
        labels = self.labels[target_imgae]
        masks = self.masks[target_imgae]

        # Apply transformations
        if self.Is_Train == True:
            image_transform = ImageNetPolicy()
            image = image_transform(image)
            t_image, t_boxes, t_masks, t_labels = transform(
                image, boxes, masks, labels
            )

            while (
                len(t_boxes) == 0
                or not (t_boxes[:, 0] <= t_boxes[:, 2]).all()
                or not (t_boxes[:, 1] <= t_boxes[:, 3]).all()
            ):
                p = random.random()
                if p > 0.3:
                    t_image, t_boxes, t_masks, t_labels = transform(
                        image, boxes, masks, labels
                    )
                else:
                    t_image, t_boxes, t_masks, t_labels = (
                        image,
                        boxes,
                        masks,
                        labels,
                    )

            image = data_transforms["train"](t_image)
            # Read objects in this image (bounding boxes, labels)
            # (n_objects), (n_objects, 4)
            t_boxes = torch.FloatTensor(t_boxes)  # (n_objects, 4)
            t_labels = torch.LongTensor(t_labels)  # (n_objects)
            t_masks = torch.ByteTensor(t_masks)  # (n_objects, H, W)
        else:
            #             image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            image = data_transforms["val"](image)
            t_boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
            t_labels = torch.LongTensor(labels)  # (n_objects)
            t_masks = torch.ByteTensor(masks)  # (n_objects, H, W)

        target = {}
        target["boxes"] = t_boxes
        target["labels"] = t_labels
        target["masks"] = t_masks
        return image, target

    def __len__(self):
        return self.num_img

    def collate_fn(self, batch):
        images = list()
        targets = list()

        for b in batch:
            images.append(b[0])
            targets.append(b[1])

        return images, targets
