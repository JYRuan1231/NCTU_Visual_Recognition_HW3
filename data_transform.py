import math
import numpy as np
import random
from numpy import zeros, newaxis
import torch
import torchvision.transforms.functional as F
from data_augment.bbox_util import *
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def draw_rect(im, cords, color=None):

    im = im.copy()

    cords = cords[:, :4]
    cords = cords.reshape(-1, 4)
    if not color:
        color = [255, 255, 255]
    for cord in cords:

        pt1, pt2 = (cord[0], cord[1]), (cord[2], cord[3])

        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])

        im = cv2.rectangle(
            im.copy(), pt1, pt2, color, int(max(im.shape[:2]) / 200)
        )
    return im


def rotate_angle(image, masks, angle):
    # grab the dimensions of the image and then determine the
    # centre

    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH), borderValue=(128, 128, 128))

    new_shape = image.shape

    rotate_masks = np.zeros(
        (masks.shape[0], new_shape[0], new_shape[1])
    )  # np.zeros((len(anns), height,width))

    for i in range(masks.shape[0]):
        rotate_masks[i] = cv2.warpAffine(masks[i], M, (nW, nH))

    #    image = cv2.resize(image, (w,h))
    return image, rotate_masks


def get_corners(boxes):
    width = (boxes[:, 2] - boxes[:, 0]).reshape(-1, 1)
    height = (boxes[:, 3] - boxes[:, 1]).reshape(-1, 1)

    x1 = boxes[:, 0].reshape(-1, 1)
    y1 = boxes[:, 1].reshape(-1, 1)
    x2 = x1 + width
    y2 = y1
    x3 = x1
    y3 = y1 + height
    x4 = boxes[:, 2].reshape(-1, 1)
    y4 = boxes[:, 3].reshape(-1, 1)
    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def change_origin2center(boxes, width, height):
    boxes[:, 0] = boxes[:, 0] - width // 2
    boxes[:, 2] = boxes[:, 2] - width // 2
    boxes[:, 1] = boxes[:, 1] - height // 2
    boxes[:, 3] = boxes[:, 3] - height // 2

    return boxes


def boxes_rotate(boxes, angle, ori_w, ori_h, new_w, new_h):
    # change coord. base to image center
    boxes = change_origin2center(boxes, ori_w, ori_h)

    # get 4 corners of boxes
    corners = get_corners(boxes)

    cos = math.cos(angle / 180 * math.pi)
    sin = math.sin(angle / 180 * math.pi)
    Rotation_matrix = [[cos, -sin], [sin, cos]]

    # rotate the corner coord.
    new_corners = corners
    new_corners[:, :2] = np.matmul(new_corners[:, :2], Rotation_matrix)
    new_corners[:, 2:4] = np.matmul(new_corners[:, 2:4], Rotation_matrix)
    new_corners[:, 4:6] = np.matmul(new_corners[:, 4:6], Rotation_matrix)
    new_corners[:, 6:] = np.matmul(new_corners[:, 6:], Rotation_matrix)
    new_corners = np.round(new_corners)

    # find correct boxes in new expanded-rotation img
    new_left1 = np.minimum(new_corners[:, 0], new_corners[:, 4]).reshape(
        -1, 1
    )  # min(x1, x3)
    new_top1 = np.minimum(new_corners[:, 1], new_corners[:, 3]).reshape(
        -1, 1
    )  # min(y1, y2)
    new_right1 = np.maximum(new_corners[:, 2], new_corners[:, 6]).reshape(
        -1, 1
    )  # max(x2, x4)
    new_bottom1 = np.maximum(new_corners[:, 5], new_corners[:, 7]).reshape(
        -1, 1
    )  # min(y3, y4)
    new_left2 = ((new_corners[:, 0] + new_corners[:, 4]) // 2).reshape(
        -1, 1
    )  # (x1 + x3) / 2
    new_top2 = ((new_corners[:, 1] + new_corners[:, 3]) // 2).reshape(
        -1, 1
    )  # (y1 + y2) / 2
    new_right2 = ((new_corners[:, 2] + new_corners[:, 6]) // 2).reshape(
        -1, 1
    )  # (x2 + x4) / 2
    new_bottom2 = ((new_corners[:, 5] + new_corners[:, 7]) // 2).reshape(
        -1, 1
    )  # (y3 + y4) / 2

    new_left = (new_left1 + new_left2) // 2
    new_top = (new_top1 + new_top2) // 2
    new_right = (new_right1 + new_right2) // 2
    new_bottom = (new_bottom1 + new_bottom2) // 2

    # change coord. base to new image left-top
    new_left = new_left + new_w // 2
    new_right = new_right + new_w // 2
    new_top = new_top + new_h // 2
    new_bottom = new_bottom + new_h // 2

    new_boxes = np.hstack((new_left, new_top, new_right, new_bottom))

    return new_boxes


class RandomRotate(object):
    def __init__(self, angle=10):
        self.angle = angle

        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"

        else:
            self.angle = (-self.angle, self.angle)

    def __call__(self, img, bboxes, masks, labels):

        angle = random.uniform(*self.angle)

        w, h = img.shape[1], img.shape[0]
        cx, cy = w // 2, h // 2

        rotate_img, rotate_masks = rotate_angle(img, masks, angle)

        rotate_w, rotate_h = img.shape[1], img.shape[0]

        corners = get_corners(bboxes)
        corners = np.hstack((corners, bboxes[:, 4:]))
        corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)
        new_bbox = get_enclosing_box(
            corners, rotate_w, rotate_h
        )  # adjust rotated boxes

        img = rotate_img
        masks = rotate_masks
        bboxes = new_bbox

        #         print(bboxes)

        #         scale_factor_x = img.shape[1] / w
        #         scale_factor_y = img.shape[0] / h
        #         img = cv2.resize(rotate_img, (w,h))
        #         for i in range(masks.shape[0]):
        #             masks[i] = cv2.resize(rotate_masks[i], (w,h))
        #         new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
        #         bboxes  = new_bbox
        #         bboxes, masks, labels = clip_box(bboxes, [0,0,rotate_w, rotate_h], 0.25, masks, labels)

        return img, bboxes, masks, labels


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes, masks, labels):
        # Flip image
        img = F.hflip(img)

        # Flip mask (HxW)
        new_masks = []
        for i in range(len(masks)):
            flipped_img = cv2.flip(masks[i], 1)
            new_masks.append(flipped_img)
        new_masks = np.array(new_masks, dtype=np.uint8)

        # Flip boxes
        new_boxes = bboxes
        new_boxes[:, 0] = img.width - bboxes[:, 0]
        new_boxes[:, 2] = img.width - bboxes[:, 2]
        new_boxes = new_boxes[:, [2, 1, 0, 3]]

        bboxes = new_boxes
        masks = new_masks

        return img, bboxes, masks, labels


class RandomScale(object):
    def __init__(self, scale=0.2, diff=False):
        self.scale = scale

        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)

        self.diff = diff

    def __call__(self, img, bboxes, masks, labels):

        # Chose a random digit to scale by

        img_shape = img.shape

        if len(bboxes) == 0:
            return img, bboxes, masks, labels

        if self.diff:
            scale_x = random.uniform(*self.scale)
            scale_y = random.uniform(*self.scale)
        else:
            scale_x = random.uniform(*self.scale)
            scale_y = scale_x

        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y

        img = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)

        new_shape = img.shape

        scale_masks = np.zeros((masks.shape[0], new_shape[0], new_shape[1]))

        for i in range(masks.shape[0]):
            scale_masks[i] = cv2.resize(
                masks[i], None, fx=resize_scale_x, fy=resize_scale_y
            )

        bboxes[:, :4] *= [
            resize_scale_x,
            resize_scale_y,
            resize_scale_x,
            resize_scale_y,
        ]

        canvas = np.zeros(img_shape, dtype=np.uint8)
        y_lim = int(min(resize_scale_y, 1) * img_shape[0])
        x_lim = int(min(resize_scale_x, 1) * img_shape[1])
        canvas[:y_lim, :x_lim, :] = img[:y_lim, :x_lim, :]

        mask_canvas = np.zeros(masks[0].shape, dtype=np.uint8)
        for i in range(masks.shape[0]):
            mask_canvas[:y_lim, :x_lim] = scale_masks[i][:y_lim, :x_lim]
            masks[i] = mask_canvas

        img = canvas

        bboxes, masks, labels = clip_box(
            bboxes, [0, 0, img_shape[1], img_shape[0]], 0.1, masks, labels
        )

        return img, bboxes, masks, labels


class RandomTranslate(object):
    def __init__(self, translate=0.2, diff=False):
        self.translate = translate

        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"
        #             assert self.translate[0] > 0 & self.translate[0] < 1
        #             assert self.translate[1] > 0 & self.translate[1] < 1

        else:
            assert self.translate > 0 and self.translate < 1
            self.translate = (-self.translate, self.translate)

        self.diff = diff

    def __call__(self, img, bboxes, masks, labels):
        # Chose a random digit to scale by
        img_shape = img.shape

        if len(bboxes) == 0:
            return img, bboxes, masks, labels

        # translate the image

        # percentage of the dimension of the image to translate
        translate_factor_x = random.uniform(*self.translate)
        translate_factor_y = random.uniform(*self.translate)

        if not self.diff:
            translate_factor_y = translate_factor_x

        canvas = np.ones(img_shape) - 2 * np.ones(img_shape)
        canvas_mask = np.zeros((img_shape[0], img_shape[1])).astype(np.uint8)

        corner_x = int(translate_factor_x * img.shape[1])
        corner_y = int(translate_factor_y * img.shape[0])

        # change the origin to the top-left corner of the translated box
        orig_box_cords = [
            max(0, corner_y),
            max(corner_x, 0),
            min(img_shape[0], corner_y + img.shape[0]),
            min(img_shape[1], corner_x + img.shape[1]),
        ]

        mask = img[
            max(-corner_y, 0) : min(img.shape[0], -corner_y + img_shape[0]),
            max(-corner_x, 0) : min(img.shape[1], -corner_x + img_shape[1]),
            :,
        ]
        canvas[
            orig_box_cords[0] : orig_box_cords[2],
            orig_box_cords[1] : orig_box_cords[3],
            :,
        ] = mask

        canvas[np.where(canvas == -1)] = 256 * 0.485

        img = canvas.astype(np.uint8)

        for i in range(masks.shape[0]):
            mask = masks[i][
                max(-corner_y, 0) : min(
                    img.shape[0], -corner_y + img_shape[0]
                ),
                max(-corner_x, 0) : min(
                    img.shape[1], -corner_x + img_shape[1]
                ),
            ]
            canvas_mask[
                orig_box_cords[0] : orig_box_cords[2],
                orig_box_cords[1] : orig_box_cords[3],
            ] = mask

            masks[i] = canvas_mask

        bboxes[:, :4] += [corner_x, corner_y, corner_x, corner_y]

        bboxes, masks, labels = clip_box(
            bboxes, [0, 0, img_shape[1], img_shape[0]], 0.1, masks, labels
        )

        return img, bboxes, masks, labels


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(
        set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0)
    )  # (n1, n2, 2)
    upper_bounds = torch.min(
        set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0)
    )  # (n1, n2, 2)
    intersection_dims = torch.clamp(
        upper_bounds - lower_bounds, min=0
    )  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (
        set_1[:, 3] - set_1[:, 1]
    )  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (
        set_2[:, 3] - set_2[:, 1]
    )  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = (
        areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection
    )  # (n1, n2)

    return intersection / union  # (n1, n2)


def RandomCrop(image, boxes, masks, labels):
    original_w, original_h = image.size
    image = F.to_tensor(image)
    #         merged_mask = target['merged_mask']

    # Keep choosing a minimum overlap until a successful crop is made

    # Try up to 50 times for this choice of minimum overlap
    max_trials = 50
    while True:
        min_overlap = random.choice([0.2])
        max_trials = 50
        for _ in range(max_trials):
            min_scale = 0.5
            scale_h = random.uniform(min_scale, 0.8)
            scale_w = random.uniform(min_scale, 0.8)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            find_box = torch.FloatTensor(boxes)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(
                crop.unsqueeze(0), find_box
            )  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Crop masks
            new_masks = masks[:, top:bottom, left:right]

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (
                (bb_centers[:, 0] > left)
                * (bb_centers[:, 0] < right)
                * (bb_centers[:, 1] > top)
                * (bb_centers[:, 1] < bottom)
            )  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue
            crop = crop.numpy()
            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_masks = new_masks[centers_in_crop, :, :]
            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = np.maximum(
                new_boxes[:, :2], crop[:2]
            )  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = np.minimum(
                new_boxes[:, 2:], crop[2:]
            )  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            boxes = new_boxes
            masks = new_masks
            labels = new_labels
            image = F.to_pil_image(new_image)
            return image, boxes, masks, labels
        image = F.to_pil_image(image)
        return image, boxes, masks, labels


def transform(image, boxes, masks, labels):

    p1 = random.random()
    p2 = random.random()
    p3 = random.random()
    p4 = random.random()
    p5 = random.random()
    p6 = random.random()

    # image, boxes, masks, labels = RandomTranslate((0.5), diff = True)(image, boxes, masks, labels)
    # image, boxes, masks, labels = RandomScale((0.3,0.9), diff = True)(image, boxes, masks, labels)

    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if p1 > 0.3:
        image, boxes, masks, labels = RandomRotate(10)(
            image, boxes, masks, labels
        )
    #     if p1 > 0.5:
    #         image, boxes, masks, labels = RandomTranslate((0.3), diff = True)(image, boxes, masks, labels)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if p2 > 0.5:
        image, boxes, masks, labels = RandomHorizontalFlip(1)(
            image, boxes, masks, labels
        )

    if p3 > 0.3:
        image, boxes, masks, labels = RandomCrop(image, boxes, masks, labels)

    return image, boxes, masks, labels
