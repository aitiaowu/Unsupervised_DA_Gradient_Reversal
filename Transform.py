#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.transforms import functional as F
import cv2
import random
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as coco_mask

class ConvertCocoPolysToMask(object):
    def convert_coco_poly_to_mask(segmentations, height, width):
        masks = []
        for polygons in segmentations:
          rles = coco_mask.frPyObjects(polygons, height, width)
          mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
          mask = mask[..., None]
          mask = torch.as_tensor(mask, dtype=torch.uint8)
          mask = mask.any(dim=2)
          masks.append(mask)
        if masks:
          masks = torch.stack(masks, dim=0)
        else:
          masks = torch.zeros((0, height, width), dtype=torch.uint8)
        return masks
  
    def __call__(self, image, target):
        w, h = image.size
 
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
 
        anno = target["annotations"]
 
        anno = [obj for obj in anno if obj['iscrowd'] == 0]
 
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
 
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
 
        segmentations = [obj["segmentation"] for obj in anno]
        masks = self.convert_coco_poly_to_mask(segmentations, h, w)
 
        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)
 
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
 
        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        return image, target






class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
 
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


    
    
    
class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob
 
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


