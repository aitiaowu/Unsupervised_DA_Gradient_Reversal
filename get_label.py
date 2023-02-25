from pycocotools.coco import COCO
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from pycocotools.coco import COCO
from google.colab.patches import cv2_imshow
from torch.utils.data import DataLoader

class Transform(object):
    def __init__(self):
        self.label_map = COCO_LABEL_MAP

    def __call__(self, target, width, height):
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("No bbox found for object ", obj)

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class Data_Finder(data.Dataset):
    def __init__(self, image_path, info_file, transform=None,
                 target_transform=None, has_gt=True):
        self.root = image_path
        self.coco = COCO(info_file)
        self.ids = list(self.coco.imgToAnns.keys())  

        if len(self.ids) == 0 or not has_gt:  
            self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

        self.has_gt = has_gt

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        im, gt, Mask, h, w, num_crowds, name, Cat = self.pull_item(index)
        return im, (gt, Mask, num_crowds), name, Cat

    def pull_item(self, index):
        img_id = self.ids[index]
        if self.has_gt:
            ann_ids = self.coco.getAnnIds(imgIds=img_id) 
            target = self.coco.loadAnns(ann_ids)
        else:
            target = []
        crowd = [x for x in target if ('iscrowd' in x and x['iscrowd'])] ## iscrowd=0: single target.   iscrowd=1: a group of target(overlapping)
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        # This is so we ensure that all crowd annotations are at the end of the array
        target += crowd
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        path = osp.join(self.root, file_name)
        img = cv2.imread(path)
        height, width, _ = img.shape
          
        if len(target) > 0: 
            Cat = []
            Mask = self.coco.annToMask(target[0]).reshape(-1)
            for obj in target:
              mask = self.coco.annToMask(obj).reshape(-1)
              cat = obj['category_id']

              if cat not in Cat:
                Cat.append(cat)
                Mask = np.vstack((Mask, mask))
              else:
                Mask[-1] = Mask[-1] + mask
              
            Mask = Mask.reshape(-1, height, width)
            Mask = Mask[1:]
        if self.target_transform is not None and len(target) > 0:
            target = self.target_transform(target, width, height)
        return torch.from_numpy(img).permute(2, 0, 1), target, Mask, height, width, num_crowds, file_name, Cat

img_path_older = '/content/DATA/older_Data/images'
annotation_path = '/content/drive/MyDrive/Study/Domain_Adaptation_Vessel/older_vessel/simulated_corrosion_multi_class_no_morph_1.json'


if __name__=='__main__':
    dataset = Data_Finder(img_path_older, annotation_path)
    Info = DataLoader(dataset)
    count = 0
    for img, label, name, Cat in Info:
        img = np.uint8(img.squeeze().numpy().transpose(1, 2, 0))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Name = name[0].replace('image', 'label')
        gt, masks, num_crowds = label
        masks = masks.squeeze(0) ## .squeeze(A): Delete A_th dimension. Now we left batch * H * w, each batch represent a category labeled by 'pixel==1'
        label_img = np.zeros_like(gray)
        Cat_np = []
        for i in Cat:
          Cat_np.append(i.numpy().reshape(-1))
        for m in range(masks.size(0)):
          mask = masks[m].numpy()
          label_img[np.where(mask==1)] = Cat_np[m][0]
        cv2.imwrite(osp.join('/content/DATA/older_Data/labeled_data', Name), label_img)

import os

def get_imlist(path, txt):
  data = open(txt, 'w+')
  for f in os.listdir(path):
    if f.endswith('.png'):
      print(f, file = data)
  data.close()


source_list = get_imlist('/content/DATA/older_Data/labeled_data', '/content/sourcelist.txt')

olderlabel_img_path = '/content/DATA/older_Data/labeled_data'



"""young data"""


img_path_young = '/content/DATA/young_Data/images'
annotation_path = '/content/DATA/young_Data/images/simulated_corrosion_multi_class_1.json'

if __name__=='__main__':
  dataset = Data_Finder(img_path_young, annotation_path)
  Info = DataLoader(dataset)
  for img, label, name, Cat in Info:
        img = np.uint8(img.squeeze().numpy().transpose(1, 2, 0))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Name = name[0].replace('image', 'label')
        gt, masks, num_crowds = label
        masks = masks.squeeze(0) 
        label_img = np.zeros_like(gray)
        Cat_np = []
        for i in Cat:
          Cat_np.append(i.numpy().reshape(-1))
        for m in range(masks.size(0)):
          mask = masks[m].numpy()
          label_img[np.where(mask==1)] = Cat_np[m][0]
        cv2.imwrite(osp.join('/content/DATA/young_Data/labeled_data', Name), label_img)

#4 cat seam spot edge background

import os

def get_imlist(path, txt):
  data = open(txt, 'w+')
  for f in os.listdir(path):
    if f.endswith('.png'):
      print(f, file = data)
  data.close()


target_list = get_imlist('/content/DATA/young_Data/labeled_data', '/content/targetlist.txt')

younglabel_img_path = '/content/DATA/young_Data/labeled_data'

'''
if __name__=='__main__':
    dataset = Data_Finder(image, info_json)
    loader = DataLoader(dataset)
    for img, label, name, Cat in loader:
        img = np.uint8(img.squeeze().numpy().transpose(1, 2, 0))
        print(name[0], '\n')
        #for i in Cat:
        #  print((i.numpy())[0])
        gt, masks, num_crowds = label
        masks = masks.squeeze(0)
        color = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]])
        for m in range(masks.size(0)):
            mask = masks[m].numpy()
            y, x = np.where(mask == 1)
            img[y, x, :] = color[[Cat[m].numpy()][0] - 1]
        cv2_imshow(img)
        '''