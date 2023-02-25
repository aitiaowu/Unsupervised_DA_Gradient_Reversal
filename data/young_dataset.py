import os
import os.path as osp
from PIL import Image
import numpy as np
import torch
from torch.utils import data

class youngDataSet(data.Dataset):

    def __init__(self, root, list_path, crop_size=(11, 11), resize=(11, 11), ignore_label=255, mean=(128, 128, 128), max_iters=None):
        self.root = root  # folder for which contains subfolder images, labels
        self.list_path = list_path   # list of image names
        self.crop_size = crop_size   # dst size for resize
        self.resize = resize
        self.ignore_label = ignore_label
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(  np.ceil(float(max_iters)/len(self.img_ids))  )

        self.files = []

        #self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3}
        self.id_to_trainid = {1: 0, 2: 1, 3: 2}
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        Imgname = name.replace("label", "image")
        image = Image.open(osp.join(self.root, "images/%s" % Imgname)).convert('RGB')
        label = Image.open(osp.join(self.root, "labeled_data/%s" % name))
        # resize
        image = image.resize(self.resize, Image.BICUBIC)
        label = label.resize(self.resize, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        #label_copy = label_copy[:,:,0]
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        

        return image.copy(), label_copy.copy(), np.array(size), name

