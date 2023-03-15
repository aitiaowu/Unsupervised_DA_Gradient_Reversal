import torch
import numpy as np
import os
import os.path as osp
from PIL import Image
from torch.utils import data

class oldDataSetLabel(data.Dataset):
    
    def __init__(self, root, list_path, crop_size=(11, 11), mean=(128, 128, 128), max_iters=None, set='vel'):
        self.root = root    
        self.list_path = list_path # list of image names
        self.crop_size = crop_size
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int( np.ceil(float(max_iters)/len(self.img_ids)) )

        self.files = []
        self.ignore_label = 255
        self.set = set

        self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3}
        #self.id_to_trainid = {1: 0, 2: 1, 3: 2}
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        img_name = name.replace("label", "image")

        image = Image.open(osp.join( self.root, "images/%s" % img_name   )).convert('RGB')
       
        label = Image.open(osp.join( self.root, "labeled_data/%s" % name   ))
        
        # resize
        image = image.resize( self.crop_size, Image.BICUBIC )
        label = label.resize( self.crop_size, Image.NEAREST )
        
        image = np.asarray( image, np.float32 )
        label = np.asarray( label, np.float32 )

        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        '''
        for m in range(-1,4):
          cnt=0
          for i in range( label_copy.shape[0]):
              for j in range ( label_copy.shape[1]):
                  if label_copy[i][j] ==m:
                      cnt += 1
          print(m,cnt)'''

        return image.copy(), label_copy.copy(), np.array(size), name

