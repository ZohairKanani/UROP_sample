import os
# import pandas as pd
from PIL import Image

import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import densenet161
from sklearn.model_selection import train_test_split
# from torch._C import namedtuple_LU_pivots_info
from torchvision import transforms

class DMIDataset(torch.utils.data.Dataset):
    def __init__(self, transforms, dataset_root, filelists=None, num_classes=4, mode='train_1'): 

        # mode = 1: measurable
        # mode = 2: unmeasurable
        # mode = 3: whole dataset
        # mode = 4: classify measurability

        self.dataset_root = dataset_root        
        self.transforms = transforms
        self.mode = mode
        self.num_classes = num_classes
        self.file_list = []

        for dir in os.listdir(dataset_root):
            if not os.path.isdir(dir):
                files = os.listdir(dataset_root + "/" + dir)
                for file in files:
                    if not os.path.isdir(file):
                        self.file_list.append([file, dir])

        if filelists is not None:
            if '1' in mode:
                self.file_list = [item for item in self.file_list if item[0] in filelists and item[1][0] in ['1', '2']]
            if '2' in mode:
                self.file_list = [item for item in self.file_list if item[0] in filelists and item[1][0] in ['3', '4']]
            if '3' in mode:
                self.file_list = [item for item in self.file_list if item[0] in filelists]
            if '4' in mode:
                self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        img_id, dir = self.file_list[idx]

        scp_img_path = os.path.join(self.dataset_root, dir, img_id)
        scp_img = Image.open(scp_img_path).convert('L')

        if self.transforms is not None:
            scp_img = self.transforms(scp_img)

        if '4' in self.mode:
            if dir[0] in ['1', '2']:
                return scp_img, 1 # measurable
            else:
                return scp_img, 0 # unmeasurable
        else:
            if dir[0] in ['1', '3']:
                return scp_img, 1 # DMI
            else:
                return scp_img, 0 # NoDMI

    def __len__(self):
        return len(self.file_list)

    

    



