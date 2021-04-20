import os
import torch
from PIL import Image
from torchvision import transforms as T
import numpy as np
from torch.utils import data
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class ImageNet(data.Dataset):
    def __init__(self, dir, csv_path, transforms = None):
        self.dir = dir   
        self.csv = pd.read_csv(csv_path, engine='python')
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId']
        Truelabel = img_obj['TrueLabel']
        # TargetClass = img_obj['TargetClass'] - 1
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            data = pil_img
        return data, ImageID, Truelabel

    def __len__(self):
        return len(self.csv)


class ImageNet_Mask(data.Dataset):
    def __init__(self, dir, mask_dir, csv_path, transforms = None):
        self.dir = dir
        self.mask_dir = mask_dir
        self.csv = pd.read_csv(csv_path, engine='python')
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId']
        Truelabel = img_obj['TrueLabel']
        # TargetClass = img_obj['TargetClass'] - 1
        img_path = os.path.join(self.dir, ImageID)
        mask_path = os.path.join(self.mask_dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
            mask_data = self.transforms(mask_img)
        else:
            data = pil_img
            mask_data = mask_img
        return data, mask_data, ImageID, Truelabel

    def __len__(self):
        return len(self.csv)


transforms = T.Compose([T.ToTensor()])








