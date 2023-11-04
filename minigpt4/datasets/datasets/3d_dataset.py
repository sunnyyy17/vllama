import torch
import torch.nn as nn
#import torchvision.transforms as transforms
#import cv2
#import math
#import json
#from PIL import Image
#import os.path as op
import numpy as np
#from numpy.random import randint
import random
#import code, time
import glob
#import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
#import pickle
#import json
import kornia
import re
import warnings
import h5py
from torch.utils.data import DataLoader
from torch.utils import data


class CTSegDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, img_path, txt_path, column='impression', size=None, transform=None, is_train=True):
        super().__init__()
        if size != None: 
            print(h5py.File(img_path, 'r').keys())
            self.img_dset = h5py.File(img_path, 'r')['ct'][:size]
            self.txt_dset = pd.read_csv(txt_path)[column][:size]
        else:
            print(h5py.File(img_path, 'r').keys())
            self.img_dset = h5py.File(img_path, 'r')['ct']
            self.txt_dset = pd.read_csv(txt_path)[column]
        self.transform = transform
        self.is_train = is_train
    
    def __len__(self):
        return len(self.txt_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = self.img_dset[idx] # np array, (320, 320)
        #img = np.expand_dims(img, axis=0)
        #img = np.repeat(img, 3, axis=0)
        txt = self.txt_dset[idx] # python str
        if type(txt) == type(float("nan")): # capture the case of empty "Impression" sections
            print("Nan Impression")
            txt = " "
        
        #img = torch.from_numpy(img) # torch, (3, 320, 320)
        #img = kornia.geometry.transform.resize(img, size=(224, 224))
        #if self.transform:
            #img = self.transform(img)
        #sample = {'img': img, 'txt': txt }
        
        return img, txt

img_path = '/scratch/slurm-user3/changsun/data/ct-for-detect-seg/ct.h5'
txt_path = '/scratch/slurm-user3/changsun/data/ct-for-detect-seg/ct_report.csv'

ctseg_dataset = CTSegDataset(img_path, txt_path)
dataloader = DataLoader(ctseg_dataset, batch_size=1, shuffle=False)
ctseg_index_list = [0, 33, 59, 91, 121, 154, 182, 214, 247, 282, 316, 343, 372, 403, 426, 462, 488, 518, 547, 575, 601, 635, 669, 698, 728, 761, 794, 827, 856, 887, 918, 944, 973, 1002, 1031, 1063, 1099, 1118, 1147, 1179, 1210, 1245, 1273, 1306, 1337, 1369, 1403, 1432, 1465, 1496, 1525, 1554, 1584, 1615, 1646, 1681, 1710, 1740, 1769, 1799, 1828, 1856, 1884, 1917, 1943, 1971, 2000, 2032, 2063, 2092, 2122, 2153, 2183, 2213, 2241, 2280, 2310, 2350, 2380, 2412, 2440, 2470]

for idx, item in enumerate(dataloader):
    
    if idx == 0:
        single_3d = item[0]
        continue

    elif idx not in ctseg_index_list and idx < ctseg_index_list[-1]:
        single_3d = np.concatenate((single_3d, item[0]), axis=0)

    elif idx in ctseg_index_list and idx < ctseg_index_list[-1]: 
        single_3d_filename = '/scratch/slurm-user3/changsun/data/ct-for-detect-seg/ct_volume_brain/volume_' + str(ctseg_index_list.index(idx)+49-1) + '.npy'
        np.save(single_3d_filename, single_3d) 
        single_3d = item[0]

    elif idx in ctseg_index_list and idx == ctseg_index_list[-1]:
        single_3d_filename = '/scratch/slurm-user3/changsun/data/ct-for-detect-seg/ct_volume_brain/volume_' + str(ctseg_index_list.index(idx)+49-1) + '.npy'
        np.save(single_3d_filename, single_3d) 
        single_3d = item[0]
    else: 
        single_3d = np.concatenate((single_3d, item[0]), axis=0)
#print(idx)
single_3d_filename = '/scratch/slurm-user3/changsun/data/ct-for-detect-seg/ct_volume_brain/volume_' + str(len(ctseg_index_list)+49-1) + '.npy'
np.save(single_3d_filename, single_3d) 
