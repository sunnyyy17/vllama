#import pydicom
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

from torch.utils import data

def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('/', ' ').replace('<person>', 'person')
    caption = caption.replace('[sep]', '[SEP]')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    # if len(caption_words)>max_words:
    caption = ' '.join(caption_words[:max_words])
    return caption


class CTDataset(object):
    def __init__(self, csv_dir, data_dir, z_length, image_res, is_train=True, is_val=False, is_large=False):

        self.csv = pd.read_csv(csv_dir)
        self.data_dir = data_dir
        self.z_length = z_length
        self.image_res = image_res
        self.all_subject = sorted(glob.glob(self.data_dir + '/*'))

        # split train/val/test set (not depending on the seed) -> fixed division
        if is_train:
            self.subject_list = self.all_subject[:int(len(self.all_subject)*0.90)]  # 90%
        else:
            if is_val:
                self.subject_list = self.all_subject[int(len(self.all_subject)*0.90):int(len(self.all_subject)*0.94)]   # 4%
            else:
                self.subject_list = self.all_subject[int(len(self.all_subject)*0.94):]  # 6%
        self.subject_list = sorted(self.subject_list)
        
        if is_train:
            self.weights = []
            for subject in self.subject_list:
                ID = subject.split('/')[-1]
                d_frame = self.csv.loc[self.csv.path == 'rsna/train/{}'.format(ID)]

                EDH = int(d_frame['epidural'].values)
                IPH = int(d_frame['intraparenchymal'].values)
                IVH = int(d_frame['intraventricular'].values)
                SAH = int(d_frame['subarachnoid'].values)
                SDH = int(d_frame['subdural'].values)
                ANY = int(d_frame['any'].values)

                all = EDH + IPH + IVH + SAH + SDH + ANY

                if all == 0:
                    weight = 0.1
                else:
                    weight = 0.9
                self.weights.append(weight)
            self.weights = np.array(self.weights)
            self.weights = self.weights / self.weights.sum()

        self.is_train = is_train
        self.is_large = is_large

        print(len(self.subject_list))

    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, idx):

        subject = self.subject_list[idx]
        ID = subject.split('/')[-1]

        # Load all images
        volume = np.load(subject + '/3d.npy')

        d_frame = self.csv.loc[self.csv.path == 'rsna/train/{}'.format(ID)]

        EDH = int(d_frame['epidural'].values)
        IPH = int(d_frame['intraparenchymal'].values)
        IVH = int(d_frame['intraventricular'].values)
        SAH = int(d_frame['subarachnoid'].values)
        SDH = int(d_frame['subdural'].values)
        ANY = int(d_frame['any'].values)
        
        caps = []
        if ANY:
            caps.append('Intracranial hemorrhage is observed in this CT study.')
        if SDH:
            caps.append('Subdural hematoma is observed.')
        if SAH:
            caps.append('Subarachnoid hemorrhage is observed.')
        if IVH:
            caps.append('Intraventricular hemorrhage is observed.')
        if IPH:
            caps.append('Intraparenchymal hemorrhage is observed.')
        if EDH:
            caps.append('Epidural hematoma is observed.')

        if len(caps) == 0:
            caps = ['No evidence of intracranial hemorrhage is observed in this CT study.']

        new_caps = []
        for i, cap in enumerate(caps):
            new_caps.append('- ' + cap)

        caption = ' '.join(new_caps)

        # caption = self.df[subject]
        caption = pre_caption(caption, max_words=200)
        #print('MRI Volume: ', volume.shape)
        length = volume.shape[2]
        indexes = list(np.arange(length))

        if self.is_train:
            sampled_ind = sorted(random.sample(indexes, self.z_length))
        else:
            sampled_ind = indexes
            # try:
            #     sampled_ind = sorted(random.sample(indexes, 32))
            # except:
            #     sampled_ind = sorted(random.choices(indexes, k=32))


        # Center crop and translation
        if self.is_train:
            d = random.randint(0, 10)
            h_t = random.randint(-d, d)
            w_t = random.randint(-d, d)
        else:
            d = 5
            h_t = 0
            w_t = 0

        preproc_frames = volume[h_t + d:h_t + 224-d, w_t + d:w_t + 224-d, sampled_ind]
        #print('volume shape', preproc_frames.shape)
        #print('volume min, max', preproc_frames.min(), preproc_frames.max())

        preproc_frames = torch.from_numpy(preproc_frames)
        preproc_frames = preproc_frames.permute(2, 0, 1)

        # Brain window
        preproc_frames_brain = preproc_frames.clone()
        preproc_frames_brain[preproc_frames_brain > 80] = 80
        preproc_frames_brain[preproc_frames_brain < 0] = 0
        preproc_frames_brain = (preproc_frames_brain - preproc_frames_brain.min()) / (preproc_frames_brain.max() - preproc_frames_brain.min())
        preproc_frames_brain = preproc_frames_brain.unsqueeze(1)

        # Subdural window
        preproc_frames_subdural = preproc_frames.clone()
        preproc_frames_subdural[preproc_frames_subdural > 170] = 170
        preproc_frames_subdural[preproc_frames_subdural < -10] = -10
        preproc_frames_subdural = (preproc_frames_subdural - preproc_frames_subdural.min()) / (preproc_frames_subdural.max() - preproc_frames_subdural.min())
        preproc_frames_subdural = preproc_frames_subdural.unsqueeze(1)

        # Bone window
        preproc_frames_bone = preproc_frames.clone()
        preproc_frames_bone[preproc_frames_bone > 1500] = 1500
        preproc_frames_bone[preproc_frames_bone < -500] = -500
        preproc_frames_bone = (preproc_frames_bone - preproc_frames_bone.min()) / (preproc_frames_bone.max() - preproc_frames_bone.min())
        preproc_frames_bone = preproc_frames_bone.unsqueeze(1)

        preproc_frames_cat = torch.cat([preproc_frames_brain, preproc_frames_brain, preproc_frames_brain], dim=1)
        # preproc_frames_cat = torch.cat([preproc_frames_brain, preproc_frames_subdural, preproc_frames_bone], dim=1)
        preproc_frames_cat = kornia.geometry.transform.resize(preproc_frames_cat,
                                                              size=(224, 224))

        if not self.is_large:
            preproc_frames_cat = kornia.geometry.transform.resize(preproc_frames_cat, size=(self.image_res, self.image_res))

        return preproc_frames_cat, caption

def pad_collate(data):
    img_caption_ID = list(zip(*data))
    batch_img = img_caption_ID[0]
    batch_caption = img_caption_ID[1]
    batch_ID = img_caption_ID[2]
    batch_img_list = list(batch_img)
    batch_caption_list = list(batch_caption)
    batch_ID_list = list(batch_ID)
    for i, img in  enumerate(batch_img_list):
        batch_img_list[i] = img.squeeze(dim=0)
    batch_img_pad = nn.utils.rnn.pad_sequence(batch_img_list, batch_first=True, padding_value=0)
    #print(batch_img_pad.shape)
    return batch_img_pad, batch_caption_list, batch_ID_list

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
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        txt = self.txt_dset[idx] # python str
        if type(txt) == type(float("nan")): # capture the case of empty "Impression" sections
            txt = " "
        
        img = torch.from_numpy(img) # torch, (3, 320, 320)
        img = kornia.geometry.transform.resize(img, size=(224, 224))
        if self.transform:
            img = self.transform(img)
        #sample = {'img': img, 'txt': txt }
        
        return img, txt
        
class ImgEmbedDataset(data.Dataset):

    def __init__(self, img_embed_path, text_path, column="impression"):
        super.__init__()
        self.img_embed_dset = sorted(glob.glob(img_embed_path, '/*'))
        self.text_dset = pd.read_csv(text_path)[column]
        
    
    def __len__(self):
        return len(self.text_dset)

    def __getitem__(self, idx):

        img_embed = torch.load(self.img_embed_dset(idx))
        #if self.transform:
        print(img_embed.shape)
        text = self.text_dset[idx]

        return img_embed, text
