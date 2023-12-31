#import pydicom
import os
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
import nibabel as nib
import json

from torch.utils import data

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

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

        
        preproc_frames = volume[h_t + d:h_t + 224-d, w_t + d:w_t + 224-d, :]
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

class CTSeg3DDataset(data.Dataset):
    def __init__(self, img_path, txt_path, column='report', transform=None, is_train=True):
        super().__init__()
        img_dset = sorted(glob.glob(img_path+'/*.npy'))
        img_dset.sort(key=natural_keys)
        #print('img_dset', img_dset)
        self.img_dset = img_dset
        self.txt_dset = pd.read_csv(txt_path)[column]
        self.transform = transform
        self.is_train = is_train
        if self.is_train:
            self.img_dset = self.img_dset[:int(0.95*len(img_dset))]
            self.txt_dset = self.txt_dset[:int(0.95*len(img_dset))]
        
    def __len__(self):
        return len(self.txt_dset)
    
    def __getitem__(self, idx):
        #print('idx', idx)
        img = np.load(self.img_dset[idx])
        
        img = torch.from_numpy(img)
        txt = self.txt_dset[idx]
        if type(txt) == type(float("nan")): # capture the case of empty "Impression" sections
            txt = " "
        
        if self.transform:
            img = self.transform(img)
        
        return img, txt

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

class rectalMRIDataset(data.Dataset):
    def __init__(self, img_path, txt_path, transform=None, is_train=True):
        super().__init__()
        #self.args = args
        #self.config = config
        #self.config
        self.img_path = img_path
        self.txt_path = txt_path
        self.is_train = is_train
        #self.csv = pd.read_csv(self.args.csv_dir)
        with open(self.txt_path, 'r') as json_reader:
            self.mri_label = json.load(json_reader)
        
        self.all_subject = sorted(glob.glob(self.img_path + '/**/*.nii.gz'))
        #print(self.all_subject)
        # split train/val/test set (not depending on the seed) -> fixed division
        if self.is_train:
            self.subject_list = self.all_subject[:int(len(self.all_subject)*0.95)]  # 90%
        else:
            self.subject_list = self.all_subject[int(len(self.all_subject)*0.95):]  # 6%
        
        self.subject_list = sorted(self.subject_list)
    
        ###Faulty Samples###
        self.fug = ['00898791_20220723_1111', '00144977_20220426_1710', '01905101_20210810_1827', '00314908_20210113_1501', '00690419_20211223_1136', '00580054_20220223_1355', '01841626_20210123_1052', '02003454_20220625_1526']
        #print(len(self.subject_list))
        #text_path = '/scratch/slurm-user3/changsun/data/rectal_MRI_label/202301_MRI_impression_final.json'
        '''
        with open(text_path, 'r') as json_reader:
            mri_label_old = json.load(json_reader)
        print('Length of orig_dict_keys: ', len(list(mri_label_old.keys())))
        '''
        filtered_dict_keys = list(self.mri_label.keys())
        print('Length of filtered_dict_keys: ', len(filtered_dict_keys))
        
        self.subject_list_copy = self.subject_list[:]
        real_training_data = {}
        for idx, item in enumerate(self.subject_list_copy):
            #print('idx then: ', idx)
            if 'SURVEY' in item:
                if item in self.subject_list:
                    self.subject_list.remove(item)
            for elem in self.fug:
                if elem in item:
                    if item in self.subject_list:
                        self.subject_list.remove(item)
                    else:
                        continue
            patient_ID = item.split('/')[-2]
            patient_ID_key = patient_ID[:-11]
            #print(patient_ID, patient_ID_key)
            if patient_ID_key not in filtered_dict_keys:
                #print('self.subject_list: ', self.subject_list[idx])
                ##print('idx now: ', idx)
                #print('before length: ', len(self.subject_list))
                if item in self.subject_list:
                    self.subject_list.remove(item)
                ##print('after length: ', len(self.subject_list))
                #print("POP complete")
            else:
                real_training_data[patient_ID_key] = self.mri_label[patient_ID_key]
        
        print('Length of real training data dict: ', len(list(real_training_data.keys())))
        with open('formatted_rectal_mri_report_real.json', 'w') as jwriter:
            json.dump(real_training_data, jwriter, indent=4)
        
        #print('self.subject_list: ', self.subject_list)
        print(len(self.subject_list))
    
    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, idx):
        
        subject = self.subject_list[idx]
        patient_ID = subject.split('/')[-2]
        patient_ID_key = patient_ID[:-11]

        #patient_ID_key = patient_ID[:-4]
        #ID = subject.split('/')[-1]
        
        #TEST EACH MRI VOLUME
        #volume_list = sorted(glob.glob(subject+'/*'))
        volume = nib.load(subject)  
        #volume = np.load(subject + './OBL AXL FSE T2_image.nii.gz')
        #volume = np.load(subject + '/3d.npy')
        #print('Volume shape: ', volume.shape)

        
        caption = self.mri_label[patient_ID_key]
        
        # caption = self.df[subject]
        caption = pre_caption(caption, max_words=200)
        
        # Center crop and translation
        if self.is_train:
            d = random.randint(0, 10)
            h_t = random.randint(-d, d)
            w_t = random.randint(-d, d)
        else:
            d = 5
            h_t = 0
            w_t = 0

        preproc_frames = np.asarray(volume.dataobj)[h_t + d:h_t + 224+d, w_t + d:w_t + 224+d, :]
        preproc_frames = preproc_frames.astype(np.float32)
        #print(preproc_frames.shape)
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
        preproc_frames_bone = preproc_frames.clone().float()
        preproc_frames_bone[preproc_frames_bone > 1500] = 1500
        preproc_frames_bone[preproc_frames_bone < -500] = -500
        preproc_frames_bone = (preproc_frames_bone - preproc_frames_bone.min()) / (preproc_frames_bone.max() - preproc_frames_bone.min())
        preproc_frames_bone = preproc_frames_bone.unsqueeze(1)

        preproc_frames_cat = torch.cat([preproc_frames_brain, preproc_frames_brain, preproc_frames_brain], dim=1)
        # preproc_frames_cat = torch.cat([preproc_frames_brain, preproc_frames_subdural, preproc_frames_bone], dim=1)
        preproc_frames_cat = kornia.geometry.transform.resize(preproc_frames_cat,
                                                              size=(224, 224))
        
        preproc_frames_cat = preproc_frames_cat[:20,:,:,:]
        preproc_frames_cat = preproc_frames_cat[:20,:,:,:]
        if preproc_frames_cat.shape[0] < 20:
            padding  = torch.zeros(20-preproc_frames_cat.shape[0], preproc_frames_cat.shape[1], preproc_frames_cat.shape[2], preproc_frames_cat.shape[3])
            preproc_frames_cat = torch.cat((preproc_frames_cat, padding), dim=0)
        #print('here')
        #print('preproc_frames_cat.shape', preproc_frames_cat.shape)
        #print('subject', subject)
        ##dir_path = subject.split('/')
        #output_path = os.path.join(*dir_path[:-1])
        #output_file_path = '/'+output_path+'/'+str(dir_path[-1])+'_convert.pt'
        #torch.save(preproc_frames_cat, output_file_path)
        
        return preproc_frames_cat, caption, "rectalMRI", patient_ID_key

class brainMRIDataset(data.Dataset):
    def __init__(self, img_path, txt_path, transform=None, is_train=True):
        super().__init__()
        #self.args = args
        #self.config = config
        #self.config
        self.img_path = img_path
        self.txt_path = txt_path
        self.is_train = is_train
        self.mri_label = pd.read_csv(self.txt_path)
        #with open(self.txt_path, 'r') as json_reader:
            #self.mri_label = json.load(json_reader)
        
        self.all_subject = sorted(glob.glob(self.img_path + '/*.npy'))
        #print(self.all_subject)
        # split train/val/test set (not depending on the seed) -> fixed division
        if self.is_train:
            self.subject_list = self.all_subject[:int(len(self.all_subject)*0.90)]  # 90%
        else:
            self.subject_list = self.all_subject[int(len(self.all_subject)*0.90):]  # 6%
        
        self.subject_list = sorted(self.subject_list)
        
        print(len(self.subject_list))

    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, idx):
        
        subject = self.subject_list[idx]
        patient_ID = subject.split('/')[-1]
        patient_ID_key = patient_ID[:-13]

        #patient_ID_key = patient_ID[:-4]
        #ID = subject.split('/')[-1]
        
        #TEST EACH MRI VOLUME
        #volume_list = sorted(glob.glob(subject+'/*'))
        volume = np.load(subject)  
        #volume = np.load(subject + './OBL AXL FSE T2_image.nii.gz')
        #volume = np.load(subject + '/3d.npy')
        #print('Volume shape: ', volume.shape)
        #print('before index', patient_ID_key)
        #print('patient_ID_key: ', patient_ID_key)
        index = self.mri_label.index[self.mri_label['Patient'] == patient_ID_key]
        ##print('index', index)
        #print('index[0]', index[0])
        caption = self.mri_label.at[index[0], 'report']
        #caption = self.mri_label[patient_ID_key]
        # caption = self.df[subject]
        caption = pre_caption(caption, max_words=200)
        
        # Center crop and translation
        if self.is_train:
            d = random.randint(0, 10)
            h_t = random.randint(-d, d)
            w_t = random.randint(-d, d)
        else:
            d = 5
            h_t = 0
            w_t = 0

        preproc_frames = volume[:, h_t + d:h_t + 224+d, w_t + d:w_t + 224+d, :]
        preproc_frames = preproc_frames.astype(np.float32)
        preproc_frames = torch.from_numpy(preproc_frames)
        #print('preproc_frames', preproc_frames.shape)
        preproc_frames = preproc_frames.permute(0, 3, 1, 2)
        
        # Brain window
        preproc_frames_brain = preproc_frames[:, 0, :, :].clone()
        preproc_frames_brain[preproc_frames_brain > 80] = 80
        preproc_frames_brain[preproc_frames_brain < 0] = 0
        preproc_frames_brain = (preproc_frames_brain - preproc_frames_brain.min()) / (preproc_frames_brain.max() - preproc_frames_brain.min())
        preproc_frames_brain = preproc_frames_brain.unsqueeze(1)
        
        # Subdural window
        preproc_frames_subdural = preproc_frames[:, 1, :, :].clone()
        preproc_frames_subdural[preproc_frames_subdural > 170] = 170
        preproc_frames_subdural[preproc_frames_subdural < -10] = -10
        preproc_frames_subdural = (preproc_frames_subdural - preproc_frames_subdural.min()) / (preproc_frames_subdural.max() - preproc_frames_subdural.min())
        preproc_frames_subdural = preproc_frames_subdural.unsqueeze(1)
        
        # Bone window
        preproc_frames_bone = preproc_frames[:, 2, :, :].clone().float()
        preproc_frames_bone[preproc_frames_bone > 1500] = 1500
        preproc_frames_bone[preproc_frames_bone < -500] = -500
        preproc_frames_bone = (preproc_frames_bone - preproc_frames_bone.min()) / (preproc_frames_bone.max() - preproc_frames_bone.min())
        preproc_frames_bone = preproc_frames_bone.unsqueeze(1)

        preproc_frames_cat = torch.cat([preproc_frames_brain, preproc_frames_brain, preproc_frames_brain], dim=1)
        # preproc_frames_cat = torch.cat([preproc_frames_brain, preproc_frames_subdural, preproc_frames_bone], dim=1)
        preproc_frames_cat = kornia.geometry.transform.resize(preproc_frames_cat,
                                                              size=(224, 224))
        
        
        preproc_frames_cat = preproc_frames_cat[:20,:,:,:]
        if preproc_frames_cat.shape[0] < 20:
            padding  = torch.zeros(20-preproc_frames_cat.shape[0], preproc_frames_cat.shape[1], preproc_frames_cat.shape[2], preproc_frames_cat.shape[3])
            preproc_frames_cat = torch.cat((preproc_frames_cat, padding), dim=0)
        #preproc_frames_cat = preproc_frames_cat.unsqueeze(0)
        #print('here')
        #print('preproc_frames_cat.shape', preproc_frames_cat.shape)
        ##print('subject', subject)
        ##dir_path = subject.split('/')
        #output_path = os.path.join(*dir_path[:-1])
        #output_file_path = '/'+output_path+'/'+str(dir_path[-1])+'_convert.pt'
        #torch.save(preproc_frames_cat, output_file_path)
        
        return preproc_frames_cat, caption, "brainMRI", patient_ID_key

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
