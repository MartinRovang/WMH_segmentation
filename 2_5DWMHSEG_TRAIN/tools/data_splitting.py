from torch._C import dtype
from torch.functional import norm
from torch.utils.data import DataLoader, Dataset
import torch
import einops
import glob
import os, shutil
import monai
import numpy as np
import nibabel as nib
from rich.progress import track
import matplotlib.pyplot as plt
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk
import six
import seaborn as sns
from scipy.spatial import distance
import pickle
import json
import skimage
import sys
sys.path.insert(1, '../')
import utils

np.random.seed(435323)

def data_splitter(path = '/mnt/HDD16TB/martinsr/annotations_v0.2', new_annot = True):

    all_files = glob.glob(path+f'/*')
    all_files_ = [x.replace('\\', '/') for x in all_files]
    patients_id = [x.split('_')[0] for x in all_files_]
    patients_id = list(dict.fromkeys(patients_id))
    patients_data_paths = []
    patients_data_paths_test = []
    for patient_id in patients_id:
        if 'A' in patient_id:
            flair_path = patient_id+'_Flair_BFCorr.nii.gz'
            annot_path = patient_id+'_cluster_size5_masked.nii.gz'
            if new_annot:
                flair_path = annot_path
            if flair_path not in all_files_ or annot_path not in all_files_:
                print(f'Files does not exist {flair_path} | {annot_path}')
            else:
                patients_data_paths_test.append([flair_path, annot_path])
        else:
            flair_path = patient_id+'_Flair_BFCorr.nii.gz'
            annot_path = patient_id+'_cluster_size5_masked.nii.gz'
            if new_annot:
                flair_path = annot_path
            if flair_path not in all_files_ or annot_path not in all_files_:
                print(f'Files does not exist {flair_path} | {annot_path}')
            else:
                patients_data_paths.append([flair_path, annot_path])


    total_datapoints = len(patients_data_paths)

    if not os.path.isdir(f'{path}/test'):
        os.mkdir(f'{path}/test')
    if not os.path.isdir(f'{path}/train'):
        os.mkdir(f'{path}/train')
    if not os.path.isdir(f'{path}/val'):
        os.mkdir(f'{path}/val')

    for files in patients_data_paths_test:
        flair_source = str(files[0]).split('/')
        pasient_id = str(flair_source[-1]).split('_')[0]
        os.mkdir(f'{path}/test/{pasient_id}')
        flair_dest = path+f'/test/{pasient_id}'+'/FLAIR.nii.gz'
        mask_dest = path+f'/test/{pasient_id}'+'/annot.nii.gz'

        if not new_annot:
            shutil.move(str(files[0]), flair_dest)
        shutil.move(str(files[1]), mask_dest)
        #exit()

    indices = np.arange(0, total_datapoints)
    np.random.shuffle(indices)

    train_amount = 0.85
    total_train = int(total_datapoints*train_amount)
    print(f'Total training data: {total_train}')

    for count, index in enumerate(indices):
        files = patients_data_paths[index]
        flair_source = str(files[0]).split('/')
        Lock = False
        if count < total_train:
            pasient_id = str(flair_source[-1]).split('_')[0]
            try:
                os.mkdir(f'{path}/train/{pasient_id}')
                flair_dest = path+f'/train/{pasient_id}'+'/FLAIR.nii.gz'
                mask_dest = path+f'/train/{pasient_id}'+'/annot.nii.gz'
                Lock = True
            except:
                print('Already exist train')
        else:
            try:
                pasient_id = str(flair_source[-1]).split('_')[0]
                os.mkdir(f'{path}/val/{pasient_id}')
                flair_dest = path+f'/val/{pasient_id}'+'/FLAIR.nii.gz'
                mask_dest = path+f'/val/{pasient_id}'+'/annot.nii.gz'
                Lock = True
            except:
                print('Already exits val')

        if Lock:
            if not new_annot:
                shutil.move(str(files[0]), flair_dest)
            shutil.move(str(files[1]), mask_dest)
            print('moved file')
        # exit()



def data_splitter_no_annot(path = '/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/fazekas3_220107', new_annot = True):

    all_files = glob.glob(path+f'/*.nii.gz')
    # all_files_ = [x.replace('\\', '/') for x in all_files]
    # patients_id = list(dict.fromkeys(patients_id))
    # patients_data_paths = []
    # patients_data_paths_test = []

    if not os.path.exists(f'{path}/train_faz'):
        os.mkdir(f'{path}/train_faz')
    if not os.path.exists(f'{path}/val_faz'):
        os.mkdir(f'{path}/val_faz')
    
    train = np.random.choice(all_files, int(len(all_files)*0.8) ,replace=False)
    val = [x for x in all_files if x not in train]


    subjects_train = [x.split('/')[-1] for x in train]
    patientid_train = [x.split('_')[0] for x in subjects_train]
    subjects_val = [x.split('/')[-1] for x in val]
    patientid_val = [x.split('_')[0] for x in subjects_val]


    for patientpath, patientid in zip(train, patientid_train):
        print(patientpath,patientid)
        flair_dest = path+f'/train_faz/{patientid}'+'/FLAIR.nii.gz'
        os.mkdir(f'{path}/train_faz/{patientid}')
        shutil.copy(patientpath, flair_dest)

    for patientpath, patientid in zip(val, patientid_val):
        #pass
        flair_dest = path+f'/val_faz/{patientid}'+'/FLAIR.nii.gz'
        os.mkdir(f'{path}/val_faz/{patientid}')
        shutil.copy(patientpath, flair_dest)
    
    print('moved files')

    
    # for count, index in enumerate(indices):
    #     files = patients_data_paths[index]
    #     flair_source = str(files[0]).split('/')
    #     Lock = False
    #     if count < total_train:
    #         pasient_id = str(flair_source[-1]).split('_')[0]
    #         try:
    #             os.mkdir(f'{path}/train_faz/{pasient_id}')
    #             flair_dest = path+f'/train_faz/{pasient_id}'+'/FLAIR.nii.gz'
    #             mask_dest = path+f'/train_faz/{pasient_id}'+'/annot.nii.gz'
    #             Lock = True
    #         except:
    #             print('Already exist train')
    #     else:
    #         try:
    #             pasient_id = str(flair_source[-1]).split('_')[0]
    #             os.mkdir(f'{path}/val_faz/{pasient_id}')
    #             flair_dest = path+f'/val_faz/{pasient_id}'+'/FLAIR.nii.gz'
    #             mask_dest = path+f'/val_faz/{pasient_id}'+'/annot.nii.gz'
    #             Lock = True
    #         except:
    #             print('Already exits val')

    #     if Lock:
    #         if not new_annot:
    #             shutil.move(str(files[0]), flair_dest)
    #         shutil.move(str(files[1]), mask_dest)
    #         print('moved file')
        # exit()


data_splitter_no_annot()
