
from curses import meta
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
from scipy.stats import entropy

def read_data(path):
    """
    Reads the data from the csv file and returns a pandas dataframe
    """
    return pd.read_csv(path, sep='\t',encoding = "iso-8859-1")

def dataprocesser_pred(pathtrain, filename = 'FLAIR.nii.gz'):

    patients_train = [file for file in os.listdir(pathtrain) if file not in ['README.md','desktop.ini','info']]
    
    volumestrain = []
    for patient in patients_train:
        pat_files_flair = glob.glob(os.path.join(pathtrain, patient)+f'/*{filename}')
        PASS_lock = True
        if len(pat_files_flair) == 0:
            PASS_lock = False
        
        if PASS_lock:
            for file in os.listdir(os.path.join(pathtrain, patient)):
                if f'{filename}' in file:
                    volumestrain.append(os.path.join(pathtrain,patient, file))
    
    train_files = [{"image": img} for img in volumestrain]
    return train_files

def get_drift(path_train:str = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/train") -> None:
    import seaborn as sns
    path = "/mnt/CRAI-NAS/all/martinsr/test_monai/WMHSEG_0.5/outputs/2022-05-09/23-35-18/"
    path_test = path + "testdatasplit_edit.txt"
    path_train = path + "traindatasplit.txt"
    # paths = {'Training Internal': path_train, 'Test internal': path_test}
    paths = {'Training Internal': path_train}

    metadata = read_data('../dataanalysis/scannerID_2.csv')
    id_as_prisma = []
    for modelname, ID in zip(metadata['ManufacturerModelName'], metadata['subject-ass']):
        if modelname == 'Prisma':
            id_as_prisma.append(ID)


    data = {}
    for key in paths:
        with open(paths[key], 'r') as f:
            temp = eval(f.read())
        data = {'Dataset': [], 'entropy': [], 'zscored': []}
        tmp_data = []
        for subject in temp:
            try:
                image = subject['image'].split('/')
                ID = image[-2]
                if ID in id_as_prisma:
                    image = "/".join(image[:-1])+'/FLAIR.nii.gz'
                    print(image)
                    load_data = nib.load(image).get_fdata()
                    load_data = load_data.flatten()
                    load_data[load_data > 0] = load_data
                    load_data = (load_data - np.mean(load_data))/np.std(load_data)
                    tmp_data += list(load_data)
            except Exception as e:
                print(e)
        data['Dataset'] += ['Training Internal']*len(tmp_data)
        data['zscored'] += tmp_data
    
    test_external_paths = glob.glob("/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/test_sample/*")
    for subject_folder in test_external_paths:
        tmp_data = []
        try:
            subject = subject_folder+'/FLAIR.nii.gz'
            load_data = nib.load(subject).get_fdata()
            load_data = load_data.flatten()
            load_data[load_data > 0] = load_data
            load_data = (load_data - np.mean(load_data))/np.std(load_data)
            tmp_data += list(load_data)
        except Exception as e:
            print(e)
        data['Dataset'] += ["Test external"]*len(tmp_data)
        data['zscored'] += tmp_data

    sns.boxplot(data=data, y ='zscored', x = 'Dataset')
    # sns.boxplot(data=data, x ='Dataset', y="mean")
    plt.tight_layout()
    plt.savefig(f'../dataanalysis/DRIFT_mean.pdf')
    plt.close()
    print('Saved')


get_drift()