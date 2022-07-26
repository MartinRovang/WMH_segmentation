
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



def get_scanner_info(path_data = '/mnt/HDD16TB/martinsr/DatasetWMH211018_v2', path_txt:str = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/scannerID_2.csv"):
    """writes the scanner info to text file."""

    # path_data = 'C:/Users/Gimpe/Google Drive/Master -Signal_processingWORK/Masteroppgave/Main_code/data'
    # path_txt = 'C:/Users/Gimpe/Documents/GitHub/WMH-Segmentation/WMHSEG_0.2/dataanalysis/scannerID_2.csv'

    data = pd.read_csv(path_txt, header = 0, error_bad_lines=False, delimiter = '\t',encoding = "ISO-8859-1")

    test_path = 'test'
    train_path = 'train'
    val_path = 'val'
    data_paths = [train_path, val_path, test_path]
    remove_data = ['subject-ass' ,'RepetitionTime', 'DeviceSerialNumber', 'SoftwareVersions', 'MagneticFieldStrength', 'RepetitionTime', 'EchoTime', 'FlipAngle']
    output_info = {'train': {}, 'val':{}, 'test':{}}
    for data_path in data_paths:
        data_path_ = path_data+'/'+data_path
        voxel_size = []
        for subject in os.listdir(data_path_):
            subject_path = data_path_+'/'+subject
            if '.txt' not in subject_path:
                for scan in os.listdir(subject_path):
                    if 'FLAIR' in scan:
                        scan_info = data[data['subject-ass'] == subject]
                        if not scan_info.empty:
                            scanner_info = scan_info.iloc[0]
                            scanner_info = scanner_info.to_dict()
                            
                            flair_header = nib.load(subject_path+'/FLAIR.nii.gz').header
                            sx, sy, sz = flair_header.get_zooms()
                            voxel_size.append(sx*sy*sz)
                            # print(voxel_size)

                            for key in scanner_info:
                                if key in remove_data:
                                    pass
                                else:
                                    if scanner_info[key] in output_info[data_path]:
                                        output_info[data_path][scanner_info[key]] += 1
                                    else:
                                        output_info[data_path][scanner_info[key]] = 1
        if len(os.listdir(data_path_)) > 0:
            output_info[data_path]['mean_voxel_size[mm^3]'] = np.mean(voxel_size)

    # print(output_info)
    # make text file with scanner info
    with open('../dataanalysis/datainfo.txt', 'w') as f:
        f.write(str(output_info))
        print('data written to file.')


get_scanner_info()