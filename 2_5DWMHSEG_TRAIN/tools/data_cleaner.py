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


def clean_data(path:str = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/test") -> None:
    """
    Cleans the data in the given path (removes empty folder/only annot).
    """
    all_files = glob.glob(path+f'/*')
    for folder in all_files:
        if not os.path.isfile(folder+'/FLAIR.nii.gz'):
            if os.path.isdir(path+'/empty'):
                shutil.move(folder, path+'/empty')
            else:
                os.mkdir(path+'/empty')
                shutil.move(folder, path+'/empty')
        
            print(f'Folder moved; {folder}')
        


if __name__ == '__main__':
    clean_data()
