
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

def get_radiomics_features(path, datatype = 'val', fazekas_text_path= "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/Fazekas.csv"):
    from radiomics import featureextractor
    import SimpleITK as sitk
    extractor = featureextractor.RadiomicsFeatureExtractor()
    features_extracted = {}
    path += '/'+datatype

    data, data = utils.dataprocesser(path, path, filename = 'FLAIR.nii.gz')
    for i, subject in enumerate(data):
        try:
            flair_data = nib.load(subject['image']).get_fdata()
            label = nib.load(subject['label']).get_fdata().astype('uint32')
            label[label > 0] = 1

            flair_id = subject['image'].split("/")[-2]
            features_extracted[flair_id] = {}

            flair_data = sitk.GetImageFromArray(flair_data)
            label = sitk.GetImageFromArray(label)
            label = sitk.Cast(label, sitk.sitkInt32) # MAKE SURE IT IS CASTED IN INT
            flair_data = sitk.Cast(flair_data, sitk.sitkInt32) # MAKE SURE IT IS CASTED IN INT

            features = extractor.execute(flair_data, label)
            for key, val in six.iteritems(features):
                features_extracted[flair_id][key] = val
        except EOFError:
            print(f"EOFError {subject['image']}")
    
    fazekas_datafile = pd.DataFrame(pd.read_csv(fazekas_text_path, delimiter = ',', header=0))
    path_subjects = [str(x)+'-'+str(y) for x, y in zip(fazekas_datafile['Subject'], fazekas_datafile['assessment'])]
    fazekas_scores = [x for x in fazekas_datafile['Fazekas']]
    fazekas_scores_table = {key:faz for key, faz in zip(path_subjects, fazekas_scores)}

    pops = []
    for key in features_extracted:
        if key in fazekas_scores_table:
            features_extracted[key]['Fazekas'] = fazekas_scores_table[key]
            if fazekas_scores_table[key] == -999:
                pops.append(key)
        else:
            pops.append(key)
    
    for popping in pops:
        features_extracted.pop(popping, None)
        
    with open(f'dataanalysis/{datatype}_radiometrics.pickle', 'wb') as handle:
        pickle.dump(features_extracted, handle)

    # with open(f'{datatype}_radiometrics.pickle', 'rb') as handle:
    #     b = pickle.load(handle)