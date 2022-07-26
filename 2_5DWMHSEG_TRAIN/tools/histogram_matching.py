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



def get_histogram_for_matching(path_train:str = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/train", path_target:str = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/test") -> None:
    
    # TEST FUNCTION
    path_train = r"C:\Users\MartinR\Google Drive\Master -Signal_processingWORK\Masteroppgave\Main_code\data\test_sample"
    path_target = r"C:\Users\MartinR\Google Drive\Master -Signal_processingWORK\Masteroppgave\Main_code\data\test_sample_brn"
    data_train, data_target = dataprocesser(path_train, path_target, filename = 'FLAIR.nii.gz')
    N = 500
    cdf_data = {'cumul': [], 'datatype': [], 'x': []}
    for i, subject in enumerate(data_train):
        try:
            flair_data_ = nib.load(subject['image']).get_fdata()
            # sh = nib.load(subject['image']).get_fdata().shape
            # Contrast stretching
            # flair_data_[flair_data_ < 0] = 0
            # flair_data_[flair_data_ > 255] = 255
            # flair_data_ = flair_data_.astype(np.uint8)
            # values, base = np.histogram(flair_data_.flatten(), bins=N)
            # data_full_values = np.append(data_full_values, values)

            if i < 1:
                values, base = np.histogram(flair_data_.flatten(), bins=N)
                values = values[:, None]
                base = base[:, None]
                cumulative = np.cumsum(values)
                cumulative = cumulative/np.max(cumulative)
                vals = base.copy()
                bins = values.copy()
            else:
                values_, base = np.histogram(flair_data_.flatten(), bins=N)
                values_ = values_[:, None]
                base = base[:, None]
                vals = np.concatenate((vals, base), axis = 1)
                bins = np.concatenate((bins, values_), axis = 1)
                cumulative = np.cumsum(values_)
                cumulative = cumulative/np.max(cumulative)


            cdf_data['cumul'].append(cumulative)
            cdf_data['datatype'].append(['training']*len(cumulative))
            cdf_data['x'].append(np.arange(0, len(cumulative)))

        except EOFError:
            print(subject['image'])
            print('EOF error!')



    # # plt.close()
    vals_mean = np.mean(vals, axis = 1)
    bins_mean = np.mean(bins, axis = 1)
    # exit()
    end_dist = np.array([])
    for j, n in enumerate(bins_mean[:-1]):
        val = np.array([vals_mean[j]]*int(n))
        end_dist = np.append(end_dist.flatten(), val)
    




    # plt.hist(FLAIR.flatten(), bins=N)
    # plt.show()
    # plt.hist(end_dist.flatten(), bins=N)
    # plt.show()

    with open(f'dataanalysis/reference_histogram.pickle', 'wb') as handle:
        pickle.dump(end_dist, handle)
    


    

    # values, base = np.histogram(end_dist.copy(), bins=N)
    # cumulative = np.cumsum(values)
    # cumulative = cumulative/np.max(cumulative)
    # plt.plot(np.arange(0, len(cumulative)), cumulative, '--', color = 'green', linewidth = 2, label = 'mean cdf', alpha = 0.6)

    
    exit()
    



    for i, subject in enumerate(data_target):
        try:
            flair_data___ = nib.load(subject['image']).get_fdata()
            # flair_data[flair_data < 0] = 0
            # flair_data[flair_data > 255] = 255
            # flair_data = flair_data.astype(np.uint8)
            shap_ = nib.load(subject['image']).get_fdata().shape

            values__, base = np.histogram(flair_data___.flatten(), bins=N)
            
            cumulative = np.cumsum(values__)
            cumulative = cumulative/np.max(cumulative)

            cdf_data['cumul'].append(cumulative)
            cdf_data['datatype'].append(['test']*len(cumulative))
            cdf_data['x'].append(np.arange(0, len(cumulative)))
            
            if i == 2:
                flair_data = flair_data___
        except EOFError:
            print(subject['image'])
            print('EOF error!')

    cdf_data['cumul'] = np.array(cdf_data['cumul']).flatten()
    cdf_data['datatype'] = np.array(cdf_data['datatype']).flatten()
    cdf_data['x'] = np.array(cdf_data['x']).flatten()
    print(cdf_data['x'].shape)
    print(cdf_data['datatype'].shape)
    print(cdf_data['cumul'].shape)
    cdf_data = pd.DataFrame(cdf_data)
    sns.lineplot(data = cdf_data, x = 'x', y = 'cumul', hue = 'datatype', palette = 'Set1')
    plt.savefig(f'dataanalysis/cdf_plot.png')
    plt.close()
    exit()

    # exit()

    print(values_.shape)

    
    # print(data_full_values.shape)
    # fig, ax = plt.subplots(1, 2)
    # values, base = np.histogram(flair_data.flatten(), bins=N)
    # cumulative = np.cumsum(values)
    # values2, base = np.histogram(flair_data_.flatten(), bins=N)
    # cumulative2 = np.cumsum(values2)

    # ax[0].plot(cumulative)
    # ax[1].plot(cumulative2)
    # plt.show()
    from skimage.exposure import match_histograms
    matched = match_histograms(flair_data.flatten(), end_dist)
    print(matched.shape)
    # print(matched.shape)
    # print(shap_)

    # matched = matched - np.mean(matched)
    # matched = matched / np.std(matched)
    values, base = np.histogram(matched.flatten(), bins=N)
    cumulative = np.cumsum(values)
    cumulative = cumulative/np.max(cumulative)
    plt.plot(np.arange(0, len(cumulative)),cumulative, '--', color = 'blue', linewidth = 2, label = 'matched', alpha = 0.6)
    
    plt.legend()
    plt.savefig('dataanalysis/matching.pdf')
    # plt.show()
    exit()



    # flair_data_ = flair_data - np.mean(flair_data_)
    # flair_data_ = flair_data_ / np.std(flair_data_)

    # matched = matched - np.mean(matched)
    # matched = matched / np.std(matched)


    fig, ax = plt.subplots(1, 3)
    ax[0].hist(flair_data_.flatten(), bins=N, fc='k', ec='k')
    ax[1].hist(flair_data.flatten(), bins=N, fc='k', ec='k')
    ax[2].hist(matched.flatten(), bins=N, fc='k', ec='k')
    plt.show()


    matched = matched.reshape(shap_)


    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(flair_data_[100, :, :], cmap='gray')
    ax[1].imshow(flair_data[130, :, :], cmap='gray')
    ax[2].imshow(matched[130, :, :], cmap='gray')
    plt.show()
