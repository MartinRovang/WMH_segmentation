#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import os
import functools
import itertools
import operator
import pickle
import scipy.stats as stats

train = pd.DataFrame(pickle.load(open('../dataanalysis/fazekas_data_train.pickle', 'rb')))
val = pd.DataFrame(pickle.load(open('../dataanalysis/fazekas_data_val.pickle', 'rb')))


# %%


def plot_histogram(data, title, xlabel, ylabel, bins=100, target = 0):
    data_lesions_path = data['labelpath'][data['label'] == target]
    data_lesions = data['lesions_cleaned'][data['label'] == target]
    flat_list_lesions = np.array([item for sublist in data_lesions for item in sublist])

    # print(data_lesions)
    max_list = [np.max(x) for x in data_lesions]
    max_idx = np.argmax(max_list)
    max_val = np.max(max_list)
    print('max lesion size: ', max_val)
    print(data_lesions_path.values[max_idx])

    plt.hist(np.log(flat_list_lesions), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# 
# max lesion size:  2229.0
# /mnt/HDD16TB/martinsr/DatasetWMH211018_v2/train/D222025K-2/annot.nii.gz
plot_histogram(train, 'Fazekas 0', 'Lesions size [log]', 'Count', target = 0)
plot_histogram(train, 'Fazekas 1', 'Lesions size [log]', 'Count', target = 1)
plot_histogram(train, 'Fazekas 2', 'Lesions size [log]', 'Count', target = 2)
plot_histogram(train, 'Fazekas 3', 'Lesions size [log]', 'Count', target = 3)
# %%

def plot_two_moments(data):
    flat_list = [item for sublist in data for item in sublist]
    print('Mean:', np.mean(flat_list))
    print('Standard deviation:', np.std(flat_list))
    print('Skewness:', stats.skew(flat_list))
    print('Amount of lesions: ', len(flat_list))


plot_two_moments(train['lesions'][train['label'] == 0])
plot_two_moments(train['lesions'][train['label'] == 1])
plot_two_moments(train['lesions'][train['label'] == 2])
plot_two_moments(train['lesions'][train['label'] == 3])



# %%


def plot_boxplot(data):
    datanew = {'label': [], 'lesions': []}
    for i, label in enumerate(data['label']):
        datanew['label'].append(label)
        datanew['lesions'].append(np.mean(data['lesions'][i]))
    sns.boxplot(data = datanew, x='label', y='lesions')
    plt.xlabel('Fazekas score')
    plt.ylabel('Lesions mean size')
    plt.show()
    datanew = {'label': [], 'lesions': []}
    for i, label in enumerate(data['label']):
        datanew['label'].append(label)
        datanew['lesions'].append(np.std(data['lesions'][i]))
    sns.boxplot(data = datanew, x='label', y='lesions')
    plt.xlabel('Fazekas score')
    plt.ylabel('Lesions std size')
    plt.show()
    datanew = {'label': [], 'lesions': []}
    for i, label in enumerate(data['label']):
        datanew['label'].append(label)
        datanew['lesions'].append(len(data['lesions'][i]))
    sns.boxplot(data = datanew, x='label', y='lesions')
    plt.xlabel('Fazekas score')
    plt.ylabel('Lesions amount')
    plt.show()


plot_boxplot(train)
# %%

def plot_placement(data):
    fazekas_labels = [0, 1, 2, 3]
    color = ['black', 'red', 'green', 'blue']
    for label in fazekas_labels:
        dataplacement = train['lesions_centroid_mm_cleaned']
        lesion_size = train['lesions_cleaned']
        lesion_size = lesion_size[data['label'] == label]
        dataplacement = dataplacement[data['label'] == label]
        # fig, ax = plt.subplots(1, 3)
        # [ax.set_xlabel('mm') and ax.set_ylabel('mm') for ax in ax.flatten()]
        fig, ax = plt.subplots(figsize = (20, 20))

        subject_dispx = np.array([])
        subject_dispy = np.array([])
        subject_dispsize = np.array([])
        for subject_disp, lesion_siz in zip(dataplacement, lesion_size):
            subject_disp = np.array(subject_disp)
            lesion_siz = np.array(lesion_siz)
            #plt.plot(subject_disp[:, 0], subject_disp[:, 1], '.', color='black', alpha = 0.1)
            subject_dispx = np.append(subject_dispx, subject_disp[:, 0])
            subject_dispy = np.append(subject_dispy, subject_disp[:, 1])
            subject_dispsize = np.append(subject_dispsize, lesion_siz)
            # ax[0].plot(subject[:, 0], subject[:, 1], '.', color=color[label], alpha = 0.1)
            # ax[1].plot(subject[:, 1], subject[:, 2], '.', color=color[label], alpha = 0.1)
            # ax[2].plot(subject[:, 2], subject[:, 0], '.', color=color[label], alpha = 0.1)
        sns.scatterplot(x=subject_dispx, y=subject_dispy, size=subject_dispsize, sizes=(40, 10000), alpha=.5, ax=ax, legend = 'brief', linewidth=1, edgecolor='pink', color='black')
        handles, labels = ax.get_legend_handles_labels()
        for h in handles:
            sizes = [s / 5 for s in h.get_sizes()]
            h.set_sizes(sizes)
        plt.title(f'Fazekas {label}', fontsize = 20)
        plt.xlabel('mm', fontsize = 20)
        plt.ylabel('mm', fontsize = 20)
        plt.legend(handles, labels, fontsize='25', title_fontsize='60', bbox_to_anchor= (1.15,1), labelspacing = 4)
        plt.tight_layout()
        plt.savefig(f'../dataanalysis/lesion_scatter{label}.pdf')
    plt.show()

plot_placement(train)

# %%

# %%

# %%
