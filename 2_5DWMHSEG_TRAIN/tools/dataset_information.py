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


def get_dataset_information(paths:list) -> list:
	all_datasets = []
	for path in paths:
		path_files = glob.glob(path+'/*')
		voxel_sizes = []
		mean_array = []
		std_array = []
		total_wmh_volume = []
		for i, subject in enumerate(path_files):
			subject = subject+'/reduced_an.nii.gz'
			try:
				annot = nib.load(subject).get_fdata()
				header_info = nib.load(subject).header
				sx, sy, sz = header_info.get_zooms()
				voxel_size = sx*sy*sz
				total_wmh_volume.append(np.sum(annot)*voxel_size*0.001)
				voxel_sizes.append(voxel_size)
				if voxel_size != 1.00:
					with open('../dataanalysis/dataset_voxelsize_not_one.txt', 'a') as fp:
						output = f'voxelsize: {voxel_size}; {subject}\n'
						print(output)
						fp.write(output)
			except Exception as e:
				print(f'{e}; {subject}')
		avg_voxelsize = np.mean(voxel_sizes)
		total_volumes = len(voxel_sizes)
		all_datasets.append({'avg_voxelsize': avg_voxelsize, 'total_volumes': total_volumes})
	
		mean = np.mean(total_wmh_volume)
		std = np.std(total_wmh_volume)
		median = np.median(total_wmh_volume)

		#all_lesions = np.append(all_lesions,)

		print(median)
		print(f'{mean:.2f} +- {std:.2f}')


		sns.rugplot(x = total_wmh_volume, alpha = 0.5, linewidth = 1, expand_margins = False)
		sns.histplot(x = total_wmh_volume, alpha = 0.5, linewidth = 0)
		plt.xlabel('Total WMH volume (mL)')
		plt.ylabel('Number of participants')
		plt.savefig(f'../dataanalysis/total_wmh_volume_{path.split("/")[-1]}.pdf')
		plt.close()
		with open(f'../dataanalysis/total_wmh_volume_{path.split("/")[-1]}.txt', 'a') as fp:
			output = f"""mean: {mean:.2f} +- {std:.2f}\n min: {np.min(total_wmh_volume)}\n max: {np.max(total_wmh_volume)}\n
			median: {median:.2f}, 25 and 75 percentile: {np.percentile(total_wmh_volume, 25):.2f} and {np.percentile(total_wmh_volume, 75):.2f}\n
			"""
			print(output)
			fp.write(output)

	#return all_datasets, [avg_mean, avg_std]




if __name__ == "__main__":
	paths = ['/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/alldata']
	get_dataset_information(paths)
