

from cgi import test
from dataclasses import replace
from torch._C import dtype
from torch.functional import norm
from torch.utils.data import DataLoader, Dataset
import torch
import einops
import glob
import os, shutil
import monai
import numpy as np
from transform_functions import correct_input
import nibabel as nib
from rich.progress import track
import matplotlib.pyplot as plt
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk
from typing import Type, Union
from sklearn.model_selection import train_test_split
from rich.console import Console
console = Console()


class Minisampler3slice(monai.data.Dataset):
    """Loads the path of images and masks"""
    def __init__(self, data, transform = False, all_orientations = True):



        self.data_all = data.copy()
        self.transform = transform
        self.all_orientations = all_orientations

        # Remove one eight from each orthogonal direction
        self.rem_eight_slices_1 = data['image'].shape[2]//12
        self.rem_eight_slices_2 = data['image'].shape[3]//12
        self.rem_eight_slices_3 = data['image'].shape[4]//12

        # if self.all_orientations:
        #     random_slices_1 = np.arange(0, data['image'].shape[2]-self.rem_eight_slices_1*2)
        #     random_slices_2 = np.arange(0, data['image'].shape[3]-self.rem_eight_slices_2*2)
        #     random_slices_3 = np.arange(0, data['image'].shape[4]-self.rem_eight_slices_3*2)


        #     random_slices_1 = np.random.choice(random_slices_1, 100 , replace = False)
        #     random_slices_2 = np.random.choice(random_slices_2, 100 , replace = False)
        #     random_slices_3 = np.random.choice(random_slices_3, 100 , replace = False)
        for key in data:
            if key == 'image' or key == 'label':
                # border_fixed_image = remove_einops_border_noise(data[key])
                border_fixed_image = data[key]
   
                reshaped_in = einops.rearrange(border_fixed_image, 'b c d h w -> (b w) c h d')


                reshaped_in = correct_input(reshaped_in)

                reshaped_in =  reshaped_in.rot90(k = -2, dims = [-2, -1])
                reshaped_in = reshaped_in[self.rem_eight_slices_1:-self.rem_eight_slices_1, :, :, :]



                if self.all_orientations:
                    reshaped_in_sag = einops.rearrange(border_fixed_image, 'b c d h w -> (b d) c h w')
                    reshaped_in_cor = einops.rearrange(border_fixed_image, 'b c d h w -> (b h) c d w')


                    reshaped_in_sag = correct_input(reshaped_in_sag)
                    reshaped_in_cor = correct_input(reshaped_in_cor)


                    reshaped_in_sag = reshaped_in_sag[self.rem_eight_slices_2:-self.rem_eight_slices_2, :, :, :]
                    reshaped_in_cor = reshaped_in_cor[self.rem_eight_slices_3:-self.rem_eight_slices_3, :, :, :]


                    reshaped_in_cor =  reshaped_in_cor.rot90(k = 1, dims = [-2, -1])
                    reshaped_in_sag =  reshaped_in_sag.rot90(k = 1, dims = [-2, -1])
                    reshaped_in_cor = reshaped_in_cor.flip(3)
                    reshaped_in_sag = reshaped_in_sag.flip(3)


                    self.data_all[key] = {'ax':reshaped_in , 'sag': reshaped_in_sag, 'cor': reshaped_in_cor}
                else:
                    self.data_all[key] = reshaped_in
            else:
                pass
        if all_orientations:
            sizes = [self.data_all['image']['ax'].shape[0], self.data_all['image']['sag'].shape[0], self.data_all['image']['cor'].shape[0]]
            self.N = np.min(sizes)
        else:
            self.N = self.data_all['image'].shape[0]
            # self.num_of_iters = int(len(self.data['image'])//samples)
        # print(self.num_of_iters)
        del data

    def __len__(self):
        """For the pytorch dataloader to keep track of all the indices."""
        return self.N-2
    
    def __getitem__(self, idx):
        """What comes out for a given index"""


        if self.all_orientations:
            image_ax = self.data_all['image']['ax'][idx:idx+3]
            label_ax = self.data_all['label']['ax'][idx:idx+3]
            image_sag = self.data_all['image']['sag'][idx:idx+3]
            label_sag = self.data_all['label']['sag'][idx:idx+3]
            image_cor = self.data_all['image']['cor'][idx:idx+3]
            label_cor = self.data_all['label']['cor'][idx:idx+3]

            image = torch.cat((image_ax, image_sag, image_cor), dim = 0)
            label = torch.cat((label_ax, label_sag, label_cor), dim = 0)

            image = einops.rearrange(image, 'c b h w ->(b c) h w')
            label = einops.rearrange(label, 'c b h w ->(b c) h w')

            if 'fazekas' in self.data_all:
                fazekas_label = self.data_all['fazekas']
                fazekas_label = torch.Tensor(list(fazekas_label)*label.shape[0])
                out = {'image': image, 'label': label, 'fazekas': fazekas_label}
            else:
                out = {'image': image, 'label': label}

        else:
            image = self.data_all['image'][idx:idx+3]
            label = self.data_all['label'][idx:idx+3]
            image = einops.rearrange(image, 'c b h w ->(b c) h w')
            label = einops.rearrange(label, 'c b h w ->(b c) h w')
            if 'fazekas' in self.data_all:
                fazekas_label = self.data_all['fazekas']
                out = {'image': image, 'label': label, 'fazekas': fazekas_label}
            else:
                out = {'image': image, 'label': label}

        if self.transform:
            out = self.transform(out)
        
        return out
    
        


def rearrange_data(input:dict) -> dict:
    data = input
    for key in input:
        if key == 'image' or key == 'label':
            reshaped_in = einops.rearrange(input[key], 'b c d h w -> (b w) c h d')
            data[key] = reshaped_in
        else:
            pass
    return data



def make_dataset(patients, path, filename, annotfilename = 'annot.nii.gz', fazekas_subjects = False, fazekas_scores = False):

    labels = []
    volume = []
    fazekas_labels = []
    for patient in patients:
        pat_files_mask = glob.glob(os.path.join(path, patient)+f'/*{annotfilename}')
        pat_files_flair = glob.glob(os.path.join(path, patient)+f'/*{filename}')
        PASS_lock = True
        if len(pat_files_mask) == 0:
            PASS_lock = False
        if len(pat_files_flair) == 0:
            PASS_lock = False
        
        if type(fazekas_subjects) != bool:
            if patient not in fazekas_subjects:
                PASS_lock = False
            else:
                idx_fazekas = np.where(fazekas_subjects == patient)[0][0]
                fazekas_label_value = fazekas_scores[idx_fazekas]
                if fazekas_label_value == -999:
                    PASS_lock = False

        if PASS_lock:
            for file in os.listdir(os.path.join(path, patient)):

                if f'{filename}' in file:
                    volume.append(os.path.join(path, patient, file))

                if f'{annotfilename}' in file:
                    labels.append(os.path.join(path, patient, file))
                    if type(fazekas_subjects) != bool:
                        idx_fazekas = np.where(fazekas_subjects == patient)[0][0]
                        fazekas_labels.append(fazekas_scores[idx_fazekas])

    if type(fazekas_subjects) != bool:
        files = [{"image": image, "label": label, "fazekas": fazekas} for image, label, fazekas in zip(volume, labels, fazekas_labels)]
    else:
        files = [{"image": image, "label": label} for image, label in zip(volume, labels)]
    return files



def sort_subjects_based_on_scanners(all_subjects):
    path_txt = "/mnt/CRAI-NAS/all/martinsr/test_monai/WMHSEG_0.5/dataanalysis/scannerID_2.csv"
    data = pd.read_csv(path_txt, header = 0, error_bad_lines=False, delimiter = '\t',encoding = "ISO-8859-1")
    data = data[['subject-ass', 'ManufacturerModelName']]
    uniques = data.ManufacturerModelName.unique()
    uniques = [x for x in uniques if x not in ['Biograph_mMR', 'Intera', 'Avanto_fit']]
    data = data[data['subject-ass'].isin(all_subjects)]
    all_files_existing = data['subject-ass'].tolist()
    files_not_in_datainfo_file = [x for x in all_subjects if x not in all_files_existing]
    print('Not available in datainfo file:', files_not_in_datainfo_file)
    return data, all_files_existing, uniques


def merge_new_paths(files, datapath):
    for file in files:
        file['image'] = file['image'].split('/')[-2:]
        file['label'] = file['label'].split('/')[-2:]
        file['image'] = datapath+'/'+"/".join(file['image'])
        file['label'] = datapath+'/'+"/".join(file['label'])
    return files

def dataprocesser(config, datapath:str, filename:str = 'Normed_F.nii.gz', annotfilename = 'reduced_an.nii.gz') -> tuple:
    """Adds the data in a tuple of paths to the dataset in the form (train_paths, val_paths)"""

    console.rule("[bold red] Data")
    if config.datasets['presplit'] == False:
        p_test = float(config.datasets['datasplit_test'])
        p_val = float(config.datasets['datasplit_val'])
        console.print('Gathering data...', style="bold cyan")

        all_subjects = [file for file in os.listdir(datapath) if file not in ['README.md','desktop.ini','info']]
        np.random.seed(1)
        console.print('seed set to 1...', style="bold cyan")
        console.print(f'Split setup: test: {p_test}, val: {p_val}', style="bold cyan")
        console.print('Splitting dataset...', style="bold cyan")
        
        scanner_based_list, all_subjects, uniques = sort_subjects_based_on_scanners(all_subjects)
        testdata = []
        console.print('Stratified test data sampling at 10% for each scanner...', style="bold cyan")

        # Do the test data sampling, fraction per machine scanner type
        for uniq in uniques:
            all_from_scanner = scanner_based_list.loc[scanner_based_list['ManufacturerModelName'] == uniq]
            sampled = all_from_scanner.sample(frac=p_test, replace=False, random_state=1)
            console.print(f"{list(sampled['ManufacturerModelName'])[0]} {len(sampled['ManufacturerModelName'].to_list())}", style="bold yellow")
            sampled = sampled['subject-ass'].to_list()
            testdata += sampled

        # testdata = np.random.choice(all_subjects, size=int(N_total*p_test), replace=False)
        all_subjects = [x for x in all_subjects if x not in testdata]
        N_total_after = len(all_subjects)
        valdata = np.random.choice(all_subjects, size=int(np.ceil(N_total_after*p_val)), replace=False)
        traindata = [x for x in all_subjects if x not in valdata]

        for test_example in testdata:
            assert test_example not in valdata, 'Test data found in validation data'
            assert test_example not in traindata, 'Test data found in training data'
    

        train_files = make_dataset(traindata, datapath, filename, annotfilename)
        val_files = make_dataset(valdata, datapath, filename, annotfilename)
        test_files = make_dataset(testdata, datapath, filename, annotfilename)

        N_train = len(train_files)
        N_val = len(val_files)
        N_test  = len(test_files)

        console.print('Dataset splits:', style="bold cyan")
        console.print(f'N Total: {N_train + N_val + N_test}', style="bold cyan")
        console.print(f'N train: {N_train}', style="bold cyan")
        console.print(f'N val: {N_val}', style="bold cyan")
        console.print(f'N test: {N_test}', style="bold cyan")

        with open(f'testdatasplit.txt', 'w') as f:
            f.write(str(test_files))
        with open(f'valdatasplit.txt', 'w') as f:
            f.write(str(val_files))
        with open(f'traindatasplit.txt', 'w') as f:
            f.write(str(train_files))
        del test_files
        console.print('Saved test split to file.', style="bold cyan")
    
    else: 
        console.print(f'Loading presplit data from experiment: {config.datasets["presplit_path"]}', style="bold cyan")
        with open(f'{config.datasets["presplit_path"]}/traindatasplit.txt', 'r') as f:
            train_files = eval(f.read())
        with open(f'{config.datasets["presplit_path"]}/valdatasplit.txt', 'r') as f:
            val_files = eval(f.read())
        with open(f'{config.datasets["presplit_path"]}/testdatasplit.txt', 'r') as f:
            test_files = eval(f.read())
        
        train_files = merge_new_paths(train_files, datapath)
        val_files = merge_new_paths(val_files, datapath)
        test_files = merge_new_paths(test_files, datapath)

        N_train = len(train_files)
        N_val = len(val_files)
        N_test = len(test_files)
        console.print(f'N Total: {N_train+N_val+N_test}', style="bold cyan")
        console.print(f'N train: {N_train}', style="bold cyan")
        console.print(f'N val: {N_val}', style="bold cyan")
        console.print(f'N test: {N_test}', style="bold cyan")
        
        for test_example in test_files:
            assert test_example not in val_files, 'Test data found in validation data'
            assert test_example not in train_files, 'Test data found in training data'
        
        with open(f'testdatasplit.txt', 'w') as f:
            f.write(str(test_files))
        with open(f'valdatasplit.txt', 'w') as f:
            f.write(str(val_files))
        with open(f'traindatasplit.txt', 'w') as f:
            f.write(str(train_files))
        del test_files
        console.print('Saved test split to file.', style="bold cyan")
    
    console.rule("[bold red] Datasetup finished!")

    return train_files, val_files

def dataprocesser_v2() -> tuple:
    """
    Adds the data in a tuple of paths to the dataset in the form (train_paths, val_paths)
    This functions is hardcoded for the local wmh project structure and is only for convience.
    """

    console.rule("[bold red] Data")
    # val_file_path = "/mnt/CRAI-NAS/all/martinsr/NNunet/report/valsplit_3D_13_10_2022.txt"
    # with open(val_file_path, 'r') as f:
    #     val_files_from_text = eval(f.read())

    # val_ids = [x.split("/")[-1] for x in val_files_from_text]
    # val_ids = [x.split("_")[0] for x in val_ids]
    # train_ids = os.listdir("./imagesTr")
    # train_ids = [x.split(".nii.gz")[0] for x in train_ids]
    # # remove val ids from train ids
    # train_ids = [x for x in train_ids if x not in val_ids]

    all_files = glob.glob("./data/imagesTr/*.nii.gz")
    
    train_files = [x for x in all_files]
    val_files = [x.replace("imagesTr", "imagesvalTr") for x in all_files]

    train_labels = [x.replace("imagesTr", "labelsTr") for x in train_files]
    val_labels = [x.replace("imagesTr", "labelsvalTr") for x in val_files]

    N_train = len(train_files)
    N_val = len(val_files)

    console.print(f'N Total: {N_train+N_val}', style="bold cyan")
    console.print(f'N train: {N_train}', style="bold cyan")
    console.print(f'N val: {N_val}', style="bold cyan")

    assert len(train_files) == len(train_labels), "Number of training files and labels do not match"
    assert len(val_files) == len(val_labels), "Number of validation files and labels do not match"

    # save to text file the train and val files
    with open(f'traindatasplit.txt', 'w') as f:
        f.write(str(train_files))
    with open(f'valdatasplit.txt', 'w') as f:
        f.write(str(val_files))


    train = [{"image": image, "label": label} for image, label in zip(train_files, train_labels)]
    val = [{"image": image, "label": label} for image, label in zip(val_files, val_labels)]
    return train, val



def z_standardize_brain(path:str) -> None:
    """Makes a brain segmentation map to z-standardize the volume with the brain intensities. Saves new files to the subject folder."""
    patients = [file for file in os.listdir(path) if file not in ['README.md','desktop.ini','info']]

    for patient in track(patients):
        PASS_lock = True
        pat_files_flair = glob.glob(os.path.join(path, patient)+'/*FLAIR.nii.gz')
        # Normed = glob.glob(os.path.join(path, patient)+'/*Normed_F.nii.gz')
        Normed = []
        if len(pat_files_flair) == 0:
            PASS_lock = False
        if PASS_lock:
            for file in os.listdir(os.path.join(path, patient)):
                if 'FLAIR' in file and len(Normed) == 0:
                    try:
                        flair_path_file = os.path.join(path,patient, file)
                        FLAIR_loaded = nib.load(flair_path_file)
                        FLAIR_data = FLAIR_loaded.get_fdata()
                        FLAIR_data[FLAIR_data < 0] = 0 # Remove negative valued noise


                        FLAIR_data_normalized = NormalizeBrain(FLAIR_data)
                        FLAIR_loaded.set_data_dtype(np.float)
                        newflair = nib.Nifti1Image(FLAIR_data_normalized, affine = FLAIR_loaded.affine, header = FLAIR_loaded.header)
                        nib.save(newflair,  f'{os.path.join(path, patient)}/Normed_F.nii.gz')
                    except Exception as e:
                        print(e)
    return None



def NormalizeBrain(FLAIR:np.array) -> np.array:
    # Z-standardize the volume based on the brain intensity values
    
    brain_mean  = np.mean(FLAIR)
    brain_std = np.std(FLAIR)
    FLAIR = (FLAIR - brain_mean)/brain_std

    return FLAIR


# if __name__ == "__main__":
#     z_standardize_brain("/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/test_sample")
    # dataprocesser("/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/alldata")
    # get_drift()
    # get_histogram_for_matching()
    # get_radiomics_features("/mnt/HDD16TB/martinsr/DatasetWMH211018_v2", 'train')
    # get_radiomics_features("/mnt/HDD16TB/martinsr/annotations_v0.2/train", 'train')
    # get_radiomics_features("/mnt/HDD16TB/martinsr/annotations_v0.2/val", 'val')
