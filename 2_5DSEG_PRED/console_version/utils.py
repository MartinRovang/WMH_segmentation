import glob
import os
import monai
import einops
import torch
import nibabel as nib
import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.text import Text
from monai.transforms import (
    Compose,
    EnsureType,
    CenterSpatialCrop,
    SpatialPad,
)

correct_input = Compose([EnsureType('tensor'), CenterSpatialCrop(roi_size = [-1,256,256]), EnsureType('numpy'), SpatialPad(spatial_size = [-1,256,256], mode = 'minimum'), EnsureType('tensor')])



class Minisampler3slice(monai.data.Dataset):
    """Loads the path of images and masks"""
    def __init__(self, data, transform = False, orientation = 'ax'):
        self.data = data.copy()
        self.transform = transform
        # self.num_of_iters = int(len(self.data['img'])//samples)
        # print(self.num_of_iters)
        for key in data:
            if key == 'img':
                if orientation == 'ax':
                    reshaped_in = einops.rearrange(data[key], 'b c d h w -> (b w) c h d')
                    reshaped_in =  reshaped_in.rot90(k = -2, dims = [-2, -1])
                    self.data[key] = reshaped_in

                if orientation == 'sag':
                    reshaped_in = einops.rearrange(data[key], 'b c d h w -> (b d) c h w')
                    reshaped_in =  reshaped_in.rot90(k = 1, dims = [-2, -1])
                    reshaped_in = reshaped_in.flip(3)
                    self.data[key] = reshaped_in

                if orientation == 'cor':
                    reshaped_in = einops.rearrange(data[key], 'b c d h w -> (b h) c d w')
                    reshaped_in =  reshaped_in.rot90(k = 1, dims = [-2, -1])
                    reshaped_in = reshaped_in.flip(3)
                    self.data[key] = reshaped_in


        self.N = reshaped_in.shape[0]
        del data


    def __len__(self):
        """For the pytorch dataloader to keep track of all the indices."""
        return self.N-1
    
    def __getitem__(self, idx):
        """What comes out for a given index"""

        if idx < self.N-3:
            img = self.data['img'][idx:idx+3]
        else:
            img = self.data['img'][idx-3:idx]
        img = einops.rearrange(img, 'c b h w ->(b c) h w')

        out = {'img': img}

        
        return out


def merge_new_paths(files, datapath):
    """Change path to data when loading from existing datapath file"""
    for file in files:
        file['image'] = file['image'].split('/')[-2:]
        file['label'] = file['label'].split('/')[-2:]
        file['image'] = datapath+'/'+"/".join(file['image'])
        file['label'] = datapath+'/'+"/".join(file['label'])
    return files

def dataprocesser_pred(cfg, path) -> None:
    files = glob.glob(path+"/*.nii.gz")


    # list comprehension dictionary "img": file
    files = [{"img": file} for file in files]
    return files
