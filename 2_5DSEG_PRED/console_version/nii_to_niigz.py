import glob
import os
import shutil
from rich.progress import track
import json
import nibabel as nib
import subprocess


path_to_data= "/mnt/CRAI-NAS/all/martinsr/WMH-Segmentation_Production"
path_to_data_all = glob.glob(path_to_data+'/*') # FLAIR_A, FLAIR_B ...
path_to_data_all= ['data']
for path_folder in path_to_data_all:
    all_subjects = glob.glob(path_folder+'/*') # a103, b104...
    for subject in all_subjects:
        target_image = subject+"/Image.nii"
        subprocess.call(f"gzip -f {target_image}", shell=True)


subprocess.call(f"python3 main_prediction.py", shell=True)