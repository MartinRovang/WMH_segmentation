import glob
import os
import shutil

path_target = (
    "/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/Projects/Lymedisease_control"
)
path_original = "/mnt/CRAI-NAS/all/atle/BOR_SEG/BorrSci_FLAIR_500_600/FLAIR_F/*"


for i in glob.glob(path_original):
    if os.path.isdir(i):
        patient_folder = glob.glob(i + "/*")
        for file in patient_folder:
            if file.split("/")[-1] == "Image.nii.gz":
                id_patient = file.split("/")[-2]
                shutil.copy(file, path_target + "/" + id_patient + ".nii.gz")
                print("Copied: ", file)
