import glob
import shutil
import os

"""MOVE DWI"""

# path_to_folders = glob.glob("/mnt/HDD16TB/martinsr/eva_data/DWI_new/*")

# for folder in path_to_folders:
#     if os.path.isdir(folder):
#         path_to_files = glob.glob(folder + "/*")
#         for file in path_to_files:
#             # if _mask
#             id = file.split("/")[-1].split("_")[0]
#             if file.endswith("_mask.nii.gz"):
#                 shutil.copy(
#                     file,
#                     f"/mnt/CRAI-NAS/all/martinsr/NNunet/data/slagprosjekt/labeslTr/{id}.nii.gz",
#                 )
#             # if _image
#             else:
#                 shutil.copy(
#                     file,
#                     f"/mnt/CRAI-NAS/all/martinsr/NNunet/data/slagprosjekt/imagesTr/{id}.nii.gz",
#                 )

#             # print
#             print(f"copied {file} to {id}")


"""MOVE FLAIR"""

all_subjects = glob.glob("/mnt/CRAI-NAS/all/martinsr/NNunet/data/slagprosjekt/imagesTr/*.nii.gz")


for subject in all_subjects:
    id = subject.split("/")[-1].split(".")[0]
    # if file exist
    if os.path.isfile(f"/mnt/HDD16TB/martinsr/eva_data/bids_output/{id}/ses-study-MR/anat/{id}_ses-study-MR_3D_FLAIR.nii.gz"):
        shutil.copy(
            f"/mnt/HDD16TB/martinsr/eva_data/bids_output/{id}/ses-study-MR/anat/{id}_ses-study-MR_3D_FLAIR.nii.gz",
            f"/mnt/CRAI-NAS/all/martinsr/NNunet/data/slagprosjekt/imagesTr_FLAIR/{id}.nii.gz",
        )
        print(f"copied {id} to {id}")
    else:
        print(f"did not find {id}")

    





# /mnt/HDD16TB/martinsr/eva_data/bids_output + "id/ses-study-MR/anat/*FLAIR.nii.gz")