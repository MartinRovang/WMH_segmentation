from asyncore import write
import SimpleITK as sitk
import glob
import os

path_to_original = glob.glob(
    "/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/Projects/Lymedisease_control/*.nii.gz"
)
# path_to_edited = "/mnt/CRAI-NAS/all/martinsr/NNunet/results/predictions_epoch=550-dice_mean=77_44_3D_task=15_fold=0_tta_lyme_control/"
path_to_edited = "/mnt/CRAI-NAS/all/martinsr/NNunet/data/test_predictions/imagesTs/"
path_to_save = (
    "/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/Projects/temp_pred/"
)
print(path_to_original)
error_list = []

for i in path_to_original:
    try:
        img_orig = sitk.ReadImage(i)
        img_edited = sitk.ReadImage(path_to_edited + i.split("/")[-1])

        # get all files in path_to_save
        files_in_path = os.listdir(path_to_save)
        # check if image is already in path_to_save
        if i.split("/")[-1] in files_in_path:
            print("file already exists")
            continue

        # original to array
        # img_orig_ = sitk.GetArrayFromImage(img_orig)
        # img_edited = sitk.GetArrayFromImage(img_edited)
        # # change the edited array to original array

        # # reshape to same axis as original einops
        # img_edited = einops.rearrange(img_edited, 'z y x -> x z y')

        # change axis to fit original
        img_edited = sitk.GetArrayFromImage(img_edited)
        img_edited = img_edited.transpose(2, 0, 1)
        img_edited = sitk.GetImageFromArray(img_edited)
        print("Original shape: ", img_orig.GetSize())
        print("Edited shape: ", img_edited.GetSize())
        # clip coronal up down
        img_edited = sitk.Flip(img_edited, [False, True, False])
        # # flip sagittal left right
        img_edited = sitk.Flip(img_edited, [True, False, False])

        # same metadata as original
        img_edited.CopyInformation(img_orig)

        # save
        sitk.WriteImage(img_edited, path_to_save + i.split("/")[-1])
        print(path_to_save + i.split("/")[-1])

        # sitk.WriteImage(img_edited, path_to_save + i.split('/')[-1])
        exit()
    except Exception as e:
        with open(
            "/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/Projects/Lymedisease/error.txt",
            "a",
        ) as f:
            print("failed to change axis for: ", i)
            f.write(str(e))
            f.write(f", id: {i.split('/')[-1]}\n")
        error_list.append(i)
        with open(
            "/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/Projects/Lymedisease/error_list.txt",
            "w",
        ) as f:
            f.write(str(error_list))
