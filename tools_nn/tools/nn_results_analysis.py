import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import sys
import matplotlib.colors as colors

def select_subjects_randomly(view, viewlabel, n_slices):
    """
    Selects a random subset of subjects from the list of subjects.
    """
    # Find the smalles dimension of the flair volume and use that
    shape = view.shape
    if shape[0] < shape[1]:
        if shape[0] < shape[2]:
            all_random_slices = np.random.randint(0, shape[0], n_slices)
            return (0, view[all_random_slices, :, :], viewlabel[all_random_slices, :, :])
        else:
            all_random_slices = np.random.randint(0, shape[2], n_slices)
            return (2, view[:, :, all_random_slices], viewlabel[:, :, all_random_slices])
    else:
        if shape[1] < shape[2]:
            all_random_slices = np.random.randint(0, shape[1], n_slices)
            return (1, view[:, all_random_slices, :], viewlabel[:, all_random_slices, :])
        else:
            all_random_slices = np.random.randint(0, shape[2], n_slices)
            return (2, view[:, :, all_random_slices], viewlabel[:, :, all_random_slices])
def generate_mosaic(subj_path, border=1, n_slices=10):
    """
    Given a nifty flair volume plot the center in a mosaic grid with
    nrows and ncols and save them to a file with matplotlib.
    """
    
    # nibabel_img = nib.load(f'/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/Projects/Lymedisease/unedited/{subj_path}')
    nibabel_img = nib.load(f'/mnt/CRAI-NAS/all/martinsr/NNunet/data/test_predictions/imagesTs/{subj_path}')
    img_flair = nibabel_img.get_fdata()
    # img_label = nib.load(f'/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/Projects/Lymedisease/predictions/{subj_path}').get_fdata()
    img_label = nib.load(f'/mnt/CRAI-NAS/all/martinsr/NNunet/results/predictions_epoch=550-dice_mean=77_44_3D_task=15_fold=0_tta_biobank_fixed_threshold/{subj_path}')
    img_label = img_label.get_fdata()
    
    center_idx = img_flair.shape[0] // 2
    center_idy = img_flair.shape[1] // 2
    center_idz = img_flair.shape[2] // 2
    # Slice out border slices from the center
    img_flair_x = img_flair[center_idx - border:center_idx + border, :, :]
    img_flair_y = img_flair[:, center_idy - border:center_idy + border, :]
    img_flair_z = img_flair[:, :, center_idz - border:center_idz + border]
    img_label_x = img_label[center_idx - border:center_idx + border, :, :]
    img_label_y = img_label[:, center_idy - border:center_idy + border, :]
    img_label_z = img_label[:, :, center_idz - border:center_idz + border]
    # Concatenate the slices
    total_wmh = round(np.sum(img_label)*0.001*nibabel_img.header.get_zooms()[0]*nibabel_img.header.get_zooms()[1]*nibabel_img.header.get_zooms()[2],5)
    # calculate the number of ncols and nrows needed based on the border parameter

    img_flair_x = select_subjects_randomly(img_flair_x, img_label_x, n_slices=n_slices)
    img_flair_y = select_subjects_randomly(img_flair_y, img_label_y, n_slices=n_slices)
    img_flair_z = select_subjects_randomly(img_flair_z, img_label_z, n_slices=n_slices)

    # add images to list and labels in their own lists
    img_list = [img_flair_x, img_flair_y, img_flair_z]

    ncols = 6
    nrows = n_slices
    # Plot the mosaic with labels overlayed on top of the flair image and save it
    kk = 3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4))
    for i in range(ncols):
        if i > 2:
            for j in range(nrows):
                idx, img, labl = img_list[i-3]
                if idx == 0:
                    ax[j, i].imshow(np.fliplr(img[j, :, :]), cmap='gray')
                elif idx == 1:
                    ax[j, i].imshow(img[:, j, :], cmap='gray')
                else:
                    ax[j, i].imshow(np.rot90(img[:, :, j], k = kk), cmap='gray')
                ax[j, i].axis('off')
        else:
            for j in range(nrows):
                idx, img, labl = img_list[i]
                cmap = colors.ListedColormap(['black', 'red'])
                if idx == 0:
                    ax[j, i].imshow(np.fliplr(img[j, :, :]), cmap='gray')
                    ax[j, i].imshow(np.fliplr(labl[j, :, :]), interpolation='none', alpha = 0.5, cmap = cmap, vmin = 0, vmax = 1)
                elif idx == 1:
                    ax[j, i].imshow(img[:, j, :], cmap='gray')
                    ax[j, i].imshow(labl[:, j, :], interpolation='none', alpha = 0.5, cmap = cmap, vmin = 0, vmax = 1)
                else:
                    ax[j, i].imshow(np.rot90(img[:, :, j], k = kk), cmap='gray')
                    ax[j, i].imshow(np.rot90(labl[:, :, j], k = kk), interpolation='none', alpha = 0.5, cmap = cmap, vmin = 0, vmax = 1)
                ax[j, i].axis('off')
    
    subj_path = subj_path.replace(".nii.gz", "")
    plt.tight_layout()
    # plt.savefig(f'/mnt/CRAI-NAS/all/martinsr/NNunet/report/projects/lyme/mosaic/{subj_path}_{total_wmh}.jpg')
    plt.savefig(f'/mnt/CRAI-NAS/all/martinsr/NNunet/report/mosaic/{subj_path}_{total_wmh}.jpg')
    plt.close()


# Generate a mosaic for each subject
# for subj_path in os.listdir(f'/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/Projects/Lymedisease/unedited/'):
for subj_path in os.listdir(f'/mnt/CRAI-NAS/all/martinsr/NNunet/data/test_predictions/imagesTs'):
    subj_path = subj_path.split('/')[-1]
    print(subj_path)
    generate_mosaic(subj_path, border = 30)
    print(f'{subj_path} done')