import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.segmentation import slic, join_segmentations
from skimage.color import label2rgb
import nibabel as nib
import os
import glob
import matplotlib.patches as patches
import matplotlib.colors as colors


def colorize_label(labelsliced, label2sliced):
    """
    Make colorized labelmap based on FP, TP and FN
    Returns:
        labelmap_colorized: colorized labelmap
        label_bb: bounding box of lesion
    """

    colortmp = np.zeros((*labelsliced.shape, 3))

    colortmp[:, :, 2] = (1 - labelsliced) * label2sliced # False positive
    colortmp[:, :, 1] = labelsliced * label2sliced # True positive
    colortmp[:, :, 0] = labelsliced * (1 - label2sliced) # False negative

    labelmap_colorized = np.zeros((*labelsliced.shape,))
    idx = np.where(colortmp[:, :, 0] == 1.0)
    labelmap_colorized[idx] = 1
    idx = np.where(colortmp[:, :, 1] == 1.0)
    labelmap_colorized[idx] = 2
    idx = np.where(colortmp[:, :, 2] == 1.0)
    labelmap_colorized[idx] = 3

    return labelmap_colorized


def get_slice_with_largest_lesion(labelmap1, labelmap2, FLAIR_FOR_BACKGROUND = None, axis = 0, include_imgslice = False):
    tot_temp = 0
    if axis == 0:
        for i in range(1, labelmap1.shape[0]):
            tot = np.sum(labelmap1[i, :, :])
            if tot > tot_temp:
                tot_temp = tot
                i_temp = i
        labelsliced = labelmap1[i_temp, :, :]
        labelsliced = np.rot90(labelsliced, axes=(0,1))

        label2sliced = labelmap2[i_temp, :, :]
        label2sliced = np.rot90(label2sliced, axes=(0,1))

        if include_imgslice:
            imgslic = FLAIR_FOR_BACKGROUND[i_temp, :, :]
            imgslic = np.rot90(imgslic, axes=(0,1))
    
    if axis == 1:
        for i in range(1, labelmap1.shape[1]):
            tot = np.sum(labelmap1[:, i, :])
            if tot > tot_temp:
                tot_temp = tot
                i_temp = i
        labelsliced = labelmap1[:, i_temp, :]
        labelsliced = np.rot90(labelsliced, axes=(0,1))

        label2sliced = labelmap2[:, i_temp, :]
        label2sliced = np.rot90(label2sliced, axes=(0,1))

        if include_imgslice:
            imgslic = FLAIR_FOR_BACKGROUND[:, i_temp, :]
            imgslic = np.rot90(imgslic, axes=(0,1))
    
    if axis == 2:
        for i in range(1, labelmap1.shape[2]):
            tot = np.sum(labelmap1[:, :, i])
            if tot > tot_temp:
                tot_temp = tot
                i_temp = i
        labelsliced = labelmap1[:, :, i_temp]
        labelsliced = np.rot90(labelsliced, axes=(0,1))

        label2sliced = labelmap2[:, :, i_temp]
        label2sliced = np.rot90(label2sliced, axes=(0,1))

        if include_imgslice:
            imgslic = FLAIR_FOR_BACKGROUND[:, :, i_temp]
            imgslic = np.rot90(imgslic, axes=(0,1))

    if include_imgslice:
        return labelsliced, label2sliced, imgslic

    return labelsliced, label2sliced



def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)



folder = 'DatasetWMH211018_v2_test_fixed_threshold'
all_subjects = glob.glob("/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/imagesTs/*.nii.gz")

# CRAI-NAS/all/martinsr/test_monai/WMH-Segmentation_Production/console_version/data_brno

if not os.path.exists(f'comparisons_{folder}'):
    os.mkdir(f'comparisons_{folder}')
for subject in all_subjects:
    try:
        pat_id = subject.split('/')[-1]
        print(pat_id)
        # exit()

        FLAIR_FOR_BACKGROUND = nib.load(f'/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/imagesTs/{pat_id}').get_fdata()
        GROUND_TRUTH = nib.load(f'/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/labelsTs/{pat_id}').get_fdata().astype(np.uint8)
        UNET2D_PRED = nib.load(f'/mnt/CRAI-NAS/all/martinsr/test_monai/WMH-Segmentation_Production/console_version/result_test/{pat_id}').get_fdata().astype(np.uint8)

        NNUNET_PRED = nib.load(f'/mnt/CRAI-NAS/all/martinsr/NNunet/results/predictions_epoch=550-dice_mean=77_44_3D_task=15_fold=0_tta_test_fixed_threshold/{pat_id}').get_fdata().astype(np.uint8)
        BAYESIAN_PRED = nib.load(f'/mnt/CRAI-NAS/all/martinsr/test_monai/DatasetWMH211018_v2_newphase/pred/{pat_id}').get_fdata().astype(np.uint8)

        slice_axis = [0, 1, 2]
        fig, ax = plt.subplots(3, 5, figsize=(20, 10))
        for slc in slice_axis:
            labelsliced, labelsliced2 = get_slice_with_largest_lesion(GROUND_TRUTH, UNET2D_PRED, axis = slc)
            labelsliced, labelsliced3 = get_slice_with_largest_lesion(GROUND_TRUTH, NNUNET_PRED, axis = slc)
            labelsliced, labelsliced4, imgslic = get_slice_with_largest_lesion(GROUND_TRUTH, BAYESIAN_PRED, FLAIR_FOR_BACKGROUND, axis = slc, include_imgslice=True)

            color1 = colorize_label(labelsliced, labelsliced2)
            color2 = colorize_label(labelsliced, labelsliced3)
            color3 = colorize_label(labelsliced, labelsliced4)

            label_bb = regionprops(labelsliced)
            bb = label_bb[0].bbox
            rmin, cmin, rmax, cmax = bb
            zoom_marker = patches.Rectangle((cmin-20,rmin-20),cmax-cmin+40,rmax-rmin+40, linewidth=2, edgecolor='r', facecolor='none')
            ax[0, 0].set_title('FLAIR', fontsize=20)
            ax[0, 1].set_title('Zoomed in', fontsize=20)
            ax[0, 2].set_title('Prediction 2.5D', fontsize=20)
            ax[0, 3].set_title('Prediction 3D nn-UNet', fontsize=20)
            ax[0, 4].set_title('Prediction Deep Bayesian', fontsize=20)
            ax[slc, 0].imshow(imgslic, cmap='gray')
            ax[slc, 1].imshow(imgslic[bb[0]-20:bb[2]+20, bb[1]-20:bb[3]+20], cmap='gray', interpolation='none')
            ax[slc, 2].imshow(imgslic[bb[0]-20:bb[2]+20, bb[1]-20:bb[3]+20], cmap='gray', interpolation='none')
            ax[slc, 3].imshow(imgslic[bb[0]-20:bb[2]+20, bb[1]-20:bb[3]+20], cmap='gray', interpolation='none')
            ax[slc, 4].imshow(imgslic[bb[0]-20:bb[2]+20, bb[1]-20:bb[3]+20], cmap='gray', interpolation='none')

            # cmap = mpl.colors.ListedColormap(["black", "red", "yellow", "green"])
            # norm = mpl.colors.BoundaryNorm(np.arange(0, 4), cmap.N) 
            N = 4

            # 1 FN
            # 2 TP
            # 3 FP
            cmap = colors.ListedColormap(['black', 'red', 'green', 'yellow'])

            ax[slc, 2].imshow(color1[bb[0]-20:bb[2]+20, bb[1]-20:bb[3]+20], interpolation='none', alpha = 0.5, cmap = cmap, vmin = 0, vmax = 3)
            ax[slc, 3].imshow(color2[bb[0]-20:bb[2]+20, bb[1]-20:bb[3]+20], interpolation='none', alpha = 0.5, cmap = cmap, vmin = 0, vmax = 3)
            ax[slc, 4].imshow(color3[bb[0]-20:bb[2]+20, bb[1]-20:bb[3]+20], interpolation='none', alpha = 0.5, cmap = cmap, vmin = 0, vmax = 3)
            ax[slc, 0].add_patch(zoom_marker)
            ax[slc, 0].axis('off')
            ax[slc, 1].axis('off')
            ax[slc, 2].axis('off')
            ax[slc, 3].axis('off')
            ax[slc, 4].axis('off')

        plt.tight_layout()
        plt.savefig(f'comparisons_{folder}/{pat_id}.svg', dpi=300)
        plt.savefig(f'comparisons_{folder}/{pat_id}.tif', dpi=300)
        plt.savefig(f'comparisons_{folder}/{pat_id}.jpg', dpi=300)
        plt.close()
    except Exception as e:
        print(e)