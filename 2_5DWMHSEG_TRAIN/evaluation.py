
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

#from torch_receptive_field import receptive_field
import nibabel as nib
from rich.progress import track
import glob
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.patches as patches
from skimage.segmentation import join_segmentations


def lesion_counting(path:str) -> None:
    """Finds the volume and position of lesion in 3D space"""

    num_leasion_plots = 5
    leasion_plot_volume_fp = 400
    leasion_plot_volume_fn = 400
    leasion_plot_volume = 500

    paths = glob.glob(path+'/*')
    lesion_path = path+'/leasion_counting'
    lesion_path_fp = lesion_path+'/lesion_counting_fp'
    lesion_path_fn = lesion_path+'/lesion_counting_fn'
    Path(lesion_path).mkdir(parents=True, exist_ok=True)
    Path(lesion_path_fp).mkdir(parents=True, exist_ok=True)
    Path(lesion_path_fn).mkdir(parents=True, exist_ok=True)
    patients_path_array = []
    

    lesion_storage = {'GT':{'all': [], 'recall_minor': [], 'recall_50': [], 'recall_200': [], 'recall_large': []}}
    

    lesion_volume_count = 0
    lesion_volume_count_fp = 0
    lesion_volume_count_fn = 0
    ###### Stats #####
    # GT
    vol_minor_GT = 0
    recall_minor = 0
    z_depth_minor_GT = 0
    N_minor_GT = 0
    

    vol_50_GT = 0
    recall_50 = 0
    N_50_GT = 0
    z_depth_50_GT = 0


    vol_200_GT = 0
    recall_200 = 0
    N_200_GT = 0
    z_depth_200_GT = 0


    vol_large_GT = 0
    recall_large = 0
    N_large_GT = 0
    z_depth_large_GT = 0


    # PRED
    vol_minor_pred = 0
    z_depth_minor_pred = 0
    N_minor_pred = 0


    vol_50_pred = 0
    N_50_pred = 0
    z_depth_50_pred = 0


    vol_200_pred = 0
    N_200_pred = 0
    z_depth_200_pred = 0


    vol_large_pred = 0
    N_large_pred = 0
    z_depth_large_pred = 0

    FP_minor = 0
    FP_50 = 0
    FP_200 = 0
    FP_large = 0

    FN_minor = 0
    FN_50 = 0
    FN_200 = 0
    FN_large = 0

    TP_minor = 0
    TP_50 = 0
    TP_200 = 0
    TP_large = 0

    # Only find patients
    for path in paths:
        patient_num = re.findall(r'\d+', path)
        if len(patient_num) > 0:
            patients_path_array.append(path)
    for patient_path in track(patients_path_array, description="Iterating patients:"):
        # Grab patient folder path name
        l = str(patient_path)
        l = l.replace('\\', '/')
        l = l.split('/')
        patient_num = l[-1]
        # if config.legacy_dataset:
        #     patient_num = re.findall(r'\d+', patient_path)[0]
        # else:
        #     if len(patient_path) > 0:
        #         l = str(patient_path)
        #         l = l.split('R')
        #         patient_num = 'R'+l[1][:7]
        filenotfoundlock = True

        try:
            FLAIR_path = patient_path+'/FLAIR.nii.gz'
            # pred_path = patient_path+f'/{patient_num}_T1acq_nu_wmh_pred.nii.gz'
            pred_path = patient_path+f'/F_seg_testseg.nii.gz'
            annot_path = patient_path+'/reduced_an.nii.gz'
            lesion_FLAIR = (nib.load(FLAIR_path).get_fdata())
            pred_label = (nib.load(pred_path).get_fdata())# Mask loads in as 0-1
            mask_label = (nib.load(annot_path).get_fdata()) # Mask loads in as 0-1
            
        except Exception as e:
            print(e)
            print(patient_path)
            filenotfoundlock = False
        if filenotfoundlock:
            mask_label[mask_label > 0.89] = 1
            mask_label[mask_label <= 0.89] = 0
            mask_label = mask_label.astype('int')
            pred_label[pred_label >= 0.5] = 1
            pred_label[pred_label < 0.5] = 0
            pred_label = pred_label.astype('int')
            

            mask_label = np.rot90(mask_label)
            pred_label = np.rot90(pred_label)
            lesion_FLAIR = np.rot90(lesion_FLAIR)

            # patient_num = patient_num[0]
            pred_label_ = label(pred_label, connectivity=2)
            mask_label_ = label(mask_label, connectivity=2)
            pred_prop = regionprops(pred_label_)
            mask_prop = regionprops(mask_label_)

            for prop_pred in pred_prop:

                zsize_pred = (prop_pred['bbox'][-1] - prop_pred['bbox'][2]) # Number of depth slices in boundingbox

                header_info = nib.load(FLAIR_path).header
                sx, sy, sz = header_info.get_zooms()
                voxel_size = sx*sy*sz
                
                volume = prop_pred['area']*voxel_size
                rmin, cmin, zmin, rmax, cmax, zmax = prop_pred['bbox']
                coords = prop_pred['coords']
                # mask_flat = mask_label[rmin:rmax, cmin:cmax, zmin:zmax].copy().flatten()
                # pred_flat = pred_label[rmin:rmax, cmin:cmax, zmin:zmax].copy().flatten()
                mask_flat = mask_label.copy()[coords[:, 0], coords[:, 1], coords[:, 2]]
                mask_flat_num_real = mask_flat.sum()

                if mask_flat_num_real != 0:
                    lesion_found = True
                else:
                    lesion_found = False

                # Plot FP lesions in folder
                if not lesion_found:
                    lesion_volume_count_fp += 1
                    if (lesion_volume_count_fp < num_leasion_plots) or (volume >= leasion_plot_volume_fp):
                        volume_path_fp = lesion_path_fp+f'/lesion_vol_{lesion_volume_count_fp}_{volume}_{patient_num}'
                        Path(volume_path_fp).mkdir(parents=True, exist_ok=True)
                        rmin -= 1
                        cmin -= 1
                        for zz, z in enumerate(range(zmin, zmax)):

                            fig, ax = plt.subplots(1, 3)
                            pred_big = pred_label.copy()
                            # Colorize/shade lesion in question
                            pred_big[coords[:, 0], coords[:, 1], coords[:, 2]] = 2
                            pred_big = pred_big.copy()[:, :, z]
                            mask_big = mask_label.copy()[:, :, z]
                            img_big = lesion_FLAIR.copy()[:, :, z]
                            rect1 = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin, linewidth=1, edgecolor='r', facecolor='none')
                            rect2 = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin, linewidth=1, edgecolor='r', facecolor='none')
                            rect3 = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin, linewidth=1, edgecolor='r', facecolor='none')
                            # Add the patch to the Axes

                            ax[0].imshow(img_big, cmap = 'gray')
                            ax[1].imshow(pred_big, cmap = 'YlGn', vmin = 0, vmax = 2)
                            ax[2].imshow(mask_big, cmap = 'OrRd', vmin = 0, vmax = 1)
                            ax[0].set_title('Flair slice')
                            ax[1].set_title('Prediction mask')
                            ax[2].set_title('GT mask')
                            #ax[0].annotate(f"Depth:{zsize_pred}", xy=(cmin+16, rmin-6), xycoords="data", va="center", ha="center", size=2, bbox=dict(boxstyle="square", fc="red", ec="red"))
                            ax[0].add_patch(rect1)
                            ax[1].add_patch(rect2)
                            ax[2].add_patch(rect3)
                            ax[0].axis('off')
                            ax[1].axis('off')
                            ax[2].axis('off')
                            plt.tight_layout()
                            plt.savefig(f'{volume_path_fp}/volslice_{z}.png')
                            plt.close()

                # VOLUME FOR PREDICTION MIGHT BE HIGHER OR LOWER THAN GT VOLUME

                if volume < 10:
                    vol_minor_pred += volume
                    z_depth_minor_pred += zsize_pred
                    N_minor_pred += 1
                

                if 400 > volume >= 10:
                    vol_50_pred += volume
                    z_depth_50_pred += zsize_pred
                    N_50_pred += 1

                
                if 1000 > volume >= 400:
                    vol_200_pred += volume
                    z_depth_200_pred += zsize_pred
                    N_200_pred += 1

                if volume >= 1000:
                    vol_large_pred += volume
                    z_depth_large_pred += zsize_pred
                    N_large_pred += 1

                
                if not lesion_found:
                    if volume < 10:
                        FP_minor += 1

                    if 400 > volume >= 10:
                        FP_50 += 1
                    
                    if 1000 > volume >= 400:
                        FP_200 += 1
                    
                    if volume >= 1000:
                        FP_large += 1
                

            for prop_real in mask_prop:
                zsize = (prop_real['bbox'][-1] - prop_real['bbox'][2]) # Number of depth slices in boundingbox
                rmin, cmin, zmin, rmax, cmax, zmax = prop_real['bbox']
                coords = prop_real['coords']
                # mask_flat = mask_label[rmin:rmax, cmin:cmax, zmin:zmax].copy().flatten()
                # pred_flat = pred_label[rmin:rmax, cmin:cmax, zmin:zmax].copy().flatten()
                mask_flat = mask_label.copy()[coords[:, 0], coords[:, 1], coords[:, 2]]
                pred_flat = pred_label.copy()[coords[:, 0], coords[:, 1], coords[:, 2]]
                recall = recall_score_numpy(pred_flat, mask_flat) # Dice does not make sense since we only grab the positives from GT masks
                volume = prop_real['area']

                lesion_storage['GT']['all'].append([volume, recall])


                pred_flat_num_real = pred_flat.copy().sum() # Check if no lesion are predicted
                if pred_flat_num_real >= volume*0.50:
                    lesion_found = True
                else:
                    lesion_found = False

                # Plot FN lesions in folder
                if not lesion_found:
                    lesion_volume_count_fn += 1
                    if (lesion_volume_count_fn < num_leasion_plots) or (volume >= leasion_plot_volume_fn):
                        volume_path_fn = lesion_path_fn+f'/lesion_vol_{lesion_volume_count_fn}_{volume}_{patient_num}'
                        Path(volume_path_fn).mkdir(parents=True, exist_ok=True)
                        rmin -= 1
                        cmin -= 1
                        for zz, z in enumerate(range(zmin, zmax)):
                            fig, ax = plt.subplots(1, 3)
                            pred_big = pred_label.copy()
                            mask_big = mask_label.copy()
                            # Colorize/shade lesion in question
                            pred_big[coords[:, 0], coords[:, 1], coords[:, 2]] *= 2
                            mask_big[coords[:, 0], coords[:, 1], coords[:, 2]] = 2

                            pred_big = pred_big.copy()[:, :, z]
                            mask_big = mask_big.copy()[:, :, z]
                            img_big = lesion_FLAIR.copy()[:, :, z]
                            rect1 = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin, linewidth=1, edgecolor='r', facecolor='none')
                            rect2 = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin, linewidth=1, edgecolor='r', facecolor='none')
                            rect3 = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin, linewidth=1, edgecolor='r', facecolor='none')
                            # Add the patch to the Axes


                            ax[0].imshow(img_big, cmap = 'gray')
                            ax[1].imshow(pred_big, cmap = 'YlGn', vmin = 0, vmax = 2)
                            ax[2].imshow(mask_big, cmap = 'OrRd', vmin = 0, vmax = 2)
                            ax[0].set_title('Flair slice')
                            ax[1].set_title('Prediction mask')
                            ax[2].set_title('GT mask')
                            #ax[0].annotate(f"Depth:{zsize_pred}", xy=(cmin+16, rmin-6), xycoords="data", va="center", ha="center", size=2, bbox=dict(boxstyle="square", fc="red", ec="red"))
                            ax[0].add_patch(rect1)
                            ax[1].add_patch(rect2)
                            ax[2].add_patch(rect3)
                            ax[0].axis('off')
                            ax[1].axis('off')
                            ax[2].axis('off')

                            plt.tight_layout()
                            plt.savefig(f'{volume_path_fn}/volslice_{z}.png')
                            plt.close()
                


                if volume < 10:
                    vol_minor_GT += volume
                    lesion_storage['GT']['recall_minor'].append(recall)
                    z_depth_minor_GT += zsize
                    N_minor_GT += 1
                    if not lesion_found:
                        FN_minor += 1 # FN
                    else:
                        TP_minor += 1 # TN

                if 400 > volume >= 10:
                    vol_50_GT += volume
                    lesion_storage['GT']['recall_50'].append(recall)
                    z_depth_50_GT += zsize
                    N_50_GT += 1
                    if not lesion_found:
                        FN_50 += 1 # FN
                    else:
                        TP_50 += 1 # TN
                
                if 1000 > volume >= 400:
                    vol_200_GT += volume
                    lesion_storage['GT']['recall_200'].append(recall)
                    z_depth_200_GT += zsize
                    N_200_GT += 1
                    if not lesion_found:
                        FN_200 += 1
                    else:
                        TP_200 += 1 # FN

                if volume >= 1000:
                    vol_large_GT += volume
                    lesion_storage['GT']['recall_large'].append(recall)
                    z_depth_large_GT += zsize
                    N_large_GT += 1
                    if not lesion_found:
                        FN_large += 1
                    else:
                        TP_large += 1 # TN


                if lesion_volume_count < num_leasion_plots:
                    if volume > leasion_plot_volume:
                        lesion_volume_count += 1
                        volume_path = lesion_path+f'/lesion_vol_{lesion_volume_count}'
                        Path(volume_path).mkdir(parents=True, exist_ok=True)
                        rmin -= 1
                        cmin -= 1
                        # xx, yy, zz = prop_real['image'].shape
                        # for ii in range(0, zz):
                        #     fig, ax = plt.subplots(1, 2)
                        #     ax[0].imshow(prop_real['image'][:, :, ii], vmin = 0, vmax = 1)
                        #     ax[1].imshow(mask.copy()[:, :, ii+zmin], vmin = 0, vmax = 1)
                        #     plt.show()

                        
                        #cmin, rmin, zmin, cmax, rmax, zmax = prop_real['bbox']
                        for z in range(zmin, zmax):

                            pred_big = pred_label.copy()
                            mask_big = mask_label.copy()
                            # Colorize/shade lesion in question
                            pred_big[coords[:, 0], coords[:, 1], coords[:, 2]] *= 2
                            mask_big[coords[:, 0], coords[:, 1], coords[:, 2]] = 2

                            pred_big_ = pred_label.copy()[:, :, z]
                            mask_big_ = mask_label.copy()[:, :, z]

                            mask_small = mask_big.copy()[rmin:rmax, cmin:cmax, z]
                            pred_small = pred_big.copy()[rmin:rmax, cmin:cmax, z]
                            img_small = lesion_FLAIR.copy()[rmin:rmax, cmin:cmax, z]
                            img_big = lesion_FLAIR.copy()[:, :, z]

                            pred_big = pred_big.copy()[:, :, z]
                            mask_big = mask_big.copy()[:, :, z]

                            segj_big = join_segmentations(pred_big_, mask_big_)
                            segj_big_overlayed = label2rgb(segj_big, bg_label = 0, colors = ['blue', 'red', 'green'], bg_color = (1, 1, 1))
                            segj_small_overlayed = segj_big_overlayed[rmin:rmax, cmin:cmax, :]
                            segj_big = label2rgb(segj_big, bg_label = 0, colors = ['blue', 'red', 'green'] , bg_color = (1, 1, 1))
                            segj_small = segj_big[rmin:rmax, cmin:cmax, :]


                            fig, ax = plt.subplots(2, 5, figsize = [15, 9])
                            ax[0, 0].imshow(segj_big, vmin = 0, vmax = 1, interpolation = None)
                            ax[0, 2].imshow(mask_big, cmap = 'OrRd', vmin = 0, vmax = 1, interpolation = None)
                            ax[0, 3].imshow(pred_big, cmap = 'YlGn', vmin = 0, vmax = 1, interpolation = None)
                            ax[0, 4].imshow(img_big, cmap = 'gray', interpolation = None)
                            ax[0, 1].imshow(img_big, cmap = 'gray', interpolation = None)
                            ax[0, 1].imshow(segj_big_overlayed, alpha = 0.7, interpolation = None)

                            ax[1, 0].imshow(segj_small, vmin = 0, vmax = 1, interpolation = None)
                            ax[1, 2].imshow(mask_small, cmap = 'OrRd', vmin = 0, vmax = 1, interpolation = None)
                            ax[1, 3].imshow(pred_small, cmap = 'YlGn', vmin = 0, vmax = 1, interpolation = None)
                            ax[1, 4].imshow(img_small, cmap = 'gray', interpolation = None)

                            ax[1, 1].imshow(img_small, cmap = 'gray')
                            ax[1, 1].imshow(segj_small_overlayed, alpha = 0.7, interpolation = None)

                            #ax[0, 0].set_title('Red = Predicted\nBlue = GT\nGreen = Overlap')
                            ax[0, 1].set_title('Overlap')
                            ax[0, 2].set_title('GT mask')
                            ax[0, 3].set_title('Pred mask')
                            ax[0, 4].set_title('Flair')

                            #Create a Rectangle patch
                            rect1 = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin, linewidth=1, edgecolor='r', facecolor='none')
                            rect2 = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin, linewidth=1, edgecolor='r', facecolor='none')
                            ax[0, 0].annotate(f"Depth:{zsize}", xy=(cmin+16, rmin-6), xycoords="data", va="center", ha="center", size=6, bbox=dict(boxstyle="square", fc="red", ec="red"))
                            ax[0, 4].annotate(f"Depth:{zsize}", xy=(cmin+16, rmin-6), xycoords="data", va="center", ha="center", size=6, bbox=dict(boxstyle="square", fc="red", ec="red"))
                            # Add the patch to the Axes

                            ax[0, 0].add_patch(rect1)
                            ax[0, 4].add_patch(rect2)
                            ax[0, 0].axis('off')
                            ax[0, 1].axis('off')
                            ax[0, 2].axis('off')
                            ax[0, 3].axis('off')
                            ax[0, 4].axis('off')

                            ax[1, 0].axis('off')
                            ax[1, 1].axis('off')
                            ax[1, 2].axis('off')
                            ax[1, 3].axis('off')
                            ax[1, 4].axis('off')
                            

                            plt.tight_layout()
                            plt.tight_layout()
                            plt.savefig(f'{volume_path}/volslice_{z}.png')
                            plt.close()
                            #plt.show()

    vol_minor_GT /= N_minor_GT if N_minor_GT > 0 else 1
    recall_minor = np.nanmean(lesion_storage['GT']['recall_minor'])

    vol_50_GT /= N_50_GT if N_50_GT > 0 else 1
    recall_50 = np.nanmean(lesion_storage['GT']['recall_50'])
    
    vol_200_GT /= N_200_GT if N_200_GT > 0 else 1
    recall_200 = np.nanmean(lesion_storage['GT']['recall_200'])
    
    vol_large_GT /= N_large_GT if N_large_GT > 0 else 1
    recall_large = np.nanmean(lesion_storage['GT']['recall_large'])

    print()

    z_depth_large_GT /= N_large_GT if N_large_GT > 0 else 1
    z_depth_200_GT /= N_200_GT if N_200_GT > 0 else 1
    z_depth_50_GT /= N_50_GT if N_50_GT > 0 else 1
    z_depth_minor_GT /= N_minor_GT if N_minor_GT > 0 else 1

                
    vol_minor_pred /= N_minor_pred if N_minor_pred > 0 else 1
    vol_50_pred /= N_50_pred if N_50_pred > 0 else 1
    vol_200_pred /= N_200_pred if N_200_pred > 0 else 1
    vol_large_pred /= N_large_pred if N_large_pred > 0 else 1
    z_depth_large_pred /= N_large_pred if N_large_pred > 0 else 1
    z_depth_200_pred /= N_200_pred if N_200_pred > 0 else 1
    z_depth_50_pred /= N_50_pred if N_50_pred > 0 else 1
    z_depth_minor_pred /= N_minor_pred if N_minor_pred > 0 else 1

    # TP_volume_in_predicted_large /= N_large_pred
    # TP_volume_in_predicted_minor /= N_minor_pred
    # TP_volume_in_predicted_50 /= N_50_pred
    # TP_volume_in_predicted_200 /= N_200_pred

    np.save(lesion_path+'/lesion_data.npy',  lesion_storage)


    with open(lesion_path+'/lesion_evaluation.txt', 'w') as f:
        output = f"GT: \n \
        Avg Vol_minor [0, 10) voxels: {vol_minor_GT}, Total: {N_minor_GT}, Avg recall: {recall_minor}, Avg depth: {z_depth_minor_GT} \n\
        Avg Vol:50 [10, 400) voxels: {vol_50_GT}, Total: {N_50_GT}, Avg recall:{recall_50}, Avg depth: {z_depth_50_GT}\n\
        Avg Vol:200 [400, 1000) voxels: {vol_200_GT}, Total: {N_200_GT}, Avg recall: {recall_200}, Avg depth: {z_depth_200_GT} \n\
        Avg Vol:large [1000, ~) voxels: {vol_large_GT}, Total: {N_large_GT}, Avg recall: {recall_large}, Avg depth: {z_depth_large_GT} \n\
        Pred: \n\
        Avg Vol_minor [0, 10) voxels: {vol_minor_pred}, Total: {TP_minor + FP_minor}, Avg depth: {z_depth_minor_pred}, Total FN: {FN_minor}, Total TP: {TP_minor}, Total FP: {FP_minor} \n\
        Avg Vol:50 [10, 400) voxels: {vol_50_pred}, Total: {TP_50 + FP_50}, Avg depth: {z_depth_50_pred}, Total FN: {FN_50} , Total TP: {TP_50}, Total FP: {FP_50} \n\
        Avg Vol:200 [400, 1000) voxels: {vol_200_pred}, Total: {TP_200 + FP_200}, Avg depth: {z_depth_200_pred}, Total FN: {FN_200} , Total TP: {TP_200}, Total FP: {FP_200} \n\
        Avg Vol:large [1000, ~) voxels: {vol_large_pred}, Total: {TP_large + FP_large}, Avg depth: {z_depth_large_pred}, Total FN: {FN_large}, Total TP: {TP_large}, Total FP: {FP_large} \n "
        f.write(output)


    with open(lesion_path+'/lesion_evaluation_textable.txt', 'w') as f:
        output = f"""
        \\begin{{table}}[H]
        \centering
        \caption{{Prediction lesion metrics for only flair model. 
        The detection of the lesion is not as great as the model with both FLAIR and T1.*TP, FN, and FP are not pixel-wise, but for predicted lesions, It is counted if the predicted lesion is at least 50\% of GT lesion. *Recall is calculated using the pixels inside the lesions bounding box. This is on the validation data.}}
        \\begin{{tabular}}{{l|llll}}
        Lesion Size & {{[}}0, 10) & {{[}}10, 400) & {{[}}400, 1000) & {{[}}1000, $\sim$) \\\ \hline
        \\multicolumn{{1}}{{l|}}{{Avg. Volume [mm^3]}} & {vol_minor_pred:.0f}       & {vol_50_pred:.0f}            & {vol_200_pred:.0f}             & {vol_large_pred:.0f}                   \\\ 
        \\multicolumn{{1}}{{l|}}{{Avg. depth}}  & {z_depth_minor_pred:.0f}   & {z_depth_50_pred:.0f}        & {z_depth_200_pred:.0f}         & {z_depth_large_pred:.0f}                        \\\ 
        \\multicolumn{{1}}{{l|}}{{Total}}       & {TP_minor + FP_minor}    & {TP_50 + FP_50}            & {TP_200 + FP_200}            & {TP_large + FP_large}                         \\\ 
        \\multicolumn{{1}}{{l|}}{{Total TP*}}  & {TP_minor}               & {TP_50}                    & {TP_200}                     & {TP_large}                         \\\ 
        \\multicolumn{{1}}{{l|}}{{Total FN*}}   & {FN_minor}               & {FN_50}                    & {FN_200}                     & {FN_large}                         \\\ 
        \\multicolumn{{1}}{{l|}}{{Total FP*}}   & {FP_minor}               & {FP_50}                    & {FP_200}                     & {FP_large}                        \\\ 
        \\rowcolor[HTML]{{9AFF99}}
        \\multicolumn{{1}}{{l|}}{{\\cellcolor[HTML]{{9AFF99}}Recall*}} & {recall_minor:.2f}             & {recall_50:.2f}              & {recall_200:.2f}          & {recall_large:.2f}                       \\\ 
        \\end{{tabular}}
        \\label{{tab:lesion_PRED_FLAIR_ONLY}}
        \\end{{table}}
        """
        f.write(output)




# def evaluate(path = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/val") -> None:
def evaluate(path = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/alldata") -> None:
    """
    PLAN: 
    1. TEX report
    2. low scores
    3. not finding all groups
    etc.
    """

    print('Starting lesion analysis...')
    lesion_counting(path)




def recall_score_numpy(predicted, mask) -> float:
    """Caluclates the recall score and returns it."""
    predicted = predicted.reshape(-1)
    mask = mask.reshape(-1).astype(int)

    predicted[predicted >= 0.89] = 1
    predicted[predicted < 0.89] = 0
    predicted = predicted.astype(int)

    TP = (predicted * mask).sum()    
    FP = ((1-mask) * predicted).sum()
    FN = (mask * (1-predicted)).sum()

    R = TP/(TP + FN) # Recall
    if (TP + FN) == 0:
        return np.nan
    return R




evaluate()
