
import torch
from torch.utils.data import DataLoader
import monai
from monai.inferers import sliding_window_inference
import einops
import utils
from preproc_funcs import preprocess_trans, post_trans, postprocessing_volume_end
import nibabel as nib
from models.UUnet_model import UUNet
from rich.progress import track
import testersys
import numpy as np
import SimpleITK as sitk
from skimage.morphology import diameter_opening
import os
from monai.transforms.croppad.batch import PadListDataCollate
import pickle
from skimage.measure import label, regionprops
from sklearn.ensemble import AdaBoostClassifier
import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

def predict_fazekas(path_target:str, clf:AdaBoostClassifier) -> None:
    """Predicts the fazekas score for a given target."""
    sx, sy, sz = nib.load(path_target).header.get_zooms()
    mask_label = nib.load(path_target).get_fdata() # Mask loads in as 0-1
    mask_label[mask_label > 0.89] = 1
    mask_label[mask_label <= 0.89] = 0
    lesions_cleaned = []
    lesions_centroid_mm_cleaned = []
    to_output = {'labelpath': [], 'label': [], 'voxel_res': [], 'lesions': [], 'lesion_hist': [], 'lesions_centroid_mm': [], 'lesions_centroid_mm_cleaned': [], 'lesions_cleaned': [], 'lesion_hist_cleaned': []}

    vol = sx*sy*sz
    mask_label_cleaned = mask_label.astype('int').astype('bool')
    # mask_label_cleaned = diameter_opening(mask_label_cleaned, diameter_threshold = 5, connectivity = 2)
    # mask_label_cleaned = mask_label_cleaned.astype('int')
    mask_label_cleaned_ = label(mask_label_cleaned, connectivity=2)
    mask_label_cleaned_prop = regionprops(mask_label_cleaned_)
    # # Center of the volume
    center = np.array([mask_label.shape[0]/2, mask_label.shape[1]/2, mask_label.shape[2]/2])

    for prop_pred in mask_label_cleaned_prop:
        volume = prop_pred['area'].copy()*vol
        cx, cy, cz = prop_pred.centroid
        ccx = round((cx - center[0]) * sx, 2)
        ccy = round((cy - center[1]) * sy, 2)
        ccz = round((cz - center[2]) * sz, 2)
        lesions_cleaned.append(volume)
        lesions_centroid_mm_cleaned.append([ccx, ccy, ccz])

    # Cleaned
    lesions_cleaned = np.array(lesions_cleaned)
    to_output['lesions_cleaned'] = lesions_cleaned.copy()
    to_output['lesions_centroid_mm_cleaned'] = np.array(lesions_centroid_mm_cleaned)
    
    sum_of_lesions = np.sum(to_output['lesions_cleaned'])
    mean = np.mean(to_output['lesions_cleaned'])
    std = np.std(to_output['lesions_cleaned'])
    long = len(to_output['lesions_cleaned'])
    centroids = np.array(to_output['lesions_centroid_mm_cleaned'])
    centroidsx = np.mean(centroids[:, 0])
    centroidsy = np.mean(centroids[:, 1])
    centroidsz = np.mean(centroids[:, 2])
    centroidsx_std = np.std(centroids[:, 0])
    centroidsy_std = np.std(centroids[:, 1])
    centroidsz_std = np.std(centroids[:, 2])
    X = np.array([sum_of_lesions, mean, std, long, centroidsx, centroidsy, centroidsz, centroidsx_std, centroidsy_std, centroidsz_std])[None, :]
    predictions = clf.predict(X)[0]
    print('Prediction Fazekas: ', path_target)
    fazekas_output_path = path_target.replace('F_seg.nii.gz', 'fazekas_scale.txt')
    with open(fazekas_output_path, 'w') as f:
        f.write(f"Fazekas: {predictions}")


def orientation_fix(aggregated_volume, ori, post_trans_volume_adjust, size):
    aggregated_volume = aggregated_volume.squeeze(0)[:, :, :]
    new_size = aggregated_volume.shape
    if ori == 'ax':
        aggregated_volume = torch.rot90(aggregated_volume, k = 2, dims = (-2, -1)) # Inverse rotation transform
        aggregated_volume = einops.rearrange(aggregated_volume, 'w h d ->  d h w')#.unsqueeze(0)# Inverse rearrange
        aggregated_volume = post_trans_volume_adjust(aggregated_volume).squeeze(0)

    if ori == 'sag':
        aggregated_volume = aggregated_volume.flip(2)
        aggregated_volume = torch.rot90(aggregated_volume, k = -1, dims = (-2, -1)) # Inverse rotation transform
        aggregated_volume = post_trans_volume_adjust(aggregated_volume).squeeze(0)

    if ori == 'cor':
        aggregated_volume = aggregated_volume.flip(2)
        aggregated_volume = torch.rot90(aggregated_volume, k = -1, dims = (-2, -1)) # Inverse rotation transform
        aggregated_volume = einops.rearrange(aggregated_volume, 'h d w ->  d h w')#.unsqueeze(0)# Inverse rearrange
        aggregated_volume = post_trans_volume_adjust(aggregated_volume).squeeze(0)
    
    #change_in_dims = (size[0] - new_size[0], size[1] - new_size[1], size[2] - new_size[2])

    for j, index in enumerate(size):
        if index % 2 != 0:
            aggregated_volume = np.roll(aggregated_volume, 1, axis=j)

    return aggregated_volume



@hydra.main(config_path=".", config_name="config")
def predict_many(cfg: DictConfig) -> None:
    """Main function for prediction"""
    path = cfg.data.path
    modelname = cfg.model.modeltype
    NUM_WORKERS = cfg.data.workers
    WEIGHTPATH = cfg.model.weights
    monai.utils.set_determinism(seed=0, additional_settings=None)
    print('Predicting: ', path)
    # Volume input
    pred_files = utils.dataprocesser_pred(cfg, path)

    # Load fazekas model
    #clf = pickle.load(open(WEIGHTPATH_FAZEKAS, 'rb'))

    pred_ds = monai.data.Dataset(data = pred_files, transform = preprocess_trans)

    pred_loader = DataLoader(
        pred_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=PadListDataCollate(method = 'symmetric', mode = 'minimum'),
        drop_last = False,
        pin_memory=torch.cuda.is_available(),
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)


    # if modelname == 'UNET':
    from models.network_model_256 import UNet
    model = UNet(in_channels=3, init_features=32, out_channels=1).to(device)
    state = torch.load(os.path.abspath(WEIGHTPATH), map_location=device)
    model.load_state_dict(state)

    console = Console()
    # Start prediction loop
    model.eval()
    model = model.float()
    table = Table(title="Prediction")
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Total WMH (mL)", justify="right", style="cyan", no_wrap=True)
    console.print("[bold cyan] Predicting... ")
    with torch.no_grad():
        for k, large_val_data in enumerate(pred_loader):
            size = (nib.load(pred_files[k]['img']).get_fdata()).shape
            img_PROPS = nib.load(pred_files[k]['img'])
            patient_id = str(pred_files[k]['img'])
            patient_id = patient_id.replace('\\', '/')
            patient_id = patient_id.split('/')
            patient_id = patient_id[-1]
            predictions = []

            header_img = img_PROPS.header
            sx, sy, sz = header_img.get_zooms()
            volume = sx*sy*sz

    
            assert not volume > 1.3 or not volume < 1, "Resolution of the volume is too high or too low. Please check the resolution of the volume"
            
            orientations = ['sag', 'cor', 'ax']
            # orientations = ['cor']

            for ori in orientations:

                sample_data_val = utils.Minisampler3slice(large_val_data, orientation = ori) # Make 3 slices channels + batch of slices
                sample_loader_val = DataLoader(sample_data_val, batch_size = 400, num_workers = NUM_WORKERS, shuffle = False, drop_last = False)

                lock = True
                for val_batch_data in sample_loader_val:

                    val_images = val_batch_data["img"]
                    val_images = utils.correct_input(val_images)

                    val_images = val_images.to(device).float()

                    testersys.values_standardized(val_images.cpu().numpy()) # Check if correct intensity

                    roi_size = (-1, -1)
                    sw_batch_size = 12
                    padding_mode = 'constant'

                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model, padding_mode = padding_mode)
                    # featuremaps.append(model.bottleneckflattened)
                    # dist_sample = np.array(torch.sigmoid(val_outputs.flatten()).cpu().numpy())
                    # output_distribution = np.append(output_distribution, dist_sample)

                    if lock:
                        lock = False
                        aggregated_volume = post_trans(val_outputs[0]).cpu()
                        aggregated_volume_extra = post_trans(val_outputs[0]).cpu()
                        aggregated_volume =  torch.cat((aggregated_volume, aggregated_volume), dim = 0)
                        for j, slice in enumerate(val_outputs):
                            if j > 0:
                                aggregated_volume = torch.cat((aggregated_volume, post_trans(slice).cpu()))
                    else:
                        for slice in val_outputs:
                            aggregated_volume = torch.cat((aggregated_volume, post_trans(slice).cpu()))

                aggregated_volume =  torch.cat((aggregated_volume, aggregated_volume_extra), dim = 0)
                post_trans_volume_adjust = postprocessing_volume_end(size)

                pred = orientation_fix(aggregated_volume, ori, post_trans_volume_adjust, size)
                predictions.append(pred)

            if len(orientations) > 1:
                seg_stack = []
                for pred in predictions:
                    sitk_pred = sitk.GetImageFromArray(pred.astype(np.int16))
                    seg_stack.append(sitk_pred) # STAPLE requires we cast into int16 arrays

                STAPLE_seg_sitk = sitk.STAPLE(seg_stack, 1.0 ) # 1.0 specifies the foreground value
                # convert back to numpy array
                STAPLE_seg = sitk.GetArrayFromImage(STAPLE_seg_sitk)

                # Fix crop/pad one pixel/slice shift issue
            else:
                STAPLE_seg = predictions[0]
            STAPLE_seg[STAPLE_seg >= 0.5] = 1
            STAPLE_seg[STAPLE_seg < 0.5] = 0
            
            STAPLE_seg = STAPLE_seg.astype('int').astype('bool')
            # print('running')
            # STAPLE_seg = diameter_opening(STAPLE_seg, diameter_threshold = 5, connectivity = 2)
            # STAPLE_seg = remove_small_objects(STAPLE_seg, min_size = 5, connectivity=2)
            # print('ended')
            STAPLE_seg = STAPLE_seg.astype('int')

            table.add_row(f"{patient_id}",f"{round(STAPLE_seg.sum()*volume*0.001, 3)} ")
            console.print(table)

            predicted_mask = nib.Nifti1Image(STAPLE_seg, affine = img_PROPS.affine, header = img_PROPS.header)


            saved_to_path = f"/mnt/CRAI-NAS/all/martinsr/test_monai/WMH-Segmentation_Production/console_version/result"
            nib.save(predicted_mask,  saved_to_path+f"/{patient_id}")
            print(f'Prediction saved to: {saved_to_path+f"/{patient_id}"}')

            # predict_fazekas(saved_to_path, clf)
            
            
            # path_annot = f'{path}/{patient_num}/annot.nii.gz'
            # if os.path.exists(path_annot):
            #     img_PROPS_data = nib.load(path_annot).get_fdata()
            #     img_PROPS_data[img_PROPS_data > 0] = 1
            #     img_PROPS_data = img_PROPS_data.astype('bool')
            #     img_PROPS_data = diameter_opening(img_PROPS_data, diameter_threshold  = 5, connectivity = 2)
            #     img_PROPS_data = img_PROPS_data.astype('int')
            #     predicted_mask = nib.Nifti1Image(img_PROPS_data, affine = img_PROPS.affine, header = img_PROPS.header)
            #     nib.save(predicted_mask,  f'{path}/{patient_num}/reduced_an.nii.gz')

                


            # output_distribution = output_distribution.flatten()
            # idx = np.where(output_distribution > 0.0001)
            # output_distribution_params['dist'] = np.append(output_distribution_params['dist'], output_distribution[idx])
            # output_distribution_params['means'].append(np.mean(output_distribution[idx]))
            # output_distribution_params['stds'].append(np.std(output_distribution[idx]))

        
    # plt.hist(output_distribution_params['dist'], bins = 100)
    # plt.show()
        



if __name__ == "__main__":

    # Deterministic seeding
    monai.utils.set_determinism(seed=0, additional_settings=None)
    predict_many()
    
