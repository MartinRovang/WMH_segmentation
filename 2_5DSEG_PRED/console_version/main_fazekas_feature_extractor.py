
import torch
from torch.utils.data import DataLoader
import monai
from monai.inferers import sliding_window_inference
import einops
import utils
from preproc_funcs import preprocess_trans, post_trans, postprocessing_volume_end
import nibabel as nib
from models.UUnet_model import UUNet
import config
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
import datetime as dt


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




def predict_many() -> None:
    """Main function for prediction"""
    path = config.TARGETFOLDER
    modelname = config.MODEL
    NUM_WORKERS = config.NUM_WORKERS
    WEIGHTPATH = config.WEIGHTPATH
    WEIGHTPATH_FAZEKAS = config.WEIGHTPATH_FAZEKAS

    monai.utils.set_determinism(seed=0, additional_settings=None)
    

    # Volume input
    pred_files = utils.dataprocesser_pred(path)

    # Load fazekas model
    clf = pickle.load(open(WEIGHTPATH_FAZEKAS, 'rb'))

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)


    if modelname == 'RESUNET':
            model = monai.networks.nets.UNet(
            dimensions=2,
            in_channels=3,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout = 0.1,
            norm = 'BATCH',
             ).to(device)


    if modelname == 'UNET':
        from models.network_model_256 import UNet
        model = UNet(in_channels=3, init_features=32, out_channels=1).to(device)
        state = torch.load(os.path.abspath(WEIGHTPATH), map_location=device)
        model.load_state_dict(state)



    if config.MODEL == 'UUNET':
        model = UUNet(in_channels=3, 
                        out_channels=1, 
                        init_features=32).to(device)
        state = torch.load(os.path.abspath(WEIGHTPATH), map_location=device)
        model.load_state_dict(state)




    
    # Start prediction loop
    model.eval()
    model = model.float()
    with torch.no_grad():
        for k, large_val_data in enumerate(track(pred_loader, description = 'Predicting...')):
            size = (nib.load(pred_files[k]['img']).get_fdata()).shape
            img_PROPS = nib.load(pred_files[k]['img'])
            l = str(pred_files[k]['img'])
            l = l.replace('\\', '/')
            l = l.split('/')
            patient_num = l[-2]

            header_img = img_PROPS.header
            sx, sy, sz = header_img.get_zooms()
            volume = sx*sy*sz

    
            assert not volume > 1.3 or not volume < 1, "Resolution of the volume is too high or too low. Please check the resolution of the volume"
            
            orientations = ['sag', 'cor', 'ax']
            # orientations = ['cor']

            for ori_num, ori in enumerate(orientations):

                sample_data_val = utils.Minisampler3slice(large_val_data, orientation = ori) # Make 3 slices channels + batch of slices
                sample_loader_val = DataLoader(sample_data_val, batch_size = 10, num_workers = NUM_WORKERS, shuffle = False, drop_last = False)

                if ori_num > 0:
                    break

                batch_counter = 0
                tot = 0
                output_bottleneck_features = {}
                output_bottleneck_features[patient_num] = np.array([])

                for val_batch_data in sample_loader_val:

                    val_images = val_batch_data["img"]
                    val_images = utils.correct_input(val_images)

                    val_images = val_images.to(device).float()

                    testersys.values_standardized(val_images.cpu().numpy()) # Check if correct intensity

                    _ = model(val_images)
                    if ori_num == 0:
                        batch_counter += 10
                        if size[0]//8 < batch_counter and tot < 50:
                            sampled = model.bottleneckflattened[::2, :, :, :]
                            tot += sampled.shape[0]
                            print(batch_counter, sampled.shape, tot)
                            if tot == sampled.shape[0]:
                                output_bottleneck_features[patient_num] = sampled
                                print(output_bottleneck_features[patient_num].shape)
                            else:
                                print(output_bottleneck_features[patient_num].shape, sampled.shape)
                                output_bottleneck_features[patient_num] = np.concatenate((output_bottleneck_features[patient_num], sampled), axis = 0)
                
                saved_to_path = f'{path}/{patient_num}/bottleneck_features.pkl'
                pickle.dump(output_bottleneck_features, open(saved_to_path, 'wb'))
                print(saved_to_path)





if __name__ == "__main__":

    # Deterministic seeding
    monai.utils.set_determinism(seed=0, additional_settings=None)

    predict_many()
    