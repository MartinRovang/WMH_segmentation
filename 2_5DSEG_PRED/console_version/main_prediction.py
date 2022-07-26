
import torch
from torch.utils.data import DataLoader
import monai
from monai.inferers import sliding_window_inference
import einops
import utils
from preproc_funcs import preprocess_trans, post_trans, postprocessing_volume_end
import nibabel as nib
import testersys
import numpy as np
import SimpleITK as sitk
from skimage.morphology import diameter_opening
import os
from monai.transforms.croppad.batch import PadListDataCollate
import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table


def orientation_fix(aggregated_volume, ori, post_trans_volume_adjust, size):
    aggregated_volume = aggregated_volume.squeeze(0)[:, :, :]

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
    

    for j, index in enumerate(size):
        if index % 2 != 0:
            aggregated_volume = np.roll(aggregated_volume, 1, axis=j)

    return aggregated_volume



@hydra.main(config_path=".", config_name="config")
def predict_many(cfg: DictConfig) -> None:
    """Main function for prediction"""
    path = cfg.data.path
    NUM_WORKERS = cfg.data.workers
    WEIGHTPATH = cfg.model.weights
    WEIGHTPATH_FAZEKAS = cfg.model.fazekas_weight
    monai.utils.set_determinism(seed=0, additional_settings=None)
    print('Predicting: ', path)
    # Volume input
    pred_files = utils.dataprocesser_pred(cfg, path)

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
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)


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
            l = str(pred_files[k]['img'])
            l = l.replace('\\', '/')
            l = l.split('/')
            patient_num = l[-2]
            predictions = []

            header_img = img_PROPS.header
            sx, sy, sz = header_img.get_zooms()
            volume = sx*sy*sz

    
            assert not volume > 1.3 or not volume < 1, "Resolution of the volume is too high or too low. Please check the resolution of the volume"
            
            orientations = ['sag', 'cor', 'ax']

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
            print('running')
            STAPLE_seg = diameter_opening(STAPLE_seg, diameter_threshold = 5, connectivity = 2)
            print('ended')
            STAPLE_seg = STAPLE_seg.astype('int')

            table.add_row(f"{patient_num}",f"{round(STAPLE_seg.sum()*volume*0.001, 3)} ")
            console.print(table)

            predicted_mask = nib.Nifti1Image(STAPLE_seg, affine = img_PROPS.affine, header = img_PROPS.header)


            path_to_current_subject = pred_files[k]['img'].split('/')[:-1]
            path_to_current_subject = "/".join(path_to_current_subject)


            saved_to_path = f'{path_to_current_subject}/{cfg.data.outputmaskname}.nii.gz'
            nib.save(predicted_mask,  saved_to_path)
            print(f'Prediction saved to: {saved_to_path}')

    

if __name__ == "__main__":

    # Deterministic seeding
    monai.utils.set_determinism(seed=0, additional_settings=None)
    predict_many()
    
