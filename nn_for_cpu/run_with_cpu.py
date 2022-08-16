
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet
import torch
import nibabel as nib
import numpy as np
from skimage.transform import resize
import os
import click
import time

model = DynUNet(
            3,
            1,
            2,
            [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            filters=None,
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            deep_supervision=False,
            deep_supr_num=2,
            res_block=False,
            trans_bias=True,)


def flip(data, axis):
    return torch.flip(data, dims=axis)

def do_inference(img):
    with torch.no_grad():
        pred = sliding_window_inference(
                inputs=img,
                roi_size=[128,128,128],
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
                mode='gaussian',
            )
        pred = torch.nn.functional.softmax(pred, dim=1)
    return pred

def tta_inference(img):
    tta_flips = [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
    pred = do_inference(img)
    for flip_idx in tta_flips:
        pred += flip(do_inference(flip(img, flip_idx)), flip_idx)
    pred /= len(tta_flips) + 1
    return pred

@click.command(help="Inference on a single image.")
@click.option("--tta", help="Infer with test time augmentation", is_flag=True, default=False)
@click.option("--gpu", help="Infer with gpu", is_flag=True, default=False)
def inference(**kwargs):
    # start timer
    start = time.time()
    print("Loading model...")
    global model
    model_state = torch.load("/results/checkpoints/epoch=285-dice_mean=79.15.ckpt")["state_dict"]
    state_dict = {} 

    for k, v in model_state.items():
        state_dict[k.replace("model.", "")] = v
    model.load_state_dict(state_dict)

    if kwargs["gpu"]:
        model = model.to("cuda").half()
    else:
        model = model.to("cpu")
    model.eval()
    # get all nifty files in data folder
    files = os.listdir("/data/")
    files = [f for f in files if f.endswith(".nii.gz")]
    print("Segmenting...")
    for file in files:
        save_path_name = file.split("/")[-1]
        data_nib = nib.load(f"/data/{file}")
        x,y,z = data_nib.header.get_zooms()
        data = data_nib.get_fdata()

        # Scale to [1,1,1] based on the resolution of the data
        ratio = np.array([x/1,y/1,z/1])

        data_rez = resize(data, (np.array(ratio*data.shape, dtype=int)), order =3, mode = "edge", cval=0, clip=True, anti_aliasing=False)


        data_rez = data_rez - np.mean(data_rez)
        data_rez = data_rez / np.std(data_rez)
        data_rez = torch.Tensor(data_rez).unsqueeze(0).unsqueeze(0)
        if kwargs["gpu"]:
            data_rez = data_rez.to("cuda:0").half()
        else:
            data_rez = data_rez.to("cpu")
        if kwargs["tta"]:
            pred = tta_inference(data_rez)
        else:
            pred = do_inference(data_rez)
        pred = pred.cpu().numpy()
        pred = pred[0, 1, :, :, :]
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        # save pred nifti
        pred = resize(pred, (data.shape[0], data.shape[1], data.shape[2]), order =0, mode = "edge", cval=0, clip=True, anti_aliasing=False)

        nib.save(nib.Nifti1Image(pred, data_nib.affine,data_nib.header), f"/results/predictions_epoch=epoch=285-dice_mean=79.15_task=01_fold=0_tta/{save_path_name}")

    # end timer
    end = time.time()
    print(f"Time elapsed: {end - start}")
if __name__ == '__main__':
    inference()