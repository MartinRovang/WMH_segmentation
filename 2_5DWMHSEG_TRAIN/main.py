import logging
import os
import sys
import torch
from torch.utils.data import DataLoader
import monai
from monai.data import pad_list_data_collate
from monai.inferers import sliding_window_inference
from transform_functions import train_transforms, val_transforms, post_trans, augmentation
import utils
import matplotlib.pyplot as plt
import numpy as np
from lossfunctions.lossfuncs import FocalTverskyLoss
import hydra
from omegaconf import DictConfig
from metrics import FB_score, dice_score
from models.couple import CopleNet
from models.UUnet_model_new import UUNet
import skimage
from rich.progress import track
import testersys
from monai.transforms.croppad.batch import PadListDataCollate
import datetime
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, fbeta_score
from skimage import segmentation
from skimage import color
import torch.nn as nn
from skimage.measure import label
import matplotlib
from matplotlib.colors import ListedColormap
from rich.console import Console

def mixup(img, labels, alpha=0.4):
    """
    Mixup data augmentation
    """
    # Together they form 1/N batch mixed together
    m = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    lam = m.sample()  # Gamma distributed with concentration=1 and rate=1

    img1 = img[0:img.shape[0] // 2]
    img2 = img[img.shape[0] // 2:]
    lab1 = labels[0:labels.shape[0] // 2]
    lab2 = labels[labels.shape[0] // 2:]

    imgs1 = len(img1)
    imgs2 = len(img2)
    if imgs1 > imgs2:
        img1 = img1[:imgs2]
    if imgs2 > imgs1:
        img2 = img2[:imgs1]

    assert len(img1) == len(lab1)
    assert len(img2) == len(lab2)
    assert len(img1) == len(img2)
    # idx = np.random.permutation(len(ds_one))
    # ds_one = ds_one[idx]
    # ds_two = ds_two[idx]
    # lam = np.random.beta(alpha, alpha)
    img_out = lam * img1 + (1 - lam) * img2
    lab_out = lam * lab1 + (1 - lam) * lab2
    return img_out, lab_out


def make_onehot(fazekas, labels):
    fazekas = fazekas + 1 # 0 = background, 1 = fazekas 0, 2 = fazekas 1, ....
    fazekas = fazekas.view(-1)[:, None, None, None]
    testersys.correct_fazekas_labels(fazekas.cpu().numpy())
    labels = labels * fazekas
    for j, batch in enumerate(labels):
        idx_0 = torch.where(batch == 0)
        idx_1 = torch.where(batch == 1)
        idx_2 = torch.where(batch == 2)
        idx_3 = torch.where(batch == 3)
        idx_4 = torch.where(batch == 4)
        for i in range(0, 5):
            batch_onehot = torch.zeros(batch.shape, dtype = torch.float)
            if i == 0:
                batch_onehot[idx_0] = 1.0
                batch_onehot_final = batch_onehot
            if i == 1:
                batch_onehot[idx_1] = 1.0
                batch_onehot_final = torch.cat((batch_onehot_final, batch_onehot), dim = 0)
            if i == 2:
                batch_onehot[idx_2] = 1.0
                batch_onehot_final = torch.cat((batch_onehot_final, batch_onehot), dim = 0)
            if i == 3:
                batch_onehot[idx_3] = 1.0
                batch_onehot_final = torch.cat((batch_onehot_final, batch_onehot), dim = 0)
            if i == 4:
                batch_onehot[idx_4] = 1.0
                batch_onehot_final = torch.cat((batch_onehot_final, batch_onehot), dim = 0)
        if j == 0:
            batch_final = batch_onehot_final[None, :, :, :]
        else:
            batch_final = torch.cat((batch_final, batch_onehot_final[None, :, :, :]), dim = 0)
    labels = batch_final
    return labels


def fbeta_score(inputs, targets, beta =  2, smoothing = 1):

    #flatten label and prediction tensors
    inputs = inputs.flatten()
    targets = targets.flatten()
    #True Positives, False Positives & False Negatives
    TP = (inputs * targets).sum()    
    FP = ((1-targets) * inputs).sum()
    FN = (targets * (1-inputs)).sum()

    fbeta_top = (1+beta**2)*TP + smoothing
    fbeta_bot = ((1+beta**2) * TP + beta**2*FN + FP)  + smoothing
    fbeta_score = fbeta_top/fbeta_bot

    return fbeta_score


def get_all_metrics(prediction, target, beta = 2):
    """
    Find f1 and fbeta scores of the flattened input.
    """
    output = {'f1': [], 'fbeta': []}
    # output['f1'] = f1_score(target.flatten(), prediction.flatten(), average='binary', zero_division=1)
    # output['fbeta'] = fbeta_score(target.flatten(), prediction.flatten(), average='binary', zero_division=1, beta=beta)
    output['f1'] = fbeta_score(prediction, target, beta =  1, smoothing = 1)
    output['fbeta'] = fbeta_score(prediction, target, beta =  2, smoothing = 1)
    return output


@hydra.main(config_path="conf", config_name="config")
def main(config : DictConfig):
    """Main training and validation"""
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    console = Console()

    
    # train_files, val_files = utils.dataprocesser(config.datasets['trainfolder'], config.datasets['valfolder'], include_fazekas = config.structure['include_fazekas'], filename = 'Normed_F.nii.gz')

    # train_files, val_files = utils.dataprocesser(config, config.datasets['datafolder'], filename = 'Normed_F.nii.gz')

    train_files, val_files = utils.dataprocesser_v2()

    
    loss_function = FocalTverskyLoss()
    # loss_function = monai.losses.DiceLoss(include_background=False, to_onehot_y=False, sigmoid=True)
    

    device = torch.device(config.hardware['gpu'] if torch.cuda.is_available() else "cpu")

    if config.structure['model'] == 'COPE':
        model = CopleNet(spatial_dims = 2,
                    in_channels = 3,
                    out_channels = 1,
                    feature_channels =(32, 64, 128, 256, 512),
                    dropout =(0.0, 0.0, 0.3, 0.4, 0.5),
                    bilinear = True,
                    ).to(device)

    if config.structure['model'] == 'RESUNET':
        model = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=3,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout = 0.1,
        norm = 'BATCH',
            ).to(device)

    if config.structure['model'] == 'UUNET':
        model = UUNet(in_channels=3, 
                        out_channels=1, 
                        init_features=32).to(device)

    if config.structure['model'] == 'UNET':
        from models.network_model_256 import UNet
        # from models.ConvneXt_segmentation import UNetConvNext
        model = UNet(in_channels=3, init_features = 32).to(device)
        # model = UNetConvNext().to(device)
        # model = torch.nn.DataParallel(model)

        # Calculate the number of trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print the number of trainable parameters in million
        print('Trainable parameters: %.3fM' % (trainable_params / 1e6))


    # Used to to train last layer only
    # if config.structure['include_fazekas']:
    #     # Only train last layer
    #     for j, i in enumerate(model.parameters()):
    #         if j < len(list(model.parameters()))-1:
    #             i.requires_grad_(False)
        


    optimizer = torch.optim.Adam(model.parameters(), lr=config.hyperparams.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config.hyperparams.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)


    if config.structure['loadweights']:
        state = torch.load(config.structure['weights'], map_location=device)
        model.load_state_dict(state)
        print('Loading weights...')
    if config.structure['loadoptim']:
        print('load optimweights...')
        optimizer.load_state_dict(torch.load(config.structure['optimweights'], map_location = device))
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.scheduelerparams.factor, patience=config.scheduelerparams.patience, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # create a training data loader
    train_ds = monai.data.Dataset(data = train_files, transform = train_transforms)


    
    # for i in train_files:
    #     print(i['image'])
    #     exit()

    train_loader = DataLoader(
        train_ds,
        batch_size = config.structure['train_patient_batch_size'],
        shuffle = True,
        num_workers = 0,
        collate_fn = PadListDataCollate(method = 'symmetric', mode = 'minimum'),
        drop_last = False,
        pin_memory = torch.cuda.is_available(),
    )

    val_ds = monai.data.Dataset(data = val_files, transform = val_transforms)



    val_loader = DataLoader(
        val_ds,
        batch_size=config.structure['validation_patient_batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=PadListDataCollate(method = 'symmetric', mode = 'minimum'),
        drop_last = False,
        pin_memory=torch.cuda.is_available(),
    )

    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    loss_array_val = []
    # writer = SummaryWriter()
    # m = torch.distributions.gamma.Gamma(torch.tensor([0.4]), torch.tensor([0.4]))


    for epoch in range(config.hyperparams.epochs):
        console.rule("[bold red] Training")
        print(f"epoch {epoch + 1}/{config.hyperparams.epochs}")
        model.train()
        epoch_len = len(train_ds)
        epoch_loss = 0
        step = 0
        fast_start_iter = 1
        a = datetime.datetime.now().replace(microsecond=0)
        for fast_start_iter, large_batch in enumerate(track(train_loader, description = f"Training...")):

            # if epoch < config.hyperparams.fast_start_epoch and fast_start_iter > config.hyperparams.fast_start_batch_iter_value:
            #     print('Fast broke train...')
            #     break


            sample_data = utils.Minisampler3slice(large_batch, transform = augmentation, all_orientations = config.structure['all_orientations'])
            sample_loader = DataLoader(sample_data, batch_size = config.structure['mini_batch_size'], num_workers = 0, shuffle = True, drop_last = False, pin_memory = torch.cuda.is_available())

            for i, batch_data_ax in enumerate(sample_loader):
                step += 1
        
                inputs = batch_data_ax['image']
                labels = batch_data_ax['label']

                testersys.values_standardized(inputs.cpu().numpy()) # Check if correct intensity
                testersys.labels_binary(labels.cpu().numpy()) # Check if labels are binary

                if config.structure['all_orientations']:
                    inputs_ax = inputs[:, 0:3, :, :]
                    inputs_sag = inputs[:, 3:6, :, :]
                    inputs_cor = inputs[:, 6:9, :, :]
                    label_ax = labels[:, 0:3, :, :]
                    label_sag = labels[:, 3:6, :, :]
                    label_cor = labels[:, 6:9, :, :]

                    inputs = torch.cat((inputs_ax, inputs_sag, inputs_cor), dim = 0)
                    labels = torch.cat((label_ax, label_sag, label_cor), dim = 0)

                    # clean memory
                    del inputs_ax
                    del inputs_sag
                    del inputs_cor
                    del label_ax
                    del label_sag
                    del label_cor


                labels = labels[:, 1, :, :] # 1 for grabbing center slice
                labels = labels[:, None, :, :] # Expand dim

                if config.augmentations['mixup']:
                    inputs, labels = mixup(inputs, labels, alpha = config.augmentations['alpha'])

                labels = labels.to(device)
                inputs = inputs.to(device)

                outputs = model(inputs)
                # sotmax of outputs

                loss = loss_function(outputs, labels, smoothing=config.hyperparams['smoothing'], alpha=config.hyperparams['alpha'], beta=config.hyperparams['beta'], gamma=config.hyperparams['gamma'])
                loss = loss / config.hyperparams['batch_accumulation']
                epoch_loss += loss.item()
                loss.backward()

                if (i + 1) % config.hyperparams['batch_accumulation'] == 0:
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), config.hyperparams['gradclip'])
                    optimizer.step()
                    model.zero_grad(set_to_none = True)
                
                del inputs
                del labels

            epoch_len += len(sample_data)

            # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        b = datetime.datetime.now().replace(microsecond=0)
        print(b-a)
        f1 = []
        fbeta = []
        fast_start_iter = 1
        if epoch % config.structure.val_interval == 0:
            model.eval()
            console.rule("[bold red] Validation")
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                step = 0
                epoch_loss_eval = 0
                for fast_start_iter, large_val_data in enumerate(track(val_loader, description = f"Validation...")):

                    # if epoch < config.hyperparams.fast_start_epoch and fast_start_iter > 1:
                    #     print('Fast broke val...')
                    #     break

                    sample_data_val_ax = utils.Minisampler3slice(large_val_data, all_orientations = config.structure['all_orientations'])
                    sample_loader_val_ax = DataLoader(sample_data_val_ax, batch_size = config.structure['mini_batch_size'], num_workers = 0, shuffle = True, drop_last = True, pin_memory = torch.cuda.is_available())


                    for val_batch_data_ax in sample_loader_val_ax:
                            
                        step += 1

                        val_images = val_batch_data_ax['image']
                        val_labels = val_batch_data_ax['label']

                        testersys.values_standardized(val_images.cpu().numpy()) # Check if correct intensity
                        testersys.labels_binary(val_labels.cpu().numpy()) # Check if labels are binary

                        if config.structure['all_orientations']:
                            val_images_ax = val_images[:, 0:3, :, :]
                            val_images_sag = val_images[:, 3:6, :, :]
                            val_images_cor = val_images[:, 6:9, :, :]
                            val_label_ax = val_labels[:, 0:3, :, :]
                            val_label_sag = val_labels[:, 3:6, :, :]
                            val_label_cor = val_labels[:, 6:9, :, :]

                            val_images = torch.cat((val_images_ax, val_images_sag, val_images_cor), dim = 0)
                            val_labels = torch.cat((val_label_ax, val_label_sag, val_label_cor), dim = 0)

                            # clean memory
                            del val_images_ax
                            del val_images_sag
                            del val_images_cor
                            del val_label_ax
                            del val_label_sag
                            del val_label_cor


                        val_labels = val_labels[:, 1, :, :]
                        val_labels = val_labels[:, None, :, :]


                        val_labels = val_labels.to(device)
                        val_images = val_images.to(device)
                        # roi_size = (-1, -1)
                        # sw_batch_size = 4
                        # padding_mode = 'constant'

                        val_outputs = model(val_images)

                        epoch_loss_eval += loss_function(val_outputs, val_labels, smoothing=config.hyperparams['smoothing'], alpha=config.hyperparams['alpha'], beta=config.hyperparams['beta'], gamma=config.hyperparams['gamma']).item()
                        
                        val_outputs = torch.sigmoid(val_outputs)
                        # val_outputs = post_trans(val_outputs) # Threshold into discrete values
                        
                        # hausdorff_distance, fbeta_score, f1_score
                        val_outputs = val_outputs.cpu().numpy()
                        val_labels = val_labels.cpu().numpy()
                        metric_outputs = get_all_metrics(val_outputs, val_labels, beta=2)
                        f1.append(metric_outputs['f1'])
                        fbeta.append(metric_outputs['fbeta'])

                metric_outputs['f1'] = np.mean(f1)
                metric_outputs['fbeta'] = np.mean(fbeta)
                if not os.path.exists(f'output_epoch_{epoch}'):
                    os.mkdir(f'output_epoch_{epoch}')
                for l, image in enumerate(val_images):
                    val_outputs = post_trans(val_outputs) # Threshold into discrete values
                    fig, ax = plt.subplots(1, 2)
                    img = image[1, :, :].cpu().numpy()
                    img = (img - np.min(img))
                    img = (img / (np.max(img) + 1e-5))
                    ax[0].imshow(img, cmap='gray')
                    ax[1].imshow(val_outputs[l][0, :, :])
                    plt.savefig(f'output_epoch_{epoch}/label{l}.png')
                    plt.close('all')
                
                del val_labels
                del val_images
                del val_outputs

                # aggregate the final mean dice result
                epoch_loss_eval /= step
                loss_array_val.append(epoch_loss_eval)
                scheduler.step(metric_outputs['fbeta'])
                # reset the status for next validation round
                if metric_outputs['fbeta'] > best_metric:
                    best_metric = metric_outputs['fbeta']
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_labelmentation2d_dict.pth")
                    torch.save(optimizer.state_dict(), "best_metric_model_labelmentation2d_dict_optim.pth")
                    print("saved new best metric model and optimizer")
                print(
                    f"current epoch: f1: {metric_outputs['f1']} best fbeta: {best_metric} at epoch {best_metric_epoch}, Fbeta: {metric_outputs['fbeta']}")
                # plot losses
                with open('losses.txt', 'a') as f:
                    f.write(f'{epoch_loss_values[-1]}|{loss_array_val[-1]}|{metric_outputs["f1"]}|{metric_outputs["fbeta"]}\n')
                plt.plot(epoch_loss_values, label = 'Training loss', color = 'black')
                plt.plot(loss_array_val, label = 'Validation loss', color = 'red')
                plt.legend()
                plt.savefig('loss.png')
                plt.close('all')

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    # writer.close()




if __name__ == "__main__":

    # Deterministic seeding
    monai.utils.set_determinism(seed=0, additional_settings=None)
    main()
