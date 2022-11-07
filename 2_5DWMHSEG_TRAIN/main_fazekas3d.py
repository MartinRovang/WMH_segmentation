from email import header
import logging
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import monai
import utils
import matplotlib.pyplot as plt
import numpy as np
from lossfunctions.lossfuncs import FocalTverskyLoss_fazekas
import hydra
from omegaconf import DictConfig
from rich.progress import track
import pickle
from sklearn.metrics import f1_score, fbeta_score
from sklearn import preprocessing
import seaborn as sns
import pandas as pd
import mplcyberpunk

lb = preprocessing.LabelBinarizer()
lb.fit([0, 1, 2, 3])

def get_all_metrics(prediction, target, beta = 2):
    """
    Get all metrics
    """
    output = {'f1': [], 'fbeta': []}
    target = lb.inverse_transform(target.cpu().numpy())
    preds = np.argmax(prediction.cpu().numpy(), axis=1)
    # preds = prediction.cpu().numpy()
    output['f1'] = f1_score(target.flatten(), preds.flatten(), average='macro', zero_division=1)
    output['fbeta'] = fbeta_score(target.flatten(), preds.flatten(), average='macro', zero_division=1, beta=beta)
    return output


@hydra.main(config_path="conf", config_name="config_fazekas")
def main(config : DictConfig):
    """Main training and validation"""
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    
    train_files, val_files = utils.dataprocesser_fazekas(config.datasets['trainfolder'], config.datasets['valfolder'], filename = 'bottleneck_features.pkl', dict_sep = True, fazekas_feature_method = True)
    # loss_function = FocalTverskyLoss_fazekas()
    loss_function = nn.CrossEntropyLoss()
    # loss_function = monai.losses.DiceLoss(include_background=False, to_onehot_y=False, sigmoid=True)
    

    device = torch.device(config.hardware['gpu'] if torch.cuda.is_available() else "cpu")

    if config.structure['model'] == 'UNET':
        from models.Unet_fazekas3D import UNet3D
        model = UNet3D(in_channels=50, use_softmax_head = True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.hyperparams.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    if config.structure['loadweights']:
        state = torch.load(config.structure['weights'], map_location=device)
        model.load_state_dict(state)
        print('Loading weights...')
    if config.structure['loadoptim']:
        print('load optimweights...')
        optimizer.load_state_dict(torch.load(config.structure['optimweights'], map_location = device))
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.scheduelerparams.factor, patience=config.scheduelerparams.patience, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    
    # create a training data loader
    train_ds = monai.data.Dataset(data = train_files)
    val_ds = monai.data.Dataset(data = val_files)
    print(len(train_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size = config.structure['train_patient_batch_size'],
        shuffle = True,
        num_workers = 0,
        drop_last = False,
        pin_memory = torch.cuda.is_available(),
    )


    val_loader = DataLoader(
        val_ds,
        batch_size=config.structure['validation_patient_batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last = False,
        pin_memory=torch.cuda.is_available(),
    )

    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    loss_array_val_final = []
    # writer = SummaryWriter()
    # m = torch.distributions.gamma.Gamma(torch.tensor([0.4]), torch.tensor([0.4]))
    for epoch in range(config.hyperparams.epochs):
        print(f"epoch {epoch + 1}/{config.hyperparams.epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for fast_start_iter, large_batch in enumerate(track(train_loader, description = "Training...")):
            

            for counter, pklfile in enumerate(large_batch['data']):
                if counter == 0:
                    array = list(pickle.load(open(pklfile, 'rb')).values())[0]
                    inputs = torch.Tensor(array)[None, :, :, :, :]
                    labl = list([large_batch['label'][counter].item()])
                    labl = torch.Tensor(labl)[None, :]
                    labels = labl
                else:
                    array = list(pickle.load(open(pklfile, 'rb')).values())[0]
                    inputs = torch.cat((inputs, torch.Tensor(array)[None, :, :, :, :]), dim = 0)
                    labl = list([large_batch['label'][counter].item()])
                    labl = torch.Tensor(labl)[None, :]
                    labels = torch.cat((labels, labl), dim = 0)


            labels = lb.transform(labels.cpu().numpy())
            labels = torch.Tensor(labels)
            labels = labels.to(device)
            inputs = inputs.to(device)
            outputs = model(inputs)
            


            # softmax of outputs
            # loss = loss_function(outputs, labels, smoothing=config.hyperparams['smoothing'], alpha=config.hyperparams['alpha'], beta=config.hyperparams['beta'], gamma=config.hyperparams['gamma'])
            outputs = outputs.softmax(dim=1)
            loss = loss_function(outputs, labels)
            loss = loss / config.hyperparams['batch_accumulation']   
            epoch_loss += loss.item()
            loss.backward()

            if (fast_start_iter + 1) % config.hyperparams['batch_accumulation'] == 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), config.hyperparams['gradclip'])
                optimizer.step()
                model.zero_grad(set_to_none = True)

            # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= (fast_start_iter + 1)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        loss_array_val = []
        f1_score = []
        fbeta_score = []
        if epoch % config.structure.val_interval == 0:
            model.eval()
            with torch.no_grad():
                step = 0

                for fast_start_iter, large_batch in enumerate(track(val_loader, description = "Validation...")):
                    for counter, pklfile in enumerate(large_batch['data']):
                        if counter == 0:
                            array = list(pickle.load(open(pklfile, 'rb')).values())[0]
                            inputs = torch.Tensor(array)[None, :, :, :, :]
                            labl = list([large_batch['label'][counter].item()])
                            labl = torch.Tensor(labl)[None, :]
                            labels = labl
                        else:
                            array = list(pickle.load(open(pklfile, 'rb')).values())[0]
                            inputs = torch.cat((inputs, torch.Tensor(array)[None, :, :, :, :]), dim = 0)
                            labl = list([large_batch['label'][counter].item()])
                            labl = torch.Tensor(labl)[None, :]
                            labels = torch.cat((labels, labl), dim = 0)

                    labels = lb.transform(labels.cpu().numpy())
                    labels = torch.Tensor(labels)
                    labels = labels.to(device)
                    inputs = inputs.to(device)
                    outputs = model(inputs)

                    # loss = loss_function(outputs, labels, smoothing=config.hyperparams['smoothing'], alpha=config.hyperparams['alpha'], beta=config.hyperparams['beta'], gamma=config.hyperparams['gamma'])
                    outputs = outputs.softmax(dim=1)
                    loss = loss_function(outputs, labels)
                    
                    #output to softmax
                    # outputs = loss_function.softmax(outputs)
                    metric_outputs = get_all_metrics(outputs, labels, beta=2)
                    f1_score.append(metric_outputs['f1'])
                    fbeta_score.append(metric_outputs['fbeta'])
                    
                    # aggregate the final mean dice result
                    loss_array_val.append(loss.item())

                scheduler.step(loss.item())
                # reset the status for next validation round
                if np.mean(f1_score) > best_metric:
                    best_metric =  np.mean(f1_score)
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_fazekas3D.pth")
                    torch.save(optimizer.state_dict(), "best_metric_model_fazekas3D_dict_optim.pth")
                    print("saved new best metric model and optimizer")
                print(
                    f"current epoch: f1: { np.mean(f1_score)} best f1: {best_metric} at epoch {best_metric_epoch}, Fbeta: {np.mean(fbeta_score)}")
                # plot losses

                loss_array_val_final.append(np.mean(loss_array_val))

                # save losses to txt file
                with open("loss_array_val_final.txt", "a") as f:
                    f.write(str(epoch_loss) + ',' + str(np.mean(loss_array_val)) + "\n")
                
                # plot losses in seaborn
                data = pd.read_csv("loss_array_val_final.txt", sep='\n', header = None)
                x = [x.split(',') for x in data[0]]
                x = pd.DataFrame(x, columns=['training', 'validation'], dtype=float)
                plt.style.use("cyberpunk")
                x.plot(legend=True)
                mplcyberpunk.add_glow_effects()
                plt.savefig('losses.pdf')
                plt.close()


    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    # writer.close()




if __name__ == "__main__":

    # Deterministic seeding
    monai.utils.set_determinism(seed=0, additional_settings=None)
    main()
