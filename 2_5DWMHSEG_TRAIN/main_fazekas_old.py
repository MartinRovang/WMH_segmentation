import logging
import os
import sys
from monai.transforms import transform
import torch
from torch._C import dtype
import torch.nn as nn
from torch.utils.data import DataLoader
import monai
from monai.data import pad_list_data_collate
import utils
import matplotlib.pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig
from rich.progress import track
import testersys
from sklearn.metrics import accuracy_score, classification_report
from transform_functions import load_fazekas

@hydra.main(config_path="conf", config_name="config")
def main(config : DictConfig):
    """Main training and validation"""
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    init_msg = f"\n \
            GPU :{config.hardware['gpu']}\n \
            Model: {config.structure['model']}\n \
            Train: {config.datasets['trainfolder']}\n \
            Validation: {config.datasets['valfolder']}\n \
            Loadweights: {config.structure['loadweights']}\n \
            Loadoptim: {config.structure['loadoptim']}\n \
            mini_batch_size: {config.structure['mini_batch_size']}\n \
            train_patient_batch_size: {config.structure['train_patient_batch_size']}\n \
            validation_patient_batch_size: {config.structure['validation_patient_batch_size']}\n \
            all_orientations: {config.structure['all_orientations']}\n \
            gpu: {config.hardware['gpu']}\n \
            lr: {config.hyperparams['lr']}\n \
            epochs: {config.hyperparams['epochs']}\n \
            gamma: {config.hyperparams['gamma']}\n \
            alpha: {config.hyperparams['alpha']}\n \
            beta: {config.hyperparams['beta']}\n \
            smoothing: {config.hyperparams['smoothing']}\n \
            "
            
    print(init_msg)
    

    
    train_files, val_files = utils.dataprocesser_fazekas(config.datasets['trainfolder'], config.datasets['valfolder'])


    loss_function = nn.CrossEntropyLoss(weight=None, reduction='mean')
    softmax = nn.Softmax(dim = 1)
    # loss_function = monai.losses.DiceLoss(include_background=False, to_onehot_y=False, sigmoid=True)
    
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device(config.hardware['gpu'] if torch.cuda.is_available() else "cpu")


    model = monai.networks.nets.EfficientNetBN("efficientnet-b0",
                                                spatial_dims=3, 
                                                pretrained=False,
                                                in_channels=1,
                                                num_classes=4
                                                ).to(device)

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
    train_ds = monai.data.Dataset(data = train_files, transform = load_fazekas)


    train_loader = DataLoader(
        train_ds,
        batch_size = config.structure['train_patient_batch_size'],
        shuffle = True,
        num_workers = 0,
        collate_fn = pad_list_data_collate,
        drop_last = True,
        pin_memory = torch.cuda.is_available(),
    )

    val_ds = monai.data.Dataset(data = val_files, transform = load_fazekas)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    val_loader = DataLoader(
        val_ds,
        batch_size=config.structure['validation_patient_batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=pad_list_data_collate,
        drop_last = True,
        pin_memory=torch.cuda.is_available(),
    )

    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    loss_array_val = []


    for epoch in range(config.hyperparams.epochs):
        print(f"epoch {epoch + 1}/{config.hyperparams.epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for large_batch in track(train_loader, description = "Training..."):

            inputs = large_batch['image']
            labels = large_batch['label']


            print(inputs.shape)


            labels = labels.to(torch.float32)
            inputs = inputs.to(torch.float32)
            labels = labels.to(device).long()
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            print(softmax(outputs.detach()))
            print(labels)
            epoch_loss += loss.item()
            step += 1


        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")


        if epoch % config.structure.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                step = 0
                epoch_loss_eval = 0
                accuracy = 0
                for large_val_data in track(val_loader, description = "Validation..."):
                    val_images = large_val_data['image']
                    val_labels = large_val_data['label']

                    val_labels = val_labels.to(torch.float32)
                    val_images = val_images.to(torch.float32)
                    val_labels = val_labels.to(device).long()
                    val_images = val_images.to(device)
                    val_outputs = model(val_images)
                    val_outputs = softmax(val_outputs)
                    predictions = torch.argmax(val_outputs, dim = 1)

                    epoch_loss_eval += loss_function(val_outputs, val_labels).item()
                    predictions = torch.argmax(val_outputs, dim = 1)
                    accuracy += accuracy_score(predictions.cpu().numpy(), val_labels.cpu().numpy())
                    step += 1

                # aggregate the final mean dice result
                epoch_loss_eval /= step
                accuracy /= step
                loss_array_val.append(epoch_loss_eval)
                scheduler.step(accuracy)
                # reset the status for next validation round
                if accuracy > best_metric:
                    best_metric = accuracy
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_labelmentation2d_dict.pth")
                    torch.save(optimizer.state_dict(), "best_metric_model_labelmentation2d_dict_optim.pth")
                    print("saved new best metric model and optimizer")
                print(f"current epoch: {epoch + 1} current avg accuracy: {accuracy} best avg accuracy: {best_metric} at epoch {best_metric_epoch}")

                # plot losses
                with open('losses.txt', 'a') as f:
                    f.write(f'{epoch_loss_values[-1]}|{loss_array_val[-1]}|{accuracy}\n')
                plt.plot(epoch_loss_values, label = 'Training loss', color = 'black')
                plt.plot(loss_array_val, label = 'Validation loss', color = 'red')
                plt.legend()
                plt.savefig('loss.png')
                plt.close()

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")



if __name__ == "__main__":

    # Deterministic seeding
    monai.utils.set_determinism(seed=0, additional_settings=None)
    main()
