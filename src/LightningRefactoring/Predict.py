import numpy as np
import pandas as pd
import os
from icecream import ic
import argparse

import torch

from Net import EffNet
from DataModule import DataModuleClass

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import SimpleITK as sitk

from monai.transforms import (
    RandRotate,
    SpatialPad,
    CenterSpatialCrop,
    RandSpatialCrop,
    Compose,
)
import matplotlib.pyplot as plt

from DataModule import LoadJsonLandmarks


def gen_plot(direction, direction_hat):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.quiver(0, 0, 0, direction[0], direction[1], direction[2], color='r', label='y')
    ax.quiver(0, 0, 0, direction_hat[0], direction_hat[1], direction_hat[2], color='b', label='y_hat')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim([0, direction[0]+0.1])
    ax.set_ylim([0, direction[1]+0.1])
    ax.set_zlim([0, direction[2]+0.1])
    ax.legend()
    plt.show()


def main():
    data_dir = "/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/Test"

    landmark = 'S'
    out_dir = "/home/luciacev/Desktop/Luc_Anchling/Training_ALI/lm_"+landmark+"/"
    
    csv_path = os.path.join(data_dir, 'CSV', 'lm_{}'.format(landmark))

    df_train = pd.read_csv(os.path.join(csv_path, 'train.csv'))
    df_val = pd.read_csv(os.path.join(csv_path, 'val.csv'))
    df_test = pd.read_csv(os.path.join(csv_path, 'test.csv'))
    
    train_transform = Compose([
                SpatialPad(
                    spatial_size=(160, 160, 160),
                    value=0,
                ),
                RandSpatialCrop(
                    roi_size=(128, 128, 128),
                    random_size=False,
                ),
                ])

    val_transform = Compose([
                SpatialPad(
                    spatial_size=(160, 160, 160),
                    value=0,
                ),
                CenterSpatialCrop(
                    roi_size=(128, 128, 128),
                ),
                ])

    db = DataModuleClass(df_train, df_val, df_test, landmark=landmark, batch_size=1, train_transform=train_transform, val_transform=val_transform, test_transform=val_transform)
    db.setup('test')

    model = EffNet(lr=0.0001)

    model.load_state_dict(torch.load(os.path.join(out_dir,'checkpoints/epoch=65-val_loss=0.00.ckpt'))['state_dict'])

    model.to('cuda')

    model.eval()
    ds_test = db.test_dataloader()
    with torch.no_grad():
        for i, batch in enumerate(ds_test):
            scan, direction, physical_origin = batch
            direction_pred = model(scan.to('cuda'))
            direction_pred = direction_pred.cpu().numpy()
            direction = direction.numpy()

            # ic(direction_pred, direction)
            direction_pred = direction_pred[0]# / np.linalg.norm(direction_pred[0])
            gen_plot(direction[0], direction_pred)
            # break

if __name__ == "__main__":
    main()