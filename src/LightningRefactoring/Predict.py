import numpy as np
import pandas as pd
import os
from icecream import ic
import argparse

import torch

from Net import EffNet
from DataModule import DataModuleClass
from utils import WriteJson

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



def gen_plot(direction, direction_hat, scale, scale_hat):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.quiver(0, 0, 0, direction[0], direction[1], direction[2], color='r', label='y')
    ax.quiver(0, 0, 0, direction_hat[0], direction_hat[1], direction_hat[2], color='b', label='y_hat')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim([-.5, .5])
    ax.set_ylim([-.5, .5])
    ax.set_zlim([-.5, .5])
    ax.legend()
    plt.title('Scale : {:.4f}  |  Scale_hat : {:.4f}'.format(scale,scale_hat))
    # plt.title('DIRECTION:  {:.4f}  |  {:.4f}  |  {:.4f}\nDIRECTION_HAT :  {:.4f}  |  {:.4f}  |  {:.4f}'.format(direction[0],direction[1],direction[2],direction_hat[0],direction_hat[1],direction_hat[2]))
    plt.show()


def main(args):
    data_dir = "/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/ALLRESAMPLED"

    landmark = args.landmark

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

    db = DataModuleClass(df_train, df_val, df_test, landmark=landmark, batch_size=1, train_transform=None, val_transform=None, test_transform=None)
    db.setup('test')
    
    model = EffNet(lr=0.0001)

    model.load_state_dict(torch.load(args.checkpoint)['state_dict'])

    model.to('cuda')
    
    model.eval()
    ds_test = db.test_dataloader()
    with torch.no_grad():
        for i, batch in enumerate(ds_test):
            scan, direction, scale, scan_path, lm = batch
            direction_pred, scale_pred = model(scan.to('cuda'))
            direction_pred = direction_pred.cpu().numpy()
            direction = direction.numpy()

            img = sitk.ReadImage(scan_path[0])
            spacing = np.array(img.GetSpacing())
            origin = np.array(img.GetOrigin())
            size = np.array(img.GetSize())

            dict_landmark = {landmark:np.array(lm[0]), '{}_pred'.format(landmark): origin + direction_pred[0] * scale_pred.item() * np.linalg.norm(size*spacing)}
            WriteJson(dict_landmark,os.path.join(out_dir, 'pred', os.path.basename(scan_path[0]).replace('.nii.gz','.mrk.json')))

            # ic(direction_pred, direction)
            # ic(scale_pred,scale)
            # direction_pred = direction_pred[0]# / np.linalg.norm(direction_pred[0])
            # gen_plot(direction[0], direction_pred[0], scale.item(), scale_pred.item())
            # print(scale.item(),scale_pred.item())
            # break
        # for i in range(5):
        #     print(model(torch.rand(1,1,128,128,128).to('cuda')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--landmark', type=str, default='UL6O')
    parser.add_argument('--checkpoint', type=str, default='/home/luciacev/Desktop/Luc_Anchling/Training_ALI/lm_UL6O/Models/lr1e-04_bs30_angle0.5.ckpt')
    args = parser.parse_args()
    main(args)    