import numpy as np
import pandas as pd
import os
from icecream import ic
import argparse

import torch

from net import EffNet
from DataModule import DataModuleClass

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from monai.transforms import (
    RandRotate,
    SpatialPad,
    RandSpatialCrop,
    Compose,
)


def main(args):
    
    mount_point = args.mount_point

    df_train = pd.read_csv(os.path.join(mount_point, args.csv_train))
    df_val = pd.read_csv(os.path.join(mount_point, args.csv_valid))
    df_test = pd.read_csv(os.path.join(mount_point, args.csv_valid))

    # test1()
    # LM=[]
    # for i in range(len(df_train)):
    #     LM.append(LoadJsonLandmarks(sitk.ReadImage(df_train['scan_path'][i]),df_train['landmark_path'][i]).keys())
    # print([True if 'B' in i else False for i in LM])    

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    transform = Compose([  
                RandRotate(
                    range_x=np.pi/4,
                    range_y=np.pi/4,
                    range_z=np.pi/4,
                    prob=1,
                    keep_size=True,
                ),
                SpatialPad(
                    spatial_size=(160, 160, 160),
                    value=0,
                ),
                RandSpatialCrop(
                    roi_size=(128, 128, 128),
                    random_size=False,
                ),
                ])

    db = DataModuleClass(df_train, df_val, df_test,'B', batch_size=1, num_workers=12,transform=transform)

    db.prepare_data()
    db.setup(stage='fit')

    model = EffNet(lr=args.lr)
    
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)
        
    trainer = pl.Trainer(
        logger = logger,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[early_stop_callback, checkpoint_callback],
        devices=torch.cuda.device_count(), 
        accelerator="cpu", 
        num_sanity_val_steps=0,
        profiler=args.profiler
    )

    trainer.fit(model, db)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Teeth challenge Training')
    parser.add_argument('--csv_train', help='CSV with column surf', type=str, default='train.csv')    
    parser.add_argument('--csv_valid', help='CSV with column surf', type=str, default='val.csv')
    parser.add_argument('--csv_test', help='CSV with column surf', type=str, default='test.csv')      
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)    
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--model', help='Model to continue training', type=str, default= None)
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/Users/luciacev-admin/Desktop/Luc_Anchling/DATA/ALICBCT/Test")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=20)    
    parser.add_argument('--train_sphere_samples', help='Number of training sphere samples or views used during training and validation', type=int, default=4)    
    parser.add_argument('--patience', help='Patience for early stopping', type=int, default=30)
    parser.add_argument('--profiler', help='Use a profiler', type=str, default=None)
    parser.add_argument('--property', help='label of segmentation', type=str, default="PredictedID")
    
    
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="monai")


    args = parser.parse_args()

    main(args)