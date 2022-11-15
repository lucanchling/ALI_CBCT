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

from monai.transforms import (
    RandRotate,
    SpatialPad,
    CenterSpatialCrop,
    RandSpatialCrop,
    Compose,
)

from CallBackClass import DirectionLogger

def main(args):
    
    mount_point = args.mount_point

    csv_path = os.path.join(mount_point, 'CSV', 'lm_{}'.format(args.landmark))

    df_train = pd.read_csv(os.path.join(csv_path, args.csv_train))
    df_val = pd.read_csv(os.path.join(csv_path, args.csv_valid))
    df_test = pd.read_csv(os.path.join(csv_path, args.csv_test))


    checkpoint_callback = ModelCheckpoint(
        dirpath= args.out + 'checkpoints',
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

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

    db = DataModuleClass(df_train, df_val, df_test, landmark=args.landmark, batch_size=args.batch_size, num_workers=args.num_workers, train_transform=train_transform, val_transform=val_transform, test_transform=val_transform)
    
    model = EffNet(lr=args.lr)
    
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    direction_logger = DirectionLogger(log_steps=args.log_every_n_steps)

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=None)
        
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[early_stop_callback, checkpoint_callback,direction_logger],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=db, ckpt_path=args.model)
    
    trainer.test(datamodule=db)

    # print the path of the best model
    ic(trainer.checkpoint_callback.best_model_path)
if __name__ == '__main__':

    
    data_dir = "/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/Test"
    landmark = 'S'
    out_dir = "/home/luciacev/Desktop/Luc_Anchling/Training_ALI/lm_"+landmark+"/"

    parser = argparse.ArgumentParser(description='ALI CBCT Training')
    parser.add_argument('--csv_train', help='CSV with Scan and Landmarks files', type=str, default='train.csv')    
    parser.add_argument('--csv_valid', help='CSV with Scan and Landmarks files', type=str, default='val.csv')
    parser.add_argument('--csv_test', help='CSV with Scan and Landmarks files', type=str, default='test.csv')      
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)    
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--model', help='Model to continue training', type=str, default= None)
    parser.add_argument('--out', help='Output', type=str, default=out_dir)
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default=data_dir)
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=20)
    parser.add_argument('--patience', help='Patience for early stopping', type=int, default=30)
    parser.add_argument('--landmark',help='landmark to train',type=str,default=landmark)

    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=out_dir+'tb_logs/')

    args = parser.parse_args()

    main(args)