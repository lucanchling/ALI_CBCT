import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


import pandas as pd
from glob import glob 

from icecream import ic

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from torchvision import transforms

import pytorch_lightning as pl

import monai 

class DataModuleClass(pl.LightningDataModule):
    # It is used to store information regarding batch size, transforms, etc. 
    def __init__(self, data_dir, batch_size, num_workers):
        #Define required parameters here
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datadir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    # On only ONE GPU. 
    # It’s usually used to handle the task of downloading the data. 
    def prepare_data(self):
        pass

    # On ALL the available GPU. 
    # It’s usually used to handle the task of loading the data. (like splitting data, applying transform etc.)
    def setup(self, stage=None):
        data_dir = self.datadir

    def train_dataloader(self):
        # Return DataLoader for Training Data here
        pass
    
    def val_dataloader(self):
        # Return DataLoader for Validation Data here
        pass
    
    def test_dataloader(self):
        # Return DataLoader for Testing Data here
        pass


if __name__ == "__main__":
    dl = DataModuleClass(data_dir="/Users/luciacev-admin/Desktop/Luc_Anchling/Projects/ALI_CBCT/src/NEW/test", batch_size=1, num_workers=1)

    ic(dl)