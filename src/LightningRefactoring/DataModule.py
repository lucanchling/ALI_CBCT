import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
from glob import glob
import numpy as np
import pandas as pd
from icecream import ic
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import SimpleITK as sitk
import json

from monai.transforms import (
    RandRotate,
    SpatialPad,
    RandSpatialCrop,
    Compose,
)

def LoadJsonLandmarks(img, ldmk_path):
    """
    Load landmarks from json file
    
    Parameters
    ----------
    img : sitk.Image
        Image to which the landmarks belong
    ldmk_path : str
        Path to the json file
    gold : bool, optional
        If True, load gold standard landmarks, by default False
    
    Returns
    -------
    dict
        Dictionary of landmarks
    
    Raises
    ------
    ValueError
        If the json file is not valid
    """
    with open(ldmk_path) as f:
        data = json.load(f)
    markups = data["markups"][0]["controlPoints"]
    landmarks = {}
    for markup in markups:
        lm_ph_coord = np.array([markup["position"][0],markup["position"][1],markup["position"][2]])
        lm_coord = lm_ph_coord.astype(np.float64)
        landmarks[markup["label"]] = lm_coord
    return landmarks

class DataModuleClass(pl.LightningDataModule):
    # It is used to store information regarding batch size, transforms, etc. 
    def __init__(self, df_train, df_val,df_test, mount_point='./', landmark=None, batch_size=1, num_workers=4, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.landmark = landmark
        self.transform = transform
        self.mount_point = mount_point

    def prepare_data(self):
    # On only ONE GPU. 
    # It’s usually used to handle the task of downloading the data. 
        pass

    def setup(self, stage=None):
    # On ALL the available GPU. 
    # It’s usually used to handle the task of loading the data. (like splitting data, applying transform etc.)
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = DatasetClass(df=self.df_train, mount_point=self.mount_point, landmark=self.landmark, transform=self.transform)
            self.val_dataset = DatasetClass(df=self.df_val, mount_point=self.mount_point, landmark=self.landmark, transform=self.transform)
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = DatasetClass(df=self.df_test, mount_point=self.mount_point, landmark=self.landmark, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class DatasetClass(Dataset):
    def __init__(self, df, mount_point='',landmark=None, transform=None):
        super().__init__()
        self.df = df
        self.transform = transform
        self.landmark = landmark

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        scan_path = self.df['scan_path'][idx]
        lm_path = self.df['landmark_path'][idx]

        scan = sitk.ReadImage(scan_path)
        spacing = np.array(scan.GetSpacing())
        origin = np.array(scan.GetOrigin())

        lm = LoadJsonLandmarks(scan,lm_path)

        scan = torch.Tensor(sitk.GetArrayFromImage(scan)).unsqueeze(0)

        if self.transform:
            scan = self.transform(scan)
            
            matrix = scan.__getattribute__('affine')    # get the affine matrix of the transform

            # Apply the Rotation Transform to the landmarks
            lm = {key : np.matmul(matrix[:3,:3], value) for key, value in lm.items()}
            
            # Get the crop parameters
            cropParam = scan.__getattribute__('data').__getattribute__('applied_operations')[2]['extra_info']['cropped']
            cropStart = [cropParam[i] for i in range(0,len(cropParam),2)]
            cropStart = cropStart[::-1]

        else:
            cropStart = [0,0,0]

        physical_origin = origin - np.array([32,32,32])*spacing + np.array(cropStart)*spacing

        # Compute the direction from the physical origin towards the Landmark position
        direction = {key : np.array(value - physical_origin) / np.linalg.norm(value-physical_origin) for key,value in lm.items()}

        # To select only one landmark
        if self.landmark is not None:
            direction = torch.Tensor(direction[self.landmark])

        return scan, direction


if __name__ == "__main__":
    data_dir="/Users/luciacev-admin/Desktop/Luc_Anchling/DATA/ALICBCT/Test"

    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_val = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    # test1()
    # LM=[]
    # for i in range(len(df_train)):
    #     LM.append(LoadJsonLandmarks(sitk.ReadImage(df_train['scan_path'][i]),df_train['landmark_path'][i]).keys())
    # print([True if 'B' in i else False for i in LM])    

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

    db = DataModuleClass(df_train, df_val, df_test,'B', batch_size=1, num_workers=0,transform=transform)

    db.prepare_data()
    db.setup(stage='fit')

    dl_train = db.train_dataloader()

    for i, (scan, direction) in enumerate(dl_train):
        ic(scan.shape, direction)
        break      
