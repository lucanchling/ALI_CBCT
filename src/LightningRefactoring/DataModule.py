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



def LoadJsonLandmarks(ldmk_path,landmark=None):
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
        try:
            lm_ph_coord = np.array([markup["position"][0],markup["position"][1],markup["position"][2]])
            lm_coord = lm_ph_coord.astype(np.float64)
            landmarks[markup["label"]] = lm_coord
        except IndexError:
            # print("Landmark {} not found for file {}".format(markup["label"],ldmk_path))
            pass
    if landmark is None:
        return landmarks
    else:
        return landmarks[landmark]


class DataModuleClass(pl.LightningDataModule):
    # It is used to store information regarding batch size, transforms, etc. 
    def __init__(self, df_train, df_val, df_test, mount_point='./', landmark=None, batch_size=1, num_workers=4, train_transform=None, val_transform=None, test_transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.landmark = landmark
        self.mount_point = mount_point
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def prepare_data(self):
    # On only ONE GPU. 
    # It’s usually used to handle the task of downloading the data. 
        pass

    def setup(self, stage=None):
    # On ALL the available GPU. 
    # It’s usually used to handle the task of loading the data. (like splitting data, applying transform etc.)
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = DatasetClass(df=self.df_train, mount_point=self.mount_point, landmark=self.landmark, transform=self.train_transform)
            self.val_dataset = DatasetClass(df=self.df_val, mount_point=self.mount_point, landmark=self.landmark, transform=self.val_transform)
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = DatasetClass(df=self.df_test, mount_point=self.mount_point, landmark=self.landmark, transform=self.test_transform)

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
        self.mount_point = mount_point

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        scan_path = self.df['scan_path'][idx]
        lm_path = self.df['landmark_path'][idx]

        scan = sitk.ReadImage(scan_path)
        size = np.array(scan.GetSize())
        spacing = np.array(scan.GetSpacing())
        origin = np.array(scan.GetOrigin())

        lm = LoadJsonLandmarks(lm_path, self.landmark)

        # if self.RandomRotation: # Random Rotation for training --> scan & landmarks
        #     scan, lm = RandRotateLandmarks(scan, lm, x_range=self.angle, y_range=self.angle, z_range=self.angle)
        
        if self.transform is not None:
            scan, lm = self.transform(scan, lm)

        scan = torch.Tensor(sitk.GetArrayFromImage(scan)).unsqueeze(0)  # Conversion for Monai transforms

        # if self.transform:
            
            
        #     scan = self.transform(scan) # Apply transforms
            
        #     try:# Get the crop parameters
        #         cropParam = scan.__getattribute__('data').__getattribute__('applied_operations')[-1]['extra_info']['cropped']
        #         cropStart = [cropParam[i] for i in range(0,len(cropParam),2)]
        #         cropStart = cropStart[::-1]
        #     except:
        #         cropStart = [0,0,0]

        physical_origin = origin #- np.array([32,32,32])*spacing + np.array(cropStart)*spacing

        # Compute the direction from the physical origin towards the Landmark position
        # direction = {key : np.array(value - physical_origin) / np.linalg.norm(value-physical_origin) for key,value in lm.items()}
        length = np.linalg.norm(lm - physical_origin)

        direction = np.array(lm - physical_origin) / length
        direction = torch.Tensor(direction)

        # Compute the scaling attribute
        scale = length / np.linalg.norm(size * spacing)
                
        return scan, direction, scale, scan_path, lm

class RandomRotation3D(pl.LightningDataModule):
    def __init__(self, x_angle=np.pi/2, y_angle=np.pi/2, z_angle=np.pi/2):
        super().__init__()
        self.x_angle = x_angle
        self.y_angle = y_angle
        self.z_angle = z_angle

    def __call__(self, scan, lm):
        randanglex = np.random.uniform(-self.x_angle, self.x_angle)
        randangley = np.random.uniform(-self.y_angle, self.y_angle)
        randanglez = np.random.uniform(-self.z_angle, self.z_angle)
        R = sitk.Euler3DTransform()
        R.SetRotation(randanglex, randangley, randanglez)
        
        scan = sitk.Resample(image1=scan, transform=R, interpolator=sitk.sitkAffine)

        rotmatrix = np.array(R.GetMatrix()).reshape(3,3)
        try:    # if lm is a single landmark
            lm = np.matmul(lm, rotmatrix)
        except ValueError:  # if lm is a dict
            lm = {key : np.matmul(lm[key], rotmatrix) for key in lm.keys()}
        
        return scan, lm   

# if __name__ == "__main__":
#     data_dir="/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/RESAMPLED/"
#     landmark = 'S'
#     csv_path = os.path.join(data_dir, 'CSV', 'lm_{}'.format(landmark))

#     df_train = pd.read_csv(os.path.join(csv_path, 'train.csv'))
#     df_val = pd.read_csv(os.path.join(csv_path, 'val.csv'))
#     df_test = pd.read_csv(os.path.join(csv_path, 'test.csv'))

#     # test1()
#     # LM=[]
#     # for i in range(len(df_train)):
#     #     LM.append(LoadJsonLandmarks(sitk.ReadImage(df_train['scan_path'][i]),df_train['landmark_path'][i]).keys())
#     # print([True if 'B' in i else False for i in LM])    

#     db = DataModuleClass(df_train, df_val, df_test,landmark=landmark, batch_size=1, num_workers=0, train_transform=RandomRotation3D(angle=np.pi/2))

#     db.prepare_data()
#     db.setup(stage='fit')

#     dl = db.train_dataloader()
#     Scale, Length = [], []
#     for i, (scan, direction, scale, path) in enumerate(dl):
#         # write sitk image
#         sitk.WriteImage(sitk.GetImageFromArray(scan.squeeze().numpy()), os.path.join(data_dir, 'TEST', 'scan_{}.nii.gz'.format(i)))
#         break