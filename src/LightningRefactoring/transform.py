from monai.transforms import (
    RandRotate,
    SpatialPad,
    RandSpatialCrop,
    Compose,
)
import pandas as pd
import os
import numpy as np
from icecream import ic
from DataModule import DataModuleClass
import SimpleITK as sitk
import torch
import json

def ApplyTransform(source,transform):
    '''
    Apply a transform matrix to a set of landmarks
    
    Parameters
    ----------
    source : dict
        Dictionary of landmarks
    transform : np.array
        Transform matrix
    
    Returns
    -------
    source : dict
        Dictionary of transformed landmarks
    '''
    sourcee = source.copy()
    for key in sourcee.keys():
        sourcee[key] = transform @ np.append(sourcee[key],1)
        sourcee[key] = sourcee[key][:3]
    return sourcee

def ApplyRotation(source,R):
    '''
    Apply a rotation matrix to a set of landmarks
    
    Parameters
    ----------
    source : dict
        Dictionary of landmarks
    R : np.array
        Rotation matrix
    
    Returns
    -------
    source : dict
        Dictionary of transformed landmarks
    '''
    sourcee = source.copy()
    for key in sourcee.keys():
        sourcee[key] = R @ sourcee[key]
    return sourcee

def WriteJsonLandmarks(landmarks, img ,output_file):
    '''
    Write the landmarks to a json file
    
    Parameters
    ----------
    landmarks : dict
        landmarks to write
    output_file : str
        output file name
    '''
    origin = np.array(img.GetOrigin())
    with open('/Users/luciacev-admin/Desktop/Luc_Anchling/DATA/ALICBCT/Test/TEMP.mrk.json', 'r') as outfile:
        tempData = json.load(outfile)
    for i in range(len(tempData["markups"][0]["controlPoints"])):
        pos = landmarks[tempData['markups'][0]['controlPoints'][i]['label']].tolist()
        pos = pos #+ origin.tolist()
        tempData['markups'][0]['controlPoints'][i]['position'] = [pos[0],pos[1],pos[2]]
    with open(output_file, 'w') as outfile:
        json.dump(tempData, outfile, indent=4)

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
    
    spacing = np.array(img.GetSpacing())
    origin = img.GetOrigin()
    origin = np.array([origin[0],origin[1],origin[2]])
    with open(ldmk_path) as f:
        data = json.load(f)
    
    markups = data["markups"][0]["controlPoints"]
    
    landmarks = {}
    for markup in markups:
        lm_ph_coord = np.array([markup["position"][0],markup["position"][1],markup["position"][2]])
        # lm_coord = ((lm_ph_coord - origin) / spacing).astype(np.float16)
        # lm_ph_coord = (lm_ph_coord - origin) / spacing
        lm_coord = lm_ph_coord.astype(np.float64)
        landmarks[markup["label"]] = lm_coord
    return landmarks

if __name__ == "__main__":
    data_dir="/Users/luciacev-admin/Desktop/Luc_Anchling/DATA/ALICBCT/Test"

    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    # test1()
    # LM=[]
    # for i in range(len(df_train)):
    #     LM.append(LoadJsonLandmarks(sitk.ReadImage(df_train['scan_path'][i]),df_train['landmark_path'][i]).keys())
    # print([True if 'B' in i else False for i in LM])    
    # db = DataModuleClass(df_train, df_val, df_test,'B', batch_size=1, num_workers=0)

    # db.prepare_data()
    # db.setup(stage='fit')

    # DS = db.train_dataset

    # # Apply transforms

    # DS.transform = transform
 
    # img = sitk.GetImageFromArray(DS[0][0].squeeze(0).numpy())
    # sitk.WriteImage(img, data_dir + '/test.nii.gz')

    # lm = DS[0][1]
    # ic(df_train['landmark_path'][0])


    scan = sitk.ReadImage(df_train['scan_path'][0])
    lm = LoadJsonLandmarks(scan,df_train['landmark_path'][0])
    spacing = np.array(scan.GetSpacing())
    origin = np.array(scan.GetOrigin())
    # scan.SetOrigin(-np.array([64,64,64])/spacing)
    
    # Set the new origin to landmarks and scan before applying the transform
    new_origin = np.array([32,32,32])*spacing
    # scan.SetOrigin((origin - new_origin))
    # lm = {key: lm[key] - (new_origin/spacing) for key in lm.keys()}

    # ic(lm['B'])
    scan_torch = torch.Tensor(sitk.GetArrayFromImage(scan))

    # ic(lm)
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

    roi_start = np.ones(3)
    scan_transformed = transform(scan_torch.unsqueeze(0))   

    # Random crop to 128x128x128
    # crop = SpatialCrop((128,128,128))
    # scan_transformed = crop(scan_transformed)
    # ic(scan_transformed.__getattribute__('data'))

    matrix = scan_transformed.__getattribute__('affine')
    ic(matrix)
    cropParam = scan_transformed.__getattribute__('data').__getattribute__('applied_operations')[2]['extra_info']['cropped']
    # ic(cropParam)

    cropStart = [cropParam[i] for i in range(0,len(cropParam),2)]
    cropStart = cropStart[::-1]
    # ic(cropStart)

    
    # Get randspatialcrop parameters
    # crop = matrix[:3,3]+32
    # ic(crop)
    # T = np.array(3*[32])
    # Origin = np.array(scan.GetOrigin())
    # Spacing = np.array(scan.GetSpacing())
    # # ic(matrix[:3, 3])

    # Apply only the rotation to the landmarks
    lm = ApplyRotation(lm,matrix[:3,:3])

    # Apply the translation to the landmarks
    # lm = {key: lm[key] - matrix[:3,3] for key in lm.keys()}


    WriteJsonLandmarks(lm,scan,output_file=data_dir+'/output.mrk.json')
    img = sitk.GetImageFromArray(scan_transformed.squeeze(0).numpy())
    # ic(new_origin)
    img.SetOrigin(origin - new_origin + cropStart*spacing)
    img.SetDirection(scan.GetDirection())
    img.SetSpacing(scan.GetSpacing())
    sitk.WriteImage(img, data_dir + '/output.nii.gz')
    
    # ic(scan.GetOrigin(),scan.GetSpacing(),scan.GetDirection(),scan.GetSize())
    # ic(img.GetOrigin(),img.GetSpacing(),img.GetDirection(),img.GetSize())