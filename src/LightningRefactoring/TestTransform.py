from monai.transforms import (
    Rotate,
    RandRotate,
    RandRotated,
    SpatialPad,
    SpatialPadd,
    RandSpatialCrop,
    RandSpatialCropd,
    CenterSpatialCrop,
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

from DataModule import LoadJsonLandmarks, RandRotateLandmarks

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
        sourcee[key] = np.matmul(sourcee[key],R)
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
    with open('/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/Test/TEMP.mrk.json', 'r') as outfile:
        tempData = json.load(outfile)
    for i in range(len(tempData["markups"][0]["controlPoints"])):
        try:
            pos = landmarks[tempData['markups'][0]['controlPoints'][i]['label']].tolist()
        except KeyError:
            pass
        tempData['markups'][0]['controlPoints'][i]['position'] = [pos[0],pos[1],pos[2]]
    with open(output_file, 'w') as outfile:
        json.dump(tempData, outfile, indent=4)

if __name__ == "__main__":
    data_dir="/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/RESAMPLED"  
    landmark = "N"

    csv_path = os.path.join(data_dir, 'CSV', 'lm_{}'.format(landmark))
    df_train = pd.read_csv(os.path.join(csv_path, 'train.csv'))

    scan = sitk.ReadImage(df_train['scan_path'][0])
    lm = LoadJsonLandmarks(df_train['landmark_path'][0],landmark=None)
    # data = {'first': {'scan': sitk.GetArrayFromImage(scan), 'landmark': np.array([[lm[landmark]], [lm[landmark]]])}}
    # ic(df_train['landmark_path'][0])
    spacing = np.array(scan.GetSpacing())
    origin = np.array(scan.GetOrigin())
    # scan.SetOrigin(-np.array([64,64,64])/spacing)

    # Set the new origin to landmarks and scan before applying the transform

    # scan.SetOrigin((origin - new_origin))
    # lm = {key: lm[key] - (new_origin/spacing) for key in lm.keys()}

    # ic(lm['B'])
    scan_torch = torch.Tensor(sitk.GetArrayFromImage(scan))

    # ic(lm)
    train_transform = Compose([  
        # RandRotate(
        #     range_x=np.pi/4,
        #     range_y=np.pi/4,
        #     range_z=np.pi/4,
        #     prob=1,
        #     keep_size=True,
        # ),
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

    img, landmark = RandRotateLandmarks(scan, lm, x_range=np.pi/2, y_range=np.pi/2, z_range=np.pi/2)
    WriteJsonLandmarks(landmark,scan,output_file=data_dir+'/output.mrk.json')

    # scan_transformed = train_transform(torch.Tensor(sitk.GetArrayFromImage(img)).unsqueeze(0))
    # try:
    #     cropParam = scan_transformed.__getattribute__('data').__getattribute__('applied_operations')[-1]['extra_info']['cropped']
    #     # ic(cropParam)

    #     cropStart = [cropParam[i] for i in range(0,len(cropParam),2)]
    #     cropStart = cropStart[::-1]
    #     # ic(cropStart)
    # except:
    #     cropStart = [0,0,0]

    # img = sitk.GetImageFromArray(scan_transformed.squeeze(0).numpy())
    # ic(new_origin)
    new_origin = origin #- np.array([32,32,32])*spacing + cropStart*spacing
    img.SetOrigin(new_origin)
    img.SetDirection(scan.GetDirection())
    img.SetSpacing(scan.GetSpacing())
    sitk.WriteImage(img, data_dir + '/output.nii.gz')
    """
    scan_transformed = train_transform(scan_torch.unsqueeze(0))

    # Random crop to 128x128x128
    # crop = SpatialCrop((128,128,128))
    # scan_transformed = crop(scan_transformed)
    # ic(scan_transformed.__getattribute__('data'))
    try:
        rotmatrix = scan_transformed.__getattribute__('data').__getattribute__('applied_operations')[0]['extra_info']['extra_info']['rot_mat']
    except:
        rotmatrix = np.eye(4)
    # ic(rotmatrix[:3,:3])
    RotTranslation = rotmatrix[:3,3]
    # RotTranslation = RotTranslation[::-1]
    matrix = scan_transformed.__getattribute__('affine')
    # ic(matrix[:3,:3])
    # ic(RotTranslation)

    # # Get randspatialcrop parameters
    # crop = matrix[:3,3]+32
    # ic(crop)
    # T = np.array(3*[32])
    # Origin = np.array(scan.GetOrigin())
    # Spacing = np.array(scan.GetSpacing())
    # ic(matrix[:3, 3])

    # Apply only the rotation to the landmarks
    lm = ApplyRotation(lm,rotmatrix[:3,:3])

    # Apply the translation to the landmarks
    # lm = {key: lm[key] - matrix[:3,3] for key in lm.keys()}

    WriteJsonLandmarks(lm,scan,output_file=data_dir+'/output.mrk.json')
    img = sitk.GetImageFromArray(scan_transformed.squeeze(0).numpy())
    new_origin = origin #- np.array([32,32,32])*spacing + cropStart*spacing
    img.SetOrigin(new_origin)
    img.SetDirection(scan.GetDirection())
    img.SetSpacing(scan.GetSpacing())
    sitk.WriteImage(img, data_dir + '/output.nii.gz')



    # img1 = sitk.GetImageFromArray(scan_transformed.squeeze(0).numpy())
    # img1.SetOrigin(new_origin - RotTranslation*spacing)
    # img1.SetDirection(scan.GetDirection())
    # img1.SetSpacing(scan.GetSpacing())
    # sitk.WriteImage(img1, data_dir + '/outputbis.nii.gz')
    # ic(scan.GetOrigin(),scan.GetSpacing(),scan.GetDirection(),scan.GetSize())
    # ic(img.GetOrigin(),img.GetSpacing(),img.GetDirection(),img.GetSize())
    """