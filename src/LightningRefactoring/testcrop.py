import SimpleITK as sitk
import numpy as np
from DataModule import LoadJsonLandmarks
import time
img = sitk.ReadImage('/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/Test/CP63_scan_sp0-3_HD.nii.gz')

# Zero padding to the image
resample = sitk.ResampleImageFilter()
resample.SetInterpolator(sitk.sitkLinear)
resample.SetOutputDirection(img.GetDirection())
resample.SetOutputOrigin(img.GetOrigin()+np.array([-10,-10,-10]))
resample.SetOutputSpacing(img.GetSpacing())
resample.SetSize((np.array(img.GetSize())*1.2).astype(int).tolist())
resample.SetDefaultPixelValue(0)
# img = resample.Execute(img)

# sitk.WriteImage(img,'/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/Test/CP63_zero.nii.gz')

lm = LoadJsonLandmarks('/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/Test/CP63_lm_MERGED.mrk.json','C2')
origin = np.array(img.GetOrigin())
spacing = np.array(img.GetSpacing())
size = np.array(img.GetSize())

transformed_lm = np.array(img.TransformPhysicalPointToContinuousIndex(lm))
    # (like downloading the dataset from the internet, or loading the dataset from the disk)
print(transformed_lm)
#print(ROI_Center)

center_image = (np.array(img.GetSize())/2.0).astype(int)

ROI_Center = transformed_lm.astype(int) 
# print(ROI_Center)
ROI_Size = np.array([180,180,180])

print(size)

Left = ROI_Center - ROI_Size//2
Right = ROI_Center + ROI_Size//2

print(Left)
print(Right)

img_cropped = img[ROI_Center[0]-ROI_Size[0]//2:ROI_Center[0]+ROI_Size[0]//2,
                  ROI_Center[1]-ROI_Size[1]//2:ROI_Center[1]+ROI_Size[1]//2,
                  ROI_Center[2]-ROI_Size[2]//2:ROI_Center[2]+ROI_Size[2]//2]

# sitk.WriteImage(img_cropped,'/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/Test/cropped_bis.nii.gz')