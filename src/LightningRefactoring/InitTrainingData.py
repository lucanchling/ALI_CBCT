import SimpleITK as sitk
import numpy as np
import glob
import os
import multiprocessing as mp
import argparse
import sys
import shutil
import time
from tqdm import tqdm
from ManagJson import MergeJson
from CSVMaker import CSVForAllLandmarks

def resample_fn(img, target_size=[-1,-1,-1],target_spacing=None, outpath=None):
    output_size = target_size 
    target_origin = None
    fit_spacing = True
    iso_spacing = True
    center = True
    linear = True
    direction = True 
    
    if linear:
        InterpolatorType = sitk.sitkLinear
    else:
        InterpolatorType = sitk.sitkNearestNeighbor
    
    spacing = img.GetSpacing()  
    size = img.GetSize()

    output_origin = img.GetOrigin()
    output_size = [si if o_si == -1 else o_si for si, o_si in zip(size, output_size)]
    # print(output_size)

    if(fit_spacing):
        output_spacing = [sp*si/o_si for sp, si, o_si in zip(spacing, size, output_size)]
    else:
        output_spacing = spacing

    if(iso_spacing):
        output_spacing_filtered = [sp for si, sp in zip(target_size, output_spacing) if si != -1]
        # print(output_spacing_filtered)
        max_spacing = np.max(output_spacing_filtered)
        output_spacing = [sp if si == -1 else max_spacing for si, sp in zip(target_size, output_spacing)]
        # print(output_spacing)

    if(target_spacing is not None):
        output_spacing = target_spacing

    if(target_origin is not None):
        output_origin = target_origin

    if(center):
        output_physical_size = np.array(output_size)*np.array(output_spacing)
        input_physical_size = np.array(size)*np.array(spacing)
        output_origin = np.array(output_origin) - (output_physical_size - input_physical_size) / 2.0

    if(direction):
        output_direction = np.identity(3).flatten()
    else:
        output_direction = img.GetDirection()

    # print("Input size:", size)
    # print("Input spacing:", spacing)
    # print("Output size:", output_size)
    # print("Output spacing:", output_spacing)
    # Spacing.append(output_spacing)
    # print("Output origin:", output_origin)

    resampleImageFilter = sitk.ResampleImageFilter()
    resampleImageFilter.SetInterpolator(InterpolatorType)
    resampleImageFilter.SetOutputSpacing(output_spacing)
    resampleImageFilter.SetSize(output_size)
    resampleImageFilter.SetOutputDirection(output_direction)
    resampleImageFilter.SetOutputOrigin(output_origin)
    # resampleImageFilter.SetDefaultPixelValue(zeroPixel)
    
    if outpath is not None:
        sitk.WriteImage(resampleImageFilter.Execute(img), outpath)
    else:
        return resampleImageFilter.Execute(img)

def SpacingResample(img,output_spacing=[0.5,0.5,0.5],outpath=-1):
    """Resample the scan to a new size and spacing"""
    spacing = np.array(img.GetSpacing())
    size = np.array(img.GetSize())
    origin = np.array(img.GetOrigin())

    # print("Input spacing :", spacing)
    # print("Input size :", size)
    # print("Input origin :", origin)
    output_spacing = np.array(output_spacing)
    output_size = size * spacing / output_spacing
    output_size = np.ceil(output_size).astype(int)  # Image dimensions are in integers

    # Find the new origin
    output_physical_size = output_size * output_spacing
    input_physical_size = size * spacing
    output_origin = origin - (output_physical_size - input_physical_size) / 2.0

    # print("Output spacing :", output_spacing)
    # print("Output size :", output_size)
    # print("Output origin :", output_origin)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputOrigin(output_origin)
    resample.SetOutputSpacing(output_spacing.tolist())
    resample.SetOutputDirection(img.GetDirection())
    resample.SetSize(output_size.tolist())
    resample.SetInterpolator(sitk.sitkLinear)

    if outpath != -1:
        sitk.WriteImage(resample.Execute(img), outpath)
    else:
        return resample.Execute(img)

def GetPatients(data_dir):
    """Return the dictionary of patients with their scans and fiducials"""
    print("Reading folder : ", data_dir)

    patients = {}
    		
    normpath = os.path.normpath("/".join([data_dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        basename = os.path.basename(img_fn)

        if True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz",".fcsv",".json"]]:
            
            patient_name = basename.split("_Or")[0].split("_lm")[0].split('_scan')[0].split('_Scan')[0].split('_OR')[0].split('.')[0]

            patient = img_fn.split('/{}'.format(patient_name))[0].split('/')[-1] + '_' + patient_name
            
            if patient not in patients.keys():
                patients[patient] = {"dir": os.path.dirname(img_fn)}


            if True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                patients[patient]["scan"] = img_fn

            elif True in [ext in img_fn for ext in [".fcsv",".json"]]:
                if "lm" not in patients[patient].keys():
                    patients[patient]["lm"] = [img_fn]
                else:
                    patients[patient]["lm"] += [img_fn]

            else:
                print("----> Unrecognise fiducial file found at :", img_fn)

        elif os.path.isfile(img_fn) and "fcsv" in img_fn:
            print("----> Unrecognise file found at :", img_fn)

    print("Number of patients:",len(patients))
    
    error = False
    for patient,data in patients.items():
        if "scan" not in data.keys():
            print("Missing scan for patient :",patient,"at",data["dir"])
            error = True
        if "lm" not in data.keys():
            print("Missing landmark for patient :",patient,"at",data["dir"])
            error = True

    if error:
        print("ERROR : folder have missing files", file=sys.stderr)
        raise
    return patients

def CheckProgress(nb_patients_done, nb_patients):
    """Check the progress of the process within the different workers and print it in percentage"""
    # Use tqdm to display the progress
    for i in tqdm(range(nb_patients)):
        while sum(nb_patients_done) < i+1:
            time.sleep(1)

    
def InitScan(args, patients, shared_list, num_worker):

    for patient, data in patients.items():
        scan = data["scan"]

        OutPath = data["dir"].replace(args.data_dir, args.out_dir)

        if not os.path.exists(OutPath):
            os.makedirs(OutPath)
        
        # Parameters for the resampling
        low_res_size = [128, 128, 128]
        low_res_spacing = [1.8264, 1.8264, 1.8264]

        high_res_spacing = [1,1,1]#[0.3, 0.3, 0.3]

        LowResScanOutPath = os.path.join(OutPath,os.path.basename(scan).replace('.nii.gz','_LD.nii.gz'))
        HighResScanOutPath = os.path.join(OutPath,os.path.basename(scan).replace('.nii.gz','_HD.nii.gz'))
                
        # Resample the scan
        img = sitk.ReadImage(scan)
        resample_fn(img, target_size=low_res_size, target_spacing=low_res_spacing, outpath=LowResScanOutPath)   # Low resolution scan
        SpacingResample(img,output_spacing=high_res_spacing, outpath=HighResScanOutPath)    # High resolution scan
        
        # Copy the landmarks files
        for lm_file in data["lm"]:
            shutil.copy(lm_file, OutPath)
        
        shared_list[num_worker] += 1


def main(args):
    data_dir = args.data_dir

    nb_workers = args.nb_workers
    if nb_workers > mp.cpu_count():
        print("WARNING : Number of workers is higher than the number of CPU available. Using {} workers instead of {}".format(mp.cpu_count(), nb_workers))
        nb_workers = mp.cpu_count()-1

    # Get the list of patients with their scans and landmarks files
    patients = GetPatients(data_dir)

    # Create a process to count the number of patients done throughout the process
    nb_patients_done = mp.Manager().list([0 for i in range(nb_workers)])
    nb_patients = len(patients)
    check = mp.Process(target=CheckProgress, args=(nb_patients_done, nb_patients))
    print("Resampling scans for Training...")
    check.start()

    key_split = np.array_split(list(patients.keys()), nb_workers)

    # TEST
    # InitScan(args, {key: patients[key] for key in key_split[0]},nb_patients_done,0)
    
    processes = [mp.Process(target=InitScan, args=(args, {key: patients[key] for key in key_split[i]},nb_patients_done,i)) for i in range(nb_workers)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    check.join()

    # Merge the landmarks files
    MergeJson(data_dir=args.out_dir)

    # Generate the CSV file for training
    print("Generating CSV file for training...")
    CSVForAllLandmarks(args.out_dir)




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/ALLDATA', help='path to data directory')
    parser.add_argument('--out_dir', type=str, default='/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/TEST', help='path to out directory')
    parser.add_argument('-nw', '--nb_workers', type=int, default=1, help='number of CPU workers for multiprocessing')

    args = parser.parse_args()
    
    main(args)