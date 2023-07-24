
from utils import*
import argparse
import glob
import sys
import os
from shutil import copyfile
from icecream import ic
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import time


LABEL_TO_REMOVE = []

def main(args):

    print("Reading folder : ", args.input_dir)
    print("Selected spacings : ", args.spacing)

    patients = {}
    		
    normpath = os.path.normpath("/".join([args.input_dir, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        basename = os.path.basename(img_fn)

        if True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz",".fcsv",".json"]]:
            #Identifying the patient id
            file_name = basename.split(".")[0]
            elements = file_name.split("_")
            patient = file_name.split("_Or")[0].split("_lm")[0].split("_scan")[0].split("_Scan")[0].split("_Cropped")[0]
            
            if patient not in patients.keys():
                patients[patient] = {"dir": os.path.dirname(img_fn)}


            if True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                patients[patient]["scan"] = img_fn

            elif True in [ext in img_fn for ext in [".fcsv",".json"]]:
                if "fid" not in patients[patient].keys():
                    patients[patient]["fid"] = [img_fn]
                else:
                    patients[patient]["fid"].append(img_fn)
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
        if "fid" not in data.keys():
            print("Missing landmark for patient :",patient,"at",data["dir"])
            error = True

    if error:
        print("ERROR : folder have missing files", file=sys.stderr)
        raise


    #print(patients)
    nb_worker = args.nb_worker

    nb_patients_done = mp.Manager().list([0 for i in range(nb_worker)])
    nb_patients = len(patients)
    check = mp.Process(target=CheckProgress, args=(nb_patients_done, nb_patients))
    check.start()

    key_split = np.array_split(list(patients.keys()),nb_worker)

    processes = [mp.Process(target=InitScan,args=(args,{key:patients[key] for key in key_split[i]},nb_patients_done,i)) for i in range(nb_worker)]

    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

def CheckProgress(nb_patients_done, nb_patients):
    """Check the progress of the process within the different workers and print it in percentage"""
    # Use tqdm to display the progress
    for i in tqdm(range(nb_patients)):
        while sum(nb_patients_done) < i+1:
            time.sleep(1)

def InitScan(args, patients, shared_list, num_worker):
    for patient,data in patients.items():
        scan = data["scan"]

        # patient_dirname = os.path.basename(data["dir"]).split(" ")[0]
        ScanOutpath = os.path.normpath("/".join([args.out,patient]))#"_".join([data["dir"].split('/')[-2],data["dir"].split('/')[-1]])]))

        # print(ScanOutpath)

        if not os.path.exists(ScanOutpath):
            os.makedirs(ScanOutpath)

        for sp in args.spacing:
            new_name = patient + "_scan_sp" + str(sp).replace(".","-") + ".nii.gz" #[data["dir"].split('/')[-2],data["dir"].split('/')[-1]])
            outpath = os.path.join(ScanOutpath,new_name)
            if not os.path.exists(outpath):
                if args.correct_histo:
                    CorrectHisto(scan, outpath,0.01, 0.99)
                SetSpacing(outpath,[sp,sp,sp],outpath)
        for lm in data["fid"]:
            copyfile(lm,os.path.join(ScanOutpath,patient+'_lm.mrk.json'))#"_".join([data["dir"].split('/')[-2],data["dir"].split('/')[-1]]) + "_lm" + os.path.basename(lm).split("_lm")[-1]))
            # copyfile(lm,os.path.join(,os.path.basename(lm))) #data["fid"],os.path.join(ScanOutpath,os.path.basename(data["fid"])))
        shared_list[num_worker] += 1
    
    #     lm = "fid"
    #     outLmPath = os.path.join(ScanOutpath,patient + "_lm_TMJ.mrk.json")
    #     if ".fcsv" in data[lm]:
    #         CorrectCSV(data[lm],Rlab=LABEL_TO_REMOVE)
    #         SaveJsonFromFcsv(data[lm],outLmPath)
    #     else:
    #         copyfile(data[lm],outLmPath)

    #     RenameLandmarkCSV(outLmPath,["AF","AE"])



    #     if patient in Left:
    #         fin = open(outLmPath, "rt")
    #         #read file contents to string
    #         data = fin.read()
    #         #replace all occurrences of the required string
    #         data = data.replace("U3", "UL3")
    #         data = data.replace("U1", "UL1")
    #         data = data.replace("U2", "UL2")
    #         data = data.replace("U3A", "UL3A")

    #         #close the input file
    #         fin.close()
    #         #open the input file in write mode
    #         fin = open(outLmPath, "wt")
    #         #overrite the input file with the resulting data
    #         fin.write(data)
    #         #close the file
    #         fin.close()

    #     elif patient in Right:
    #         fin = open(outLmPath, "rt")
    #         #read file contents to string
    #         data = fin.read()
    #         #replace all occurrences of the required string
    #         data = data.replace("U3", "UR3")
    #         data = data.replace("U1", "UR1")
    #         data = data.replace("U2", "UR2")
    #         data = data.replace("U3A", "UR3A")

    #         #close the input file
    #         fin.close()
    #         #open the input file in write mode
    #         fin = open(outLmPath, "wt")
    #         #overrite the input file with the resulting data
    #         fin.write(data)
    #         #close the file
    #         fin.close()

    #     fin = open(outLmPath, "rt")
    #     #read file contents to string
    #     data = fin.read()
    #     #replace all occurrences of the required string

    #     for name,rename in Rename.items():
    #         data = data.replace(name,rename)

    #     #close the input file
    #     fin.close()
    #     #open the input file in write mode
    #     fin = open(outLmPath, "wt")
    #     #overrite the input file with the resulting data
    #     fin.write(data)
    #     #close the file
    #     fin.close()


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Initialise data to be ready for training the CI landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('Input files')
    input_group.add_argument('-i','--input_dir', type=str, help='Input directory with 3D images',default='/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/ALLDATA')#required=True)

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('-o','--out', type=str, help='Output directory', default='/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/AIAIAIAIAIAIA')#required=True)
    output_params.add_argument('-ch','--correct_histo', type=bool, help='Is contrast adjustment needed', default=True)

    input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing', default=[0.3,1])

    input_group.add_argument('-nw','--nb_worker', type=int, help='Number of CPU workers', default=10)


    args = parser.parse_args()
    
    main(args)
