import os
import sys
import glob
import shutil

if __name__ == '__main__':
    folder = 'JOAO-6'
    data_dir = "/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/ALLDATA/Selene/"+folder
    out_dir =  "/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/ALLDATA/OutSelene/"+folder
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    normpath = os.path.normpath("/".join([data_dir, '**', '*']))
    for img in glob.iglob(normpath, recursive=True):
        if os.path.isfile(img):
            filename = img.split('/')[-1]
            patient = filename.split('_Scanreg')[0].split('_Or')[0].split('_OR')[0].split('_ScanReg')[0]
            out_path = os.path.join(out_dir, patient)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            shutil.copy(img, os.path.join(out_path, filename))

