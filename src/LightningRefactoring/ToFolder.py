import os
import sys
import glob

if __name__ == '__main__':
    data_dir = "/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/ALLDATA/Felicia"
    out_dir =  "/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/ALLDATA/OutFelicia"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    normpath = os.path.normpath("/".join([data_dir, '**', '*']))
    for img in glob.iglob(normpath, recursive=True):
        if os.path.isfile(img):
            filename = img.split('/')[-1]
            patient = filename.split('_Scanreg')[0].split('_Or')[0].split('_OR')[0]
            out_path = os.path.join(out_dir, patient)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            os.rename(img, os.path.join(out_path, filename))

