import csv
import os
import glob
from sklearn.model_selection import train_test_split

GROUP_LABELS = {
    'CB' : ['Ba', 'S', 'N', 'RPo', 'LPo', 'RFZyg', 'LFZyg', 'C2', 'C3', 'C4'],

    'U' : ['RInfOr', 'LInfOr', 'LMZyg', 'RPF', 'LPF', 'PNS', 'ANS', 'A', 'UR3O', 'UR1O', 'UL3O', 'UR6DB', 'UR6MB', 'UL6MB', 'UL6DB', 'IF', 'ROr', 'LOr', 'RMZyg', 'RNC', 'LNC', 'UR7O', 'UR5O', 'UR4O', 'UR2O', 'UL1O', 'UL2O', 'UL4O', 'UL5O', 'UL7O', 'UL7R', 'UL5R', 'UL4R', 'UL2R', 'UL1R', 'UR2R', 'UR4R', 'UR5R', 'UR7R', 'UR6MP', 'UL6MP', 'UL6R', 'UR6R', 'UR6O', 'UL6O', 'UL3R', 'UR3R', 'UR1R'],

    'L' : ['RCo', 'RGo', 'Me', 'Gn', 'Pog', 'PogL', 'B', 'LGo', 'LCo', 'LR1O', 'LL6MB', 'LL6DB', 'LR6MB', 'LR6DB', 'LAF', 'LAE', 'RAF', 'RAE', 'LMCo', 'LLCo', 'RMCo', 'RLCo', 'RMeF', 'LMeF', 'RSig', 'RPRa', 'RARa', 'LSig', 'LARa', 'LPRa', 'LR7R', 'LR5R', 'LR4R', 'LR3R', 'LL3R', 'LL4R', 'LL5R', 'LL7R', 'LL7O', 'LL5O', 'LL4O', 'LL3O', 'LL2O', 'LL1O', 'LR2O', 'LR3O', 'LR4O', 'LR5O', 'LR7O', 'LL6R', 'LR6R', 'LL6O', 'LR6O', 'LR1R', 'LL1R', 'LL2R', 'LR2R'],

    'CI' : ['UR3OIP','UL3OIP','UR3RIP','UL3RIP'],

    'TMJ' : ['AF', 'AE']
}
ALL_LANDMARKS = [value for key, value in GROUP_LABELS.items()]

def write_csv_from_dict(data_dir, filename, data):
    with open(data_dir + '/' + filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Patient', 'scan_path', 'landmark_path'])

        for key, value in data.items():
            writer.writerow([key, value['img'], value['LM']])

def GenDict(data_dir):
    DATA = {}

    normpath = os.path.normpath("/".join([data_dir, '**', '*']))

    for img in glob.iglob(normpath, recursive=True):
        if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", 'gipl.gz']]:
            patient = '_'.join(img.split('/')[-3:-1]).split('_dataset')[0] + '_' + img.split('/')[-1].split('.')[0].split('_scan')[0] #
            if patient not in DATA:
                DATA[patient] = {}
            DATA[patient]['img'] = img
            
        if os.path.isfile(img) and True in [ext in img for ext in [".json"]]:
            if 'MERGED' in img:
                patient = '_'.join(img.split('/')[-3:-1]).split('_dataset')[0] + '_' + img.split('/')[-1].split('_lm_MERGED')[0]
                if patient not in DATA:
                    DATA[patient] = {}
                DATA[patient]['LM'] = img
    return DATA

if __name__ == '__main__':
    data_dir = '/Users/luciacev-admin/Desktop/Luc_Anchling/DATA/ALICBCT/Test'
    
    data = GenDict(data_dir)

    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.1

    train, test = train_test_split(list(data.keys()), test_size=0.2, random_state=42)   
    val , test = train_test_split(test, test_size=test_ratio/(test_ratio + validation_ratio),random_state=42) 
    
    write_csv_from_dict(data_dir, 'train.csv', {k: data[k] for k in train})
    write_csv_from_dict(data_dir, 'test.csv', {k: data[k] for k in test})
    write_csv_from_dict(data_dir, 'val.csv', {k: data[k] for k in val})

    # train,val = train_test_split(list(data.keys()), test_size=0.2, random_state=42)
    # train_data = {k: data[k] for k in train}
    # val_data = {k: data[k] for k in val}

    # write_csv_from_dict(data_dir, 'train.csv', train_data)
    # write_csv_from_dict(data_dir, 'val.csv', val_data)