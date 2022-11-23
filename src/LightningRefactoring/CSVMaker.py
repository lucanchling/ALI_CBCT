import csv
import os
import glob
from sklearn.model_selection import train_test_split
from icecream import ic
from DataModule import LoadJsonLandmarks
import argparse

GROUP_LABELS = {
    'CB' : ['Ba', 'S', 'N', 'RPo', 'LPo', 'RFZyg', 'LFZyg', 'C2', 'C3', 'C4'],

    'U' : ['RInfOr', 'LInfOr', 'LMZyg', 'RPF', 'LPF', 'PNS', 'ANS', 'A', 'UR3O', 'UR1O', 'UL3O', 'UR6DB', 'UR6MB', 'UL6MB', 'UL6DB', 'IF', 'ROr', 'LOr', 'RMZyg', 'RNC', 'LNC', 'UR7O', 'UR5O', 'UR4O', 'UR2O', 'UL1O', 'UL2O', 'UL4O', 'UL5O', 'UL7O', 'UL7R', 'UL5R', 'UL4R', 'UL2R', 'UL1R', 'UR2R', 'UR4R', 'UR5R', 'UR7R', 'UR6MP', 'UL6MP', 'UL6R', 'UR6R', 'UR6O', 'UL6O', 'UL3R', 'UR3R', 'UR1R'],

    'L' : ['RCo', 'RGo', 'Me', 'Gn', 'Pog', 'PogL', 'B', 'LGo', 'LCo', 'LR1O', 'LL6MB', 'LL6DB', 'LR6MB', 'LR6DB', 'LAF', 'LAE', 'RAF', 'RAE', 'LMCo', 'LLCo', 'RMCo', 'RLCo', 'RMeF', 'LMeF', 'RSig', 'RPRa', 'RARa', 'LSig', 'LARa', 'LPRa', 'LR7R', 'LR5R', 'LR4R', 'LR3R', 'LL3R', 'LL4R', 'LL5R', 'LL7R', 'LL7O', 'LL5O', 'LL4O', 'LL3O', 'LL2O', 'LL1O', 'LR2O', 'LR3O', 'LR4O', 'LR5O', 'LR7O', 'LL6R', 'LR6R', 'LL6O', 'LR6O', 'LR1R', 'LL1R', 'LL2R', 'LR2R'],

    'CI' : ['UR3OIP','UL3OIP','UR3RIP','UL3RIP'],

    'TMJ' : ['AF', 'AE']
}


ALL_LANDMARKS = [value for key, value in GROUP_LABELS.items()]
ALL_LANDMARKS = ALL_LANDMARKS[0] + ALL_LANDMARKS[1] + ALL_LANDMARKS[2] + ALL_LANDMARKS[3] #+ ALL_LANDMARKS[4]

def write_csv_from_dict(data_dir, filename, data, ldmk=None):
    with open(data_dir + '/' + filename, 'w') as f:
        writer = csv.writer(f)
        if ldmk == 'All':
            writer.writerow(['Patient', 'scan_path', 'landmark_path', 'Ba', 'S', 'N', 'RPo', 'LPo', 'RFZyg', 'LFZyg', 'C2', 'C3', 'C4','RInfOr', 'LInfOr', 'LMZyg', 'RPF', 'LPF', 'PNS', 'ANS', 'A', 'UR3O', 'UR1O', 'UL3O', 'UR6DB', 'UR6MB', 'UL6MB', 'UL6DB', 'IF', 'ROr', 'LOr', 'RMZyg', 'RNC', 'LNC', 'UR7O', 'UR5O', 'UR4O', 'UR2O', 'UL1O', 'UL2O', 'UL4O', 'UL5O', 'UL7O', 'UL7R', 'UL5R', 'UL4R', 'UL2R', 'UL1R', 'UR2R', 'UR4R', 'UR5R', 'UR7R', 'UR6MP', 'UL6MP', 'UL6R', 'UR6R', 'UR6O', 'UL6O', 'UL3R', 'UR3R', 'UR1R','RCo', 'RGo', 'Me', 'Gn', 'Pog', 'PogL', 'B', 'LGo', 'LCo', 'LR1O', 'LL6MB', 'LL6DB', 'LR6MB', 'LR6DB', 'LAF', 'LAE', 'RAF', 'RAE', 'LMCo', 'LLCo', 'RMCo', 'RLCo', 'RMeF', 'LMeF', 'RSig', 'RPRa', 'RARa', 'LSig', 'LARa', 'LPRa', 'LR7R', 'LR5R', 'LR4R', 'LR3R', 'LL3R', 'LL4R', 'LL5R', 'LL7R', 'LL7O', 'LL5O', 'LL4O', 'LL3O', 'LL2O', 'LL1O', 'LR2O', 'LR3O', 'LR4O', 'LR5O', 'LR7O', 'LL6R', 'LR6R', 'LL6O', 'LR6O', 'LR1R', 'LL1R', 'LL2R', 'LR2R','UR3OIP','UL3OIP','UR3RIP','UL3RIP','AF', 'AE'])
        else:
            writer.writerow(['Patient', 'scan_path', 'landmark_path'])
        for key, value in data.items():
            if ldmk == 'All':
                writer.writerow([key, value['img'], value['LM'],value['Ba'],value['S'], value['N'], value['RPo'], value['LPo'], value['RFZyg'], value['LFZyg'], value['C2'], value['C3'], value['C4'], value['RInfOr'], value['LInfOr'], value['LMZyg'], value['RPF'], value['LPF'], value['PNS'], value['ANS'], value['A'], value['UR3O'], value['UR1O'], value['UL3O'], value['UR6DB'], value['UR6MB'], value['UL6MB'], value['UL6DB'], value['IF'], value['ROr'], value['LOr'], value['RMZyg'], value['RNC'], value['LNC'], value['UR7O'], value['UR5O'], value['UR4O'], value['UR2O'],value[ 'UL1O'],value[ 'UL2O'],value[ 'UL4O'],value[ 'UL5O'],value[ 'UL7O'],value[ 'UL7R'],value[ 'UL5R'],value[ 'UL4R'],value[ 'UL2R'],value[ 'UL1R'],value[ 'UR2R'], value['UR4R'], value['UR5R'], value['UR7R'], value['UR6MP'], value['UL6MP'], value['UL6R'], value['UR6R'], value['UR6O'], value['UL6O'],value[ 'UL3R'], value['UR3R'], value['UR1R'],value['RCo'], value['RGo'], value['Me'], value['Gn'], value['Pog'], value['PogL'], value['B'], value['LGo'], value['LCo'], value['LR1O'], value['LL6MB'], value['LL6DB'], value['LR6MB'], value['LR6DB'], value['LAF'], value['LAE'], value['RAF'], value['RAE'], value['LMCo'], value['LLCo'], value['RMCo'], value['RLCo'], value['RMeF'], value['LMeF'], value['RSig'], value['RPRa'], value['RARa'], value['LSig'], value['LARa'], value['LPRa'], value['LR7R'], value['LR5R'], value['LR4R'], value['LR3R'], value['LL3R'], value['LL4R'], value['LL5R'], value['LL7R'], value['LL7O'], value['LL5O'], value['LL4O'], value['LL3O'], value['LL2O'], value['LL1O'], value['LR2O'], value['LR3O'], value['LR4O'], value['LR5O'], value['LR7O'], value['LL6R'], value['LR6R'], value['LL6O'], value['LR6O'], value['LR1R'], value['LL1R'], value['LL2R'], value['LR2R'],value['UR3OIP'],value['UL3OIP'],value['UR3RIP'],value['UL3RIP'],value['AF'], value['AE']])
            else:
                writer.writerow([key, value['img'], value['LM']])

# for item in RESULTS:
#     wr.writerow([item,])

def GenDict(data_dir):
    DATA = {}
        
    
    normpath = os.path.normpath("/".join([data_dir, '**', '*']))

    for img in glob.iglob(normpath, recursive=True):
        if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", 'gipl.gz']]:
            patient = '_'.join(img.split('/')[-3:-1]).split('_dataset')[0] + '_' + img.split('/')[-1].split('.')[0].split('_scan')[0].split('_Or')[0].split('_OR')[0] #
            if patient not in DATA:
                DATA[patient] = {}
            DATA[patient]['img'] = img

            
        if os.path.isfile(img) and True in [ext in img for ext in [".json"]]:
            if 'MERGED' in img:
                patient = '_'.join(img.split('/')[-3:-1]).split('_dataset')[0] + '_' + img.split('/')[-1].split('_lm_MERGED')[0].split('_Scanreg')[0].split('_Or')[0].split('_OR')[0]
                if patient not in DATA:
                    DATA[patient] = {}
                DATA[patient]['LM'] = img
                Landmarks = LoadJsonLandmarks(img).keys()
                is_Landmarks = [True if i in Landmarks else False for i in ALL_LANDMARKS]
                for i,lm in enumerate(ALL_LANDMARKS):
                    DATA[patient][lm] = is_Landmarks[i]


    return DATA

def main(data_dir, output_dir, landmark=None, csv_summary=False):
    # data_dir = args.data_dir

    data = GenDict(data_dir)
    # ic(len(data))

    if csv_summary:
        write_csv_from_dict(data_dir, 'alldata.csv',data,ldmk='All')

    # landmark = args.landmark
    if landmark is not None:

        databis  = {}
        for patient in data.keys():
            try:
                if data[patient][landmark] == True:
                    databis[patient] = {}
                    databis[patient] = data[patient]
            except KeyError:
                pass
        # ic(len(databis))
    else:
        databis = data
        
    lostcases = len(data)-len(databis)
    # ic(lostcases)
    
    train_ratio = 0.7
    validation_ratio = 0.1
    test_ratio = 0.2

    train, test = train_test_split(list(databis.keys()), test_size=1-train_ratio, random_state=42)   
    val, test = train_test_split(test, test_size=test_ratio/(test_ratio + validation_ratio),random_state=42) 
    
    
    # ic(len(train))
    # ic(len(val))
    # ic(len(test))

    # out_dir = args.out
    
    write_csv_from_dict(output_dir, 'train.csv', {k: databis[k] for k in train})
    write_csv_from_dict(output_dir, 'val.csv', {k: databis[k] for k in val})
    write_csv_from_dict(output_dir, 'test.csv', {k: databis[k] for k in test})

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='ALI CBCT Training')
    # parser.add_argument('--data_dir', help='Directory with all data', type=str,default='/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/Test')
    # parser.add_argument('--out',help='output directory with csv files',type=str, default='')
    # parser.add_argument('--landmark', help='Landmark that you want to train', type=str,default='S')#required=True)
    # parser.add_argument('--csv_sumup',help='to creat a csv file with scans and the different landmarks that they have',type=bool,default=False)

    # args = parser.parse_args()

    # for landmark in ALL_LANDMARKS:

    #     data_dir = '/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/Test'
    #     output_dir = data_dir + '/CSV/lm_' + landmark
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #         # print("=========================================")
    #         # print("For landmark:", landmark)
    #         main(data_dir, output_dir, landmark)
    
    data_dir = '/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/RESAMPLED'
    output_dir = data_dir + '/CSV/ALL'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    main(data_dir, output_dir, landmark=None)