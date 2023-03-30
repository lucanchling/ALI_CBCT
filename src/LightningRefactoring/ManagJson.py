import json 
import os
import glob
import argparse
import shutil 


def GetJsonFiles(data_dir):

    normpath = os.path.normpath("/".join([data_dir, '**', '']))
    json_file = [i for i in sorted(glob.iglob(normpath, recursive=True)) if i.endswith('.json')]

    return json_file


def MergeJson(data_dir,extension='MERGED'):
    """
    Create one MERGED json file per scans from all the different json files (Upper, Lower...)
    """

    normpath = os.path.normpath("/".join([data_dir, '**', '']))
    json_file = [i for i in sorted(glob.iglob(normpath, recursive=True)) if i.endswith('.json')]

    # ==================== ALL JSON classified by patient  ====================
    dict_list = {}
    for file in json_file:
        patient = '_'.join(file.split('/')[-3:-1])+'#'+file.split('/')[-1].split('.')[0].split('_lm')[0]+'_lm'
        if patient not in dict_list:
            dict_list[patient] = []
        dict_list[patient].append(file)

    # ==================== MERGE JSON  ====================``
    for key, files in dict_list.items():
        file1 = files[0]
        with open(file1, 'r') as f:
            data1 = json.load(f)
            data1["@schema"] = "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#"
        for i in range(1,len(files)):
            with open(files[i], 'r') as f:
                data = json.load(f)
            data1['markups'][0]['controlPoints'].extend(data['markups'][0]['controlPoints'])
        outpath = os.path.normpath("/".join(files[0].split('/')[:-1]))        # Write the merged json file
        with open(outpath+'/'+key.split('#')[1] + '_'+ extension +'.mrk.json', 'w') as f: 
            json.dump(data1, f, indent=4)

    # ==================== DELETE UNUSED JSON  ====================
    for key, files in dict_list.items():
        for file in files:
            if extension not in os.path.basename(file):
                os.remove(file)    

def GetDic(data_dir):
    json_file = GetJsonFiles(data_dir)

    dict_list = {} 
    for file in json_file:
        patient = '_'.join(file.split('/')[-3:-1])
        if patient not in dict_list:
            dict_list[patient] = []
        dict_list[patient].append(file)
    return dict_list

def MoveJsonToFolder(args):

    data_dir = args.data_dir

    dict_list = GetDic(data_dir)

    for key in dict_list.keys():
        for lm_file in dict_list[key]:
            shutil.copy(lm_file,os.path.join(args.out_dir,key,'_'.join([key,'lm',os.path.basename(lm_file).split('_lm_')[1]])))


def CleanJson(json_path):
    '''
    Remove labels that dont have any position attribute'''
    with open(json_path) as f:
        data = json.load(f)
    
    markups = data["markups"][0]["controlPoints"]

    new_markups = [markup for markup in markups if isinstance(markup['position'],list)]

    data["markups"][0]["controlPoints"] = new_markups

    with open(json_path,'w') as f:
        json.dump(data,f,indent=4)


def main(args):

    data_dir = args.data_dir

    json_files = GetJsonFiles(data_dir)

    for file in json_files:
        CleanJson(file)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',help='directory where json files to merge are',type=str,default='/home/luciacev/Desktop/Luc_Anchling/ALICBCT/TRAINING/data/Patients')#required=True)
    parser.add_argument('--out_dir',help='output',type=str,default='/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/ALIINIT')#required=True)
    parser.add_argument('--extension',help='extension of new merged json files',type=str,default='MERGED')
    args = parser.parse_args()

    main(args)
    