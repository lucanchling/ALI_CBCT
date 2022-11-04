import json 
import os
import glob

if __name__ == "__main__":
    data_dir = "/Users/luciacev-admin/Desktop/Luc_Anchling/DATA/ALICBCT/OUTPUT"

    # if os.path.exists(out_dir):
    #     os.system('rm -rf ' + out_dir)
    # os.mkdir(out_dir)
    
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
        for i in range(1,len(files)):
            with open(files[i], 'r') as f:
                data = json.load(f)
            data1['markups'][0]['controlPoints'].extend(data['markups'][0]['controlPoints'])
        outpath = os.path.normpath("/".join(files[0].split('/')[:-1]))        # Write the merged json file
        with open(outpath+'/'+key.split('#')[1] + '_MERGED.mrk.json', 'w') as f: #out_dir + '/' + key.split('#')[0].split('_dataset')[0] + '_' + key.split('#')[1] + '_MERGED.mrk.json', 'w') as f:
            json.dump(data1, f, indent=4)

    # ==================== DELETE UNUSED JSON  ====================
    for key, files in dict_list.items():
        for file in files:
            os.remove(file)    