import json
import os
import glob
import argparse

def main(args):
    data_dir = args.data_dir

    normpath = os.path.normpath("/".join([data_dir, '**', '*']))

    for file in glob.iglob(normpath, recursive=True):
        if os.path.isfile(file) and file.endswith(".json"):
            with open(file) as f:
                data = json.load(f)
                for i in range(len(data["markups"][0]["controlPoints"])):
                    data["markups"][0]["controlPoints"][i]["positionStatus"] = "defined"
            with open(file, 'w') as f:
                json.dump(data, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)