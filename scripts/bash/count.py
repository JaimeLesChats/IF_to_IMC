import pandas as pd
import argparse
from pathlib import Path
import json
import os
import re

def get_arguments():
    parser = argparse.ArgumentParser(description="Process IF CSV files to identify each markers")
    parser.add_argument('--file', type = str)

    args = parser.parse_args()

    return args


def main():

    #directory = './data/true_data/IF_anonymized_preprocessed/IF_C1'   
    
    args = get_arguments()
    
    if args.file:
        if Path("dict.json").exists():
            with open("dict.json","r") as d:
                dict = json.load(d)
            
        else:        
            dict = {}
            with open("dict.json","w") as d:
                json.dump(dict,d)

        df = pd.read_csv(args.file)
        for row in df.iloc[:,0]:
            if row in dict:
                dict[row][0] += 1
                dict[row].append(re.split(r'_',Path(args.file).name)[0])
            else:
                dict[row] = [1,re.split(r'_',Path(args.file).name)[0]]

        with open("dict.json","w") as d:
            json.dump(dict,d)

            
    


if (__name__== "__main__"):
    main()