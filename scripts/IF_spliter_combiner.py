import pandas as pd
import os
import re
import argparse
from pathlib import Path
import glob

# import other python files
import IF_markers_spliter as split
import IF_patients_combiner as comb



def get_arguments():
    parser = argparse.ArgumentParser(description="Process IF CSV files to identify each markers")
    parser.add_argument('--file', type = str, help = 'file that has to be processed')
    parser.add_argument('--dir', type = str, help = 'folder where files have to be processed')

    args = parser.parse_args()

    return args




def main():

    directory = './data/true_data/IF_anonymized_preprocessed/IF_C1'
    
    args = get_arguments()

    if args.file:
        print('We process file: {}'.format(args.file))
        #process_csv(args.file)
        print("process finished")

    elif args.dir:
        comb.process_csv(comb.patient_files(args.dir))
        for csv_path in glob.glob(os.path.join(args.dir,'*.csv')):       
            split.split_csv(csv_path)
    


if (__name__== "__main__"):
    main()