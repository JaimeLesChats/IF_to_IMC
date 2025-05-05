import pandas as pd
import os
import re
import argparse
from pathlib import Path
import glob


def get_arguments():
    parser = argparse.ArgumentParser(description="Process IF CSV files to identify each markers")
    parser.add_argument('--file', type = str, help = 'file that has to be processed')
    parser.add_argument('--dir', type = str, help = 'folder where files have to be processed')

    args = parser.parse_args()

    return args

def patient_files(directory):
    """
    Return a dictionnary {patient_name1 = ['file1.csv,...'], ...}

    directory : directory where all patient files are
    
    """

    patient_files = {}

    for csv_path in glob.glob(os.path.join(directory,"*.csv")):
        base_name = os.path.basename(csv_path)
        patient_id = re.split(r'_',base_name)[0]
        if patient_id in patient_files:
            patient_files[patient_id].append(csv_path)
        else:
            patient_files[patient_id] = [csv_path]

    # Completed dictionnary in patient_files
    return patient_files

def merge_df_from_files(list_files):
    df = pd.DataFrame()
    for i,file in enumerate(list_files):
        read_df = pd.read_csv(file)
        df = pd.concat([df,read_df],ignore_index=True)

    df.columns.values[0]= "CellID"

    return df

def split_df(IF_df):

    cluster = IF_df[['Cluster']]
    markers_list = re.split(r'[-+]+',cluster.iloc[0,0])
    markers_list.pop(-1)

    def assign_marker_value(markers_str, marker_id):
        list_presence = re.findall(r'[+-]',markers_str)
        return (1 if list_presence[marker_id] == '+' else 0)

    for i,marker in enumerate(markers_list):
        s = pd.Series(IF_df['Cluster'].map(lambda x : assign_marker_value(x,marker_id = i)), name = marker)
        IF_df.insert(loc = 3,column = marker, value = s)

    IF_df = IF_df.drop('Cluster',axis=1)
    
    return IF_df

def write_to_csv(df, output_path):

    directory = os.path.dirname(output_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    df.to_csv(output_path,index = False) 
       # Setting index=False prevents saving the index column in the csv file
    return 0       

def process_csv(patient_files, output_dir):
    nb_patients = len(patient_files)
    for i, patient in enumerate(patient_files):
        print("#### Patient {}/{} ####".format(i+1,nb_patients))

        merged_df = merge_df_from_files(patient_files[patient])
        print("[MERGE] Patient {}: files merged".format(patient))

        split_merged_df = merged_df
        
        #split_merged_df = split_df(merged_df)
        #print("[SPLIT] Patient {}: file split".format(patient))


        base_name = patient + "_merged_split.csv"
        output_path = os.path.join(output_dir,base_name)
        write_to_csv(split_merged_df,output_path)
        print("--> Patient {}: CSV created !".format(patient))


def main():

    #directory = './data/true_data/IF_anonymized_preprocessed/IF_C1'   
    
    args = get_arguments()

    if args.dir:
        base_path = Path(args.dir)
        output_dir = base_path.parent/ str(base_path.name + '_split_merged')
        process_csv(patient_files(args.dir),output_dir)
    


if (__name__== "__main__"):
    main()