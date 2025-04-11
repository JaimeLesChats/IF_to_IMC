import pandas as pd
import os
import re
import numpy as np
import argparse
import glob

def patient_files():
    directory = './data/true_data/acquired_csv_files'

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

def reindex_df(df,idx):
    return df.loc[1:,df.columns[0]].map(lambda x : x+idx)

def load_full_df_from_files(list_files):
    idx = 0
    df = pd.DataFrame()
    for i,file in enumerate(list_files):
        read_df = pd.read_csv(file)
        read_df.loc[1:,read_df.columns[0]] = reindex_df(read_df,idx)
        df = pd.concat([df,read_df],ignore_index=True)
        idx += len(read_df)


    df.columns.values[0]= "CellID"

    return df

        

def merge_patient_files(patient_files):
    for i, patient in enumerate(patient_files):
        # merged df from same patient
        patient_df = load_full_df_from_files(patient_files[patient])
        print("[{}] Patient {} files merged".format(i+1,patient))
    
        

merge_patient_files(patient_files())