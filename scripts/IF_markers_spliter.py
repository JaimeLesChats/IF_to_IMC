import pandas as pd
import os
import re
import argparse
from pathlib import Path

def get_arguments():
    parser = argparse.ArgumentParser(description="Process IF CSV files to identify each markers")
    parser.add_argument('--file', type = str, help = 'file that has to be processed')

    args = parser.parse_args()

    return args

def write_to_csv(df, output_path):

    directory = os.path.dirname(output_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    df.to_csv(output_path,index = False) # Setting index=False prevents saving the index column in the csv file
    
    return 0


def split_csv(file):
    IF_df = pd.read_csv(Path(file))

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
    
    output_dir = './data/true_data//IF_anonymized_preprocessed/IF_C1_split_merged'
    base_name = os.path.basename(file)
    output_path = os.path.join(output_dir,base_name)

    write_to_csv(IF_df,output_path=output_path)

    print("[SPLIT]")
