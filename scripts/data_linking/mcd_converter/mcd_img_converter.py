from imctools.io.mcd.mcdparser import McdParser
import os 
from pathlib import Path
import re
import argparse
import matplotlib.pyplot as plt
import tifffile
import numpy as np
import yaml

'''
data_linking:
  mcd_converter:
    input_dir_raw_mcd: '/home/matthieu.bernard/Documents/IF_to_IMC/data/raw/imc_data'
    output_dir_raw_tiff: '/home/matthieu.bernard/Documents/IF_to_IMC/data/raw/imc_data'
    all_patients: false
    patient: '15T011146-16' # if all_patients: false
    all_rois: false # if false, it will make a file for only the first roi
    markers_combo: false
    markers_combo_list: '[]'
    all_markers: true # if markers_combo: false
    markers_list: '[]' # will make different files for each marker 
'''

def get_config():
    with open("./scripts/config.yaml",'r') as f:
        config = yaml.safe_load(f)
    return config

config = get_config()


def get_arguments():
    """
    Get arguments when launching the python script
    
    """
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--file', type = str, help = "Path to the .mcd file")
    parser.add_argument('--out', type = str, help = "Directory where the .ome.tiff file will be saved")
    parser.add_argument('-c', type = str, help = "Select the channels to use by imctools")
    parser.add_argument('--am', action= 'store_true', help = "Do an image for each marker")
    parser.add_argument('--ar', action= 'store_true', help = "Do an image for each roi, else the first roi")

    args = parser.parse_args()

    return args 


def get_output_path(patient, roi, combo):
    base_imc_dir = Path('./data/raw/imc_data')

    patient_dir = base_imc_dir.joinpath(patient)
    roi_dir = patient_dir.joinpath('roi_' + str(roi))
    if combo:  
        final_dir = roi_dir.joinpath('combos')
    else:
        final_dir = roi_dir
    if not os.path.exists(final_dir):
            os.makedirs(final_dir)
    
    return final_dir
        

def get_patient_name(input_file):
    basename = os.path.basename(input_file)

    dash_positions = [m.start() for m in re.finditer('-', basename)]
    underscore_positions = [m.start() for m in re.finditer('_', basename)]

    if len(dash_positions) < 2 or not underscore_positions:
        raise ValueError("Not enough '-' or '_' characters in string")

    second_last_dash = dash_positions[-2]
    last_underscore = underscore_positions[-1]

    return basename[second_last_dash+1:last_underscore]


def convert_mcd_to_tiff(input_file):
    print('############# Conversion ############\n\nInitializing ...')

    all_markers = config['data_linking']['mcd_converter']['all_markers']
    markers_combo = config['data_linking']['mcd_converter']['markers_combo']
    markers_list = config['data_linking']['mcd_converter']['markers_list']
    all_rois = config['data_linking']['mcd_converter']['all_rois']

    patient_id = get_patient_name(input_file)
    #patient_id = os.path.splitext(os.path.basename(input_file))[0]

    parser = McdParser(input_file)
    session = parser.session
    rois = session.acquisition_ids

    for roi in rois:
        print(f'ROI {roi} \n')

        ac_data = parser.get_acquisition_data(roi)
        print(ac_data.image_data.shape) 
        data = ac_data.get_image_by_name('Ir191')
        print(data.shape)

        if all_markers:
            channels_selected = ac_data.channel_names
        else:
            channels_selected = re.split(r',', markers_list)
            if all(item in ac_data.channel_names for item in channels_selected):
                print('All channels are available !')
            else:
                print('[!] Wrong channel names !\nCheck config file for potential mistakes...')

        if markers_combo:
            combo_name = '_'.join(channels_selected)
            file_name = patient_id  + '_' + combo_name + '_' + str(roi) + '.tiff'

            output_path = get_output_path(patient_id,roi,markers_combo).joinpath(file_name)
            ac_data.save_tiff(output_path, names=channels_selected, compression=0)
        else:
            for channel in channels_selected:
                #channel = marker
                image = ac_data.get_image_by_name(channel)
                file_name = patient_id  + '_' + channel + '_' + str(roi) + '.tiff'

                output_path = get_output_path(patient_id,roi,markers_combo).joinpath(file_name)
                tifffile.imwrite(output_path, image.astype(np.uint16), photometric='minisblack')
                print("{} channel converted !".format(channel))

        if not(all_rois):
            # stop if we didn't want all rois
            break
        
    parser.close()


# save multiple standard TIFF files in a folder
#ac_data.save_tiffs("/home/anton/tiffs", compression=0, bigtiff=False)



def main(): 

    args = get_arguments()

    # Bash arguments

    if args.file:
        convert_mcd_to_tiff(args.file)

if (__name__== "__main__"):
    main()