from imctools.io.mcd.mcdparser import McdParser
import os 
from pathlib import Path
import re
import argparse
import matplotlib.pyplot as plt
import tifffile
import numpy as np
import yaml
import sys

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

    args = parser.parse_args()

    return args 


def get_output_path(patient, roi, combo):

    base_imc_dir = Path(config['data_linking']['mcd_converter']['output_dir_raw_tiff'])

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

    all_markers = config['data_linking']['mcd_converter']['all_markers']
    markers_combo = config['data_linking']['mcd_converter']['markers_combo']
    markers_list = config['data_linking']['mcd_converter']['markers_list']
    all_rois = config['data_linking']['mcd_converter']['all_rois']

    patient_id = get_patient_name(input_file)
    #patient_id = os.path.splitext(os.path.basename(input_file))[0] 

    sys.stdout.write(f'\n[Patient : {patient_id}] File Acquisition...')

    parser = McdParser(input_file)
    session = parser.session
    rois = session.acquisition_ids

    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write(f'[Patient : {patient_id}] File successfully imported !\n')

    for roi in rois:
        # the exact output you're looking for:

        sys.stdout.write(f'[Patient : {patient_id}][ROI : {roi}] Acquisition ...')

        ac_data = parser.get_acquisition_data(roi)

        if all_markers:
            channels_selected = ac_data.channel_names
        else:
            channels_selected = re.split(r',', markers_list)

            if all(item in ac_data.channel_names for item in channels_selected):
                sys.stdout.write('\r')
                sys.stdout.flush()
                sys.stdout.write(f'[Patient : {patient_id}][ROI : {roi}] All channels available !')
            else:
                sys.stdout.write('\r')
                sys.stdout.flush()
                sys.stdout.write(f'[Patient : {patient_id}][ROI : {roi}] ERROR ! Wrong channels !\n')
                parser.close()
                exit()

        if markers_combo:
            
            sys.stdout.write('\r')
            sys.stdout.flush()
            sys.stdout.write(f'[Patient : {patient_id}][ROI : {roi}] Converting combo of {len(channels_selected)} channels...')

            if all_markers:
                combo_name = 'all_channels'
            else:
                combo_name = '_'.join(channels_selected)
            file_name = patient_id  + '_' + combo_name + '_' + str(roi) + '.ome.tiff'
            output_path = get_output_path(patient_id,roi,markers_combo).joinpath(file_name)

            img_stack = np.stack([ac_data.get_image_by_name(ch) for ch in channels_selected], axis=0)  # shape: (selected_channels, H, W)

            # Convert to (1, 2, 1, H, W) => (T, C, Z, Y, X) for OME-TIFF
            img_stack = img_stack[np.newaxis, :, np.newaxis, :, :]

            # Set axes and channel names
            metadata = {'axes': 'TCZYX','Channel': {'Name': channels_selected}}

            # Save as OME-TIFF
            tifffile.imwrite(output_path,img_stack.astype(np.float32),metadata=metadata,ome=True)

            sys.stdout.write('\r')
            sys.stdout.flush()
            sys.stdout.write(f'[Patient : {patient_id}][ROI : {roi}] DONE ! Combo of {len(channels_selected)} channels converted !\n')

        else:
            for i,channel in enumerate(channels_selected):
                image = ac_data.get_image_by_name(channel)
                file_name = patient_id  + '_' + channel + '_' + str(roi) + '.tiff'

                output_path = get_output_path(patient_id,roi,markers_combo).joinpath(file_name)
                tifffile.imwrite(output_path, image.astype(np.uint16), photometric='minisblack')
                sys.stdout.write('\r')
                sys.stdout.flush()
                sys.stdout.write(f'[Patient : {patient_id}][ROI : {roi}] {i+1}/{len(channels_selected)} channels converted ...')

            sys.stdout.write('\r')
            sys.stdout.flush()
            sys.stdout.write(f'[Patient : {patient_id}][ROI : {roi}] DONE ! {len(channels_selected)} channels converted !\n')

        if not(all_rois):
            # stop if we didn't want all rois
            break
        
    parser.close()


def main(): 

    args = get_arguments()

    # Bash arguments

    if args.file:
        convert_mcd_to_tiff(args.file)

if (__name__== "__main__"):
    main()