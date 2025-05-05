from imctools.io.mcd.mcdparser import McdParser
import os 
from pathlib import Path
import re
import argparse
import matplotlib.pyplot as plt
import tifffile
import numpy as np



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


def get_output_path():
    pass

def get_patient_name(input_file):
    basename = os.path.basename(input_file)

    dash_positions = [m.start() for m in re.finditer('-', basename)]
    underscore_positions = [m.start() for m in re.finditer('_', basename)]

    if len(dash_positions) < 2 or not underscore_positions:
        raise ValueError("Not enough '-' or '_' characters in string")

    second_last_dash = dash_positions[-2]
    last_underscore = underscore_positions[-1]

    return basename[second_last_dash+1:last_underscore]


def convert_mcd_to_tiff(input_file, output_directory, selected_channel = None, all_channels = False, all_rois = False):
    print('############# Conversion ############\n\nInitializing ...')
    patient_id = get_patient_name(input_file)
    #patient_id = os.path.splitext(os.path.basename(input_file))[0]

    parser = McdParser(input_file)

    # imctools part
    xml = parser.get_mcd_xml()
    session = parser.session
    ids = parser.session.acquisition_ids
    for id in ids:
        print(f'ROI {id} \n')
        ac_data = parser.get_acquisition_data(id)
        #plt.imshow(ac_data.get_image_by_name('Ir191'))
        #plt.show()
        if all_channels:
            for channel in ac_data.channel_names:
                image = ac_data.get_image_by_name(channel)
                
                output_path = output_directory + '/' + patient_id  + '_' + channel + '_' + str(id) + '.tiff'
                tifffile.imwrite(output_path, image.astype(np.uint16), photometric='minisblack')
                print("{} channel converted !".format(channel))
        else:
            image = ac_data.get_image_by_name(selected_channel)

            output_path = output_directory + '/' + patient_id  + '_' + selected_channel + '_' + str(id) + '.tiff'
            tifffile.imwrite(output_path, image.astype(np.uint16), photometric='minisblack')
        if not(all_rois):
            break


# save multiple standard TIFF files in a folder
#ac_data.save_tiffs("/home/anton/tiffs", compression=0, bigtiff=False)



def main(): 

    args = get_arguments()

    # Bash arguments

    if args.file and args.out:
        convert_mcd_to_tiff(args.file,args.out,selected_channel=args.c,all_channels=args.am, all_rois = args.ar)

if (__name__== "__main__"):
    main()