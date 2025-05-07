#!/bin/bash

input_dir=$(yq '.data_linking.mcd_converter.input_dir_raw_mcd' ./scripts/config.yaml)
output_dir=$(yq '.data_linking.mcd_converter.output_dir_raw_tiff' ./scripts/config.yaml)
patient=$(yq '.data_linking.mcd_converter.patient' ./scripts/config.yaml)


for file in "$input_dir"/mcd_files/22-IMC-H-27_Cordelier-15T011146-16_ROI.mcd; do
    echo  "File $file"    
    python3 ./scripts/data_linking/mcd_converter/mcd_img_converter.py --file $file 
done


