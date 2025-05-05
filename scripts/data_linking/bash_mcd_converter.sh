#!/bin/bash

input_dir=$(yq '.data_linking.input_dir_raw_mcd' ./scripts/config.yaml)
output_dir=$(yq '.data_linking.output_dir_raw_tiff' ./scripts/config.yaml)
patient=$(yq '.data_linking.patient' ./scripts/config.yaml)

for file in "$input_dir"/22-IMC-H-27_Cordelier-15T011146-16_ROI.mcd; do
    echo  "File $file"    
    python3 ./scripts/data_linking/mcd_img_converter.py --file $file --out $output_dir --am
done


