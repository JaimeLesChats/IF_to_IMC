#!/bin/bash

input_dir=$(yq '.data_linking.mcd_converter.input_dir_raw_mcd' ./scripts/config.yaml)
output_dir=$(yq '.data_linking.mcd_converter.output_dir_raw_tiff' ./scripts/config.yaml)
patient=$(yq '.data_linking.mcd_converter.patient' ./scripts/config.yaml)
all_patients=$(yq '.data_linking.mcd_converter.all_patients' ./scripts/config.yaml)
flush=$(yq '.data_linking.mcd_converter.flush' ./scripts/config.yaml)

if [ "$flush" = "true" ]; then
    rm -r "$output_dir"
fi

if [ "$all_patients" = "true" ]; then
    for file in "$input_dir"/*.mcd; do   
        python3 ./scripts/data_linking/mcd_converter/mcd_img_converter.py --file "$file" 
    done
else
    for file in "$input_dir"/*"$patient"*.mcd; do
        python3 ./scripts/data_linking/mcd_converter/mcd_img_converter.py --file "$file"
    done
fi


