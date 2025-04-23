#!/bin/bash

input_dir=$(yq '.data_linking.input_dir_raw_mcd' ./scripts/config.yaml)
output_dir=$(yq '.data_linking.output_dir_raw_tiff' ./scripts/config.yaml)

for file in "$input_dir"/*.mcd; do
    echo  "File $file"    
    python3 ./scripts/data_linking/mcd_img_converter.py --file $file --out $output_dir -c Ir191
done


