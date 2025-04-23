#!/bin/bash

input_directory='/home/matthieu.bernard/Documents/IF_to_IMC/data/raw/imc_data'
output_directory='/home/matthieu.bernard/Documents/IF_to_IMC/data/raw/imc_data'


for file in "$input_directory"/*.mcd; do
    echo  "File $file"    
    python3 ./scripts/data_linking/mcd_img_converter.py --file $file --out $output_directory -c Ir191
done


