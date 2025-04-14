#!/bin/bash

folder="../../data/true_data/IF_anonymized_preprocessed/IF_C1_split_merged"            # Folder containing CSV files

for file in "$folder"/*.csv; do
  python3 count.py --file $file
  echo $file "done"
done
python3 duplicate.py


