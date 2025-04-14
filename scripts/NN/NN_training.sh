#!/bin/bash

python ./scripts/NN/NN_model_trainer.py -n --ep 500

for i in {1..20}; do
    echo  "Iteration $i"    
    python ./scripts/NN/NN_model_trainer.py --ep 500
done

