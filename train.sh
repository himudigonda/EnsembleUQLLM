#!/bin/bash

module load cuda-12.4.1-gcc-12.1.0

CONFIG_PATH="configs/llama_config.yaml"
MAIN_SCRIPT="main.py"

for seed in {42..51}; do
  python $MAIN_SCRIPT train --config $CONFIG_PATH --seed $seed
done
