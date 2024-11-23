#!/bin/bash

module load cuda-12.4.1-gcc-12.1.0

CONFIG_PATH="configs/llama_config.yaml"
MAIN_SCRIPT="main.py"

python $MAIN_SCRIPT inference --config $CONFIG_PATH
