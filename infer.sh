#!/bin/bash

CONFIG_PATH="configs/llama_config.yaml"
MAIN_SCRIPT="main.py"

python $MAIN_SCRIPT inference --config $CONFIG_PATH
