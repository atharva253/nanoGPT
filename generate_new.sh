#!/bin/sh
. ../miniconda3/bin/activate
conda activate deep-gen
python3 generate_new.py
# python3 train.py config/finetune_multitask.py