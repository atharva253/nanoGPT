#!/bin/sh
. ../miniconda3/bin/activate
conda activate deep-gen
python3 finetune_multitask.py