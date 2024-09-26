#!/bin/bash
NUM_PROC=$1

CONFIG="/home/user/Documents/repos/pytorch-image-models/configs/cartrack/license_plates/convnext.yml"

# train tiny model, single train
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=$NUM_PROC train.py -c $CONFIG



