#!/bin/bash

for batch_size in 16; do
    CUDA_VISIBLE_DEVICES=7 python seld.py \
    -train -val \
    -b ${batch_size} \
    -s 500 -i 50000;
done
