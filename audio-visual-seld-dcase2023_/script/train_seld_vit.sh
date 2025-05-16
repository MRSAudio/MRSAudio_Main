#!/bin/bash

for batch_size in 16; do
    CUDA_VISIBLE_DEVICES=6 python seld.py \
    -train -val \
    --net vis_transformer \
    -twt ./data_dcase2023_task3/list_dataset/dcase2023t3_binaural_devtrain_audiovisual.txt \
    -valwt ./data_dcase2023_task3/list_dataset/dcase2023t3_binaural_devtest.txt \
    --feature binaural_phasediff \
    -b ${batch_size} \
    -s 500 -i 50000;
done
