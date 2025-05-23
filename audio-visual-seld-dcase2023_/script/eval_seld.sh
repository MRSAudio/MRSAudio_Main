#!/bin/bash

params=(
    # "foa ./data_dcase2023_task3/model_monitor/202303xxxxxxxx/params_202303xxxxxxxx_0010000.pth"
    )

for param in "${params[@]}"; do
    echo "Evaluating model with parameters: $param"
    for batch_size in 16; do
        p=(${param});
        CUDA_VISIBLE_DEVICES=7 python seld.py \
        # -eval \
        # -evalwt ./data_dcase2023_task3/list_dataset/dcase2023t3_${p[0]}_devtest.txt \
        -em ${p[1]} \
        # --feature binaural_phasediff \
        # --net vis_transformer \
        -b ${batch_size};
    done
done
