# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME

DATA_DIR=$EXP_HOME/data/bingrad_audio/trainset
SAVE_DIR=$EXP_HOME/checkpoints/single_audio
mkdir -p $SAVE_DIR
export MKL_THREADING_LAYER=GNU
export PYTHONPATH=${EXP_HOME}/src:$PYTHONPATH

CUDA_VISIBLE_DEVICES=4,5 python src/binauralgrad/train.py $SAVE_DIR $DATA_DIR --binaural-type leftright --params params_single_drama
