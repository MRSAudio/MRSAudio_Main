# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME

DATA_DIR=$EXP_HOME/data/bingrad_sing/trainset
SAVE_DIR=$EXP_HOME/checkpoints/single_sing
mkdir -p $SAVE_DIR
export MKL_THREADING_LAYER=GNU
export PYTHONPATH=${EXP_HOME}/src:$PYTHONPATH

CUDA_VISIBLE_DEVICES=6,7 python src/binauralgrad/train.py $SAVE_DIR $DATA_DIR --binaural-type leftright --params params_single_drama
