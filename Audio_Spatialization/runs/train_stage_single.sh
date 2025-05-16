# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME

DATA_DIR=$EXP_HOME/data/trainset
SAVE_DIR=$EXP_HOME/checkpoints/single
mkdir -p $SAVE_DIR
export MKL_THREADING_LAYER=GNU
export PYTHONPATH=${EXP_HOME}/src:$PYTHONPATH
export NCCL_DEBUG=INFO       # 开启NCCL调试信息
export NCCL_DEBUG_SUBSYS=ALL # 输出所有子系统的日志
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/binauralgrad/train.py $SAVE_DIR $DATA_DIR --binaural-type leftright --params params_single
