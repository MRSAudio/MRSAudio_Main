EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME

TEST_DATA_DIR=$EXP_HOME/data/bingrad_music/testset
STAGE_DIR=$EXP_HOME/checkpoints/single_music
export MKL_THREADING_LAYER=GNU

export PYTHONPATH=${EXP_HOME}/src:$PYTHONPATH

#Inference stage two model based on the results of stage one.
OUTPUT_DIR=${STAGE_DIR}/weights
mkdir -p ${OUTPUT_DIR}

for subject_dir in "$TEST_DATA_DIR"/*/; do
  # 提取子目录名称（去掉路径和末尾的斜杠）
  subject_id=$(basename "$subject_dir")
  CUDA_VISIBLE_DEVICES=4 python src/binauralgrad/inference.py --fast ${STAGE_DIR}/weights.pt \
    --dsp_path ${TEST_DATA_DIR}/${subject_id} \
    --binaural_type leftright \
    -o ${OUTPUT_DIR}/${subject_id}.wav \
    --params params_single_drama
done

python metric.py ${OUTPUT_DIR} ${TEST_DATA_DIR}
