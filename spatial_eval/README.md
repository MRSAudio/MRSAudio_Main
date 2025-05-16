# Spatial-AST

This repo is based on the code and models of "BAT: Learning to Reason about Spatial Sounds with Large Language Models" .

## Installation
```
conda env create -f environment.yml
bash timm_patch/patch.sh
```

## Preparation
## Checkpoint Preparation
Download the finetuned [checkpoint](https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialAST/finetuned.pth) and and save it as:
```
./ckpt/SpatialAST/finetuned.pth
```

## Data Preparation
Place the ground-truth data in:
```
./data/gt
```
Place the generated data in:
```
./data/infer
```

Make sure both folders contain the same number of files with identical file names.


## Eval
Extract Embeddings
```bash
bash scripts/eval.sh
```

Run Evaluation
```bash
python evaluate_cos.py
```