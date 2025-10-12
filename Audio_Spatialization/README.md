## Install

The codebase is benchmark code for Audio Spatialization in [BinauralGrad: A Two-Stage Conditional Diffusion Probabilistic Model for Binaural Audio Synthesis](https://arxiv.org/abs/2205.14807).
Please check the paper of BinauralGrad for details missing in this repository.

Prepare environment:
```
bash prepare_env.sh
```

## Training and Inference
### Data preparation
Process the metadata and get the file format of MRSSpeech, MRSLife(MRSSound), MRSSing, MRSMusic.
```
python process_dataset.py
```

Generate geometric warpped binaural audio from the mono audio based on the position difference of left ear and right ear.
```
bash runs/geowarp_dataset.sh
```

get split dataset
```
python get_dataset.py
```

### Training
Train the MRSSpeech model:
```
bash runs/train_stage_single.sh
```
The batch size in `src/binauralgrad/params.py`.

### Inference
Inference and evaluate the model:
```
bash train_stage_single.sh
```

