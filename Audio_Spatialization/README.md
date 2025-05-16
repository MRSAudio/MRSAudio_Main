## Install

Prepare environment:
```
bash prepare_env.sh
```

## Training and Inference
### Data preparation
Process the metadata and get the file format of MRSSpeech, MRSLife(MRSSound), MRSSing, MRSMusic.
```
python process_file_speech.py
python process_file_sound.py
python process_file_sing.py
python process_file_music.py
```

Generate geometric warpped binaural audio from the mono audio based on the position difference of left ear and right ear.
```
bash runs/geowarp_dataset.sh
```

split testset and trainset
```
python get_test_speech.py
python get_test_sound.py
python get_test_sing.py
python get_test_music.py
```
### Training
Train the MRSSpeech model:
```
bash runs/train_stage_single_speech.sh
```
Train the MRSSound model:
```
bash runs/train_stage_single_sound.sh
```
Train the MRSSing model:
```
bash runs/train_stage_single_sing.sh
```
Train the MRSMusic model:
```
bash runs/train_stage_single_music.sh
```
The batch size in `src/binauralgrad/params.py`.

### Inference
Inference and evaluate the MRSSpeech model:
```
bash test_speech.sh
```
Inference and evaluate the MRSSound model:
```
bash test_sound.sh
```
Inference and evaluate the MRSSing model:
```
bash test_sing.sh
```
Inference and evaluate the MRSMusic model:
```
bash test_music.sh
```
