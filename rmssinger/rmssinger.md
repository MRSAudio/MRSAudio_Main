
## RMSSinger (Unofficial Implementation)

> Since [RMSSinger](https://arxiv.org/abs/2305.10686) does not provide official code or pretrained checkpoints, we implemented our own version of RMSSinger to serve as a baseline for the Singing Voice Synthesis (SVS) task.

---

## Environment Setup

Please follow the instructions below to set up the Python environment based on the provided `requirements.txt`:

```bash
conda create -n rmssinger python=3.8 -y
conda activate rmssing
pip install -r requirements.txt
```

---

## Data Preparation

* **Metadata Generation**:
  Follow the metadata generation script from \[GTSinger]. The following fields are required:

  * `ph`: phoneme sequence
  * `ph_durs`: phoneme durations
  * `note`: musical note sequence
  * `note_durs`: note durations
  * `singer`: singer ID
  * `wav_fn`: path to the audio file
  * `item_name`: unique item name (you may use the basename of the audio file if unique)
  * `txt`: original lyric text
  * `ph2word`: mapping from phonemes to words
  * `note2ph`: mapping from notes to phonemes

* **Phoneme Dictionary**:
  Use [Pypinyin](https://github.com/mozillazg/python-pinyin) to generate phonemes. A sample dictionary can be found in [`phoneme_set.json`](./phoneme_set.json).

* **Singer Dictionary**:
  The `spker_set.json` file (singer dictionary) can be generated from the metadata.

---

## Data Binarization

Run the following command to binarize your dataset:

```bash
cd rmssinger
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/runs/binarize.py --config singing/svs/config/mrssing.yaml
```

> Please ensure the paths and dataset split (train/valid/test) are correctly set in the configuration file.

---

## Training

Use the following command to start training:

```bash
cd rmssinger
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --config /path/to/your/config --exp_name /path/to/your/exp/dir --reset
```

---

## Inference

Run inference using:

```bash
cd rmssinger
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --config /path/to/your/config --exp_name /path/to/your/exp/dir --reset --infer
```
