# Make-An-Audio 2 for Music Generation

This repository provides an implementation of **Make-An-Audio 2** for music generation tasks.

---

## 🚀 Quick Start

### 🔧 Environment Setup

Create a new Conda environment and install dependencies:

```bash
conda create -n maa2 python=3.10 -y
conda activate maa2
pip install -r requirements.txt
```

### 📦 Pretrained Models

Please refer to the github repo of [Make-An-Audio 2](http://github.com/bytedance/Make-An-Audio-2) to download pretrained models.

## 🎵 Data Preparation

Before training, you need to convert your dataset into a `.tsv` file with the following columns:

- `name`: unique identifier for each audio sample

- `dataset`: dataset name

- `audio_path`: full path to the .wav audio file

- `caption`: audio description or prompt

- `mel_path`: (optional) path to the precomputed mel spectrogram file

### 📝 Generate TSV File
If you're using the provided metadata, you can generate a `.tsv` file by running:

```bash
 python preprocess/convert_music_meta_to_tsv.py \
  --meta_path /path/to/your/meta/file \
  --audio_dir /path/to/your/audio_dir \
  --output-tsv /path/to/output.tsv
```

### 🎼 Generate Mel Spectrograms

Once the `.tsv` file is ready (containing at least `name`, `audio_path`, `dataset`, and `caption`), run the following command to compute mel spectrograms:

```
python ldm/data/preprocess/mel_spec.py \
  --tsv_path /path/to/your/tsv
```

Spectrograms will be saved to ./processed.

### ⏱️ Add Audio Durations (Optional)

If you'd like to add duration metadata to your `.tsv`, run:

```bash
python ldm/data/preprocess/add_duration.py
```

## 🧠 Training

### 1️⃣ Train the Variational Autoencoder (VAE)
To train a custom VAE model on your dataset (instead of using the pretrained model):

```bash
python main.py --base configs/train/vae.yaml -t --gpus 0,1,2,3
```

Make sure to modify the configuration file:

```
data.params.spec_dir_path: /path/to/your/tsv_directory
```

### 2️⃣ Train the Latent Diffusion Model

After training the VAE (or using a pretrained VAE), set the checkpoint path in the config:

```
model.params.first_stage_config.params.ckpt_path: /path/to/vae_checkpoint
```

Then start training the latent diffusion model:
```bash
python main.py --base configs/makeanaudio2.yaml -t --gpus 0,1,2,3
```

Training logs and model checkpoints will be saved to ./logs/.

## 🧪 Evaluation

For evaluation protocols and scripts, please refer to the original [Make-An-Audio Repo](https://github.com/Text-to-Audio/Make-An-Audio).