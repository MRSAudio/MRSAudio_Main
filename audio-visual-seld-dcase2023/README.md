# Audio-Visual Sound Event Localization and Detection
he Audio-Visual Sound Event Localization and Detection is mainly based on [STARSS23: An Audio-Visual Dataset of Spatial Recordings of Real Scenes with Spatiotemporal Annotations of Sound Events](https://github.com/sony/audio-visual-seld-dcase2023).

## File structure

```
audio-visual-seld-dcase2023_         This repository.
├── _audio_metadata_all_split_with_prompt.json  metadata file
├── dcase2022_task3_seld_metrics
├── feature
├── image
├── LICENSE
├── mmdetection
├── net
├── npy2csv.py                      generate csv file from npy file.
├── object_detection.py
├── para.py                         generate short video and foa/binaural clips
├── para_resize.py                  resize video
├── __pycache__
├── README.md
├── resample.py                     resample audio to 24kHz
├── resize_video.py
├── script
│   ├── eval_seld.sh                Evaluation script.
│   ├── train_seld_bin.sh           Training script for binaural input.
│   ├── train_seld_foa.sh           Training script for FOA input.
│   ├── train_seld_mic.sh
│   └── train_seld_vit.sh           Training script for vision transformer model.
├── seld_data_loader.py
├── seld_eval_dcase2022.py
├── seld_predictor.py
├── seld.py
├── seld_trainer.py
├── seld_validator.py
├── split.py                        
├── split_train_test.py               split test and train data.
├── util
├── vision_transformer.py              vision transformer model
└── wav_convertor.py
```

### Prerequisites
The provided system has been tested on linux, python 3.8.16, and pytorch 1.7.1.

You can install the requirements by running the following lines.
```
conda create --name audio_visual_seld_dcase2023 python=3.8 -y
conda activate audio_visual_seld_dcase2023
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y
```

You need to set up mmdetection for object detection model. (Ref: https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/)
```
pip install -U openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
mim install mmdet==2.28.1
mim download mmdet --config yolox_tiny_8x8_300e_coco --dest .
cd ..
```

After installation of mmdetection, you can continue to install with the below lines.
```
conda install tensorboard -y
conda install tqdm -y
conda install -c conda-forge pysoundfile -y
conda install -c conda-forge librosa -y
conda install ipython -y
conda install chardet -y
pip install moviepy==1.0.3
conda install pandas==1.5.3 -y
```

### Training trial
Before training, we need to convert HD mp4 files to low resolution mp4 files (360x180) for reduce of calculation cost.
```
python resize_video.py
```

After the conversion, run the script as shown below.
This will dump the logs and models in the `./data_dcase2023_task3/model_monitor/<DateTime>`.

You may modify the path for python on the script.
CPU memory needs 15GB and GPU memory 2GB.
The training takes around half a dozen hours.
```
bash script/train_seld_foa.sh
```

Microphone array version is the same way.
```
bash script/train_seld_mic.sh
```

The binaural version is the same way.
```
bash script/train_seld_bin.sh
```

The vision transformer version is the same way(with foa or binaural).
```
bash script/train_seld_vit.sh
```

You can check the training details using TensorBoard.

### Evaluation trial

The evaluation script is as shown below.
```
bash script/eval_seld.sh
```