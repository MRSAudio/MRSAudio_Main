# resample all wav stereo to 48kHz
import os
import glob
import json
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

base_dir = "./data/Spatial/drama_splitspker"

def resample(file, target_sr=48000):
    y, orig_sr = librosa.load(file, sr=None, mono=False)
    if orig_sr != 48000:
        print(f"文件采样率不符合要求：{file}，当前采样率：{orig_sr}")
        with sf.SoundFile(file) as f:
            originalsubtype = f.subtype
        # 分离左右声道并分别重采样
        y_resampled = np.array([
            librosa.resample(channel, orig_sr=orig_sr, target_sr=target_sr, res_type='kaiser_best')
            for channel in y
        ])

        # 合并声道并转置为 (samples, channels) 格式
        y_resampled = y_resampled.T  # 转置为 (samples, channels)
        output_path = file
        # 保存为 WAV 文件（soundfile 自动处理量化）
        sf.write(output_path, y_resampled, target_sr, subtype=originalsubtype)
        
for file in tqdm(glob.glob(os.path.join(base_dir, "*.wav"))):
    resample(file)