
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
        with sf.SoundFile(file) as f:
            originalsubtype = f.subtype
        
        y_resampled = np.array([
            librosa.resample(channel, orig_sr=orig_sr, target_sr=target_sr, res_type='kaiser_best')
            for channel in y
        ])

        
        y_resampled = y_resampled.T  
        output_path = file
        
        sf.write(output_path, y_resampled, target_sr, subtype=originalsubtype)
        
for file in tqdm(glob.glob(os.path.join(base_dir, "*.wav"))):
    resample(file)