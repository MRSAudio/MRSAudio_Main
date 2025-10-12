from split import resample
import os
from tqdm import tqdm

base_dir = './audio-visual-seld-dcase2023/data_dcase2023_task3/binaural_dev'
for file in tqdm(os.listdir(base_dir)):
    if file.endswith('.wav'):
        wav_path = os.path.join(base_dir, file)
        resample(wav_path, 24000)
        # print(f"Resampled {wav_path} to 24kHz")