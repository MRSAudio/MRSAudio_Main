from functools import partial
import os
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

from timm.models.layers import to_2tuple, trunc_normal_

from utils.stft import STFT, LogmelFilterBank

import librosa
import tqdm

def resample_audio(waveform, orig_sr, target_sr=32000):
    """将音频重采样至目标采样率"""
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr, 
            new_freq=target_sr,
            dtype=waveform.dtype
        )
        waveform = resampler(waveform)
    return waveform

def extract_feature(audio_path):
    waveform, orig_sr = torchaudio.load(audio_path)
    if orig_sr != 32000:
        waveform = resample_audio(waveform, orig_sr, target_sr=32000)
    sr = 32000
    assert waveform.shape[0] == 2, "必须为双声道音频"
    waveform = waveform.unsqueeze(0)  # (1, 2, samples)
    
    # Step 2: 应用STFT
    spectrogram_extractor = STFT(
        n_fft=1024, hop_length=320, win_length=1024, window='hann',
        center=True, pad_mode='reflect', freeze_parameters=True
    )
    
    logmel_extractor = LogmelFilterBank(
        sr=32000, n_fft=1024, n_mels=128, fmin=50, 
        fmax=14000, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True
    )
    
    B, C, T = waveform.shape
    # print(f'waveform shape: {waveform.shape}')

    # 生成STFT（关键修改：明确维度顺序）
    waveform_flat = waveform.reshape(B * C, T)
    real, imag = spectrogram_extractor(waveform_flat)  # (2, 513, T)
    
    # 分离左右声道（去除冗余维度）
    real_left = real[0]  # (513, T)
    imag_left = imag[0]
    real_right = real[1]
    imag_right = imag[1]

    # IPD计算（保持频率在前）
    phase_left = torch.atan2(imag_left, real_left)  # (513, T)
    phase_right = torch.atan2(imag_right, real_right)
    IPD = phase_right - phase_left  # (513, T)
    
    # ILD计算（直接使用现有STFT结果）
    mag_left = torch.sqrt(real_left**2 + imag_left**2)  # (513, T)
    mag_right = torch.sqrt(real_right**2 + imag_right**2)
    epsilon = 1e-10
    ILD = 20 * torch.log10((mag_right + epsilon) / (mag_left + epsilon))  # (513, T)

    # 添加批次维度（保持频率在前格式）
    IPD = IPD.squeeze(0)  # (1, 513, T)
    ILD = ILD.squeeze(0)  # (1, 513, T)
    # print(f"IPD's shape: {IPD.shape}")
    # print(f"ILD's shape: {ILD.shape}")
    
    # 应用Mel滤波器（维度对齐）
    melW = logmel_extractor.melW  # (128, 513)
    # print(f'melW shape: {melW.shape}')
    IPD_mel = torch.matmul(IPD, melW)  # (1, 128, T)
    ILD_mel = torch.matmul(ILD, melW)
    
    # 调整维度并转换为numpy
    IPD_mel = IPD_mel.squeeze(0).numpy()  # (T, 128)
    ILD_mel = ILD_mel.squeeze(0).numpy()
    
    # print(f"Final features shape: IPD {IPD_mel.shape}, ILD {ILD_mel.shape}")
    
    # 保存为npy文件
    output_path = audio_path.replace('.wav', '_feature.npy')
    np.save(output_path, {'IPD': IPD_mel, 'ILD': ILD_mel})
    
    return IPD_mel, ILD_mel

if __name__ == "__main__":
    audio_dir = "/home/panchanghao/2025-ICML/evaluation/test-data/test_FireRedTTS"
    audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    for audio_file in tqdm.tqdm(audio_files,desc='extarcting ipd and ild'):
        extract_feature(audio_file)
