# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import librosa
import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation as R
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import soundfile
from src.binauralgrad.warping import GeometricTimeWarper, MonotoneTimeWarper
from tqdm import tqdm
import multiprocessing
import os
import glob

class GeometricWarper(nn.Module):
    def __init__(self, sampling_rate=48000):
        super().__init__()
        self.warper = GeometricTimeWarper(sampling_rate=sampling_rate)

    def _3d_displacements(self, view):
        # offset between tracking markers and ears in the dataset
        left_ear_offset = th.Tensor([0, -0.11, 0]).cuda() if view.is_cuda else th.Tensor([0, -0.11, 0])
        right_ear_offset = th.Tensor([0, 0.11, 0]).cuda() if view.is_cuda else th.Tensor([0, 0.11, 0])
        # compute displacements between transmitter mouth and receiver left/right ear
        displacement_left = view[:, 0:3, :] - left_ear_offset[None, :, None]
        displacement_right = view[:, 0:3, :] - right_ear_offset[None, :, None]
        displacement = th.stack([displacement_left, displacement_right], dim=1)
        return displacement

    def _warpfield(self, view, seq_length):
        return self.warper.displacements2warpfield(self._3d_displacements(view), seq_length)

    def forward(self, mono, view):
        '''
        :param mono: input signal as tensor of shape B x 1 x T
        :param view: rx/tx position/orientation as tensor of shape B x 7 x K (K = T / 400)
        :return: warped: warped left/right ear signal as tensor of shape B x 2 x T
        '''
        return self.warper(th.cat([mono, mono], dim=1), self._3d_displacements(view))

def load_position(position_fn):
    position_list = []
    with open(position_fn, 'r', encoding='utf-8') as infile:
        for line in infile.readlines():
            line = line.strip().split()
            assert len(line) == 7
            position_list.append([float(i) for i in line])
    return np.array(position_list)

def process_item(item):
    try:
        mono_fn = os.path.join(item, 'mono.wav')
        position_fn = os.path.join(item, 'tx_positions.txt')
        binaural_fn = os.path.join(item, 'binaural_geowarp.wav')
        # if os.path.exists(binaural_fn):
        #     return True, item  # 返回处理状态和项目名称
        
        mono_audio, sr = librosa.load(mono_fn, mono=True, sr=None)
        position_array = load_position(position_fn=position_fn)
        assert len(position_array) * 400 == len(mono_audio), f'{len(position_array)*400} {len(mono_audio)}'
        assert sr == 48000, f'采样率不对，是{sr}'
        position_array[:, [0, 1]] = position_array[:, [1, 0]]

        geometric_warper = GeometricWarper()
        dsp_result = geometric_warper(
            th.Tensor(mono_audio[None, None, :]),
            th.Tensor(position_array.transpose(1,0))[None, :, :]
        )
        
        soundfile.write(binaural_fn, dsp_result[0].numpy().transpose(1,0), 48000, 'PCM_16')
        return True, item  # 返回处理状态和项目名称
    except Exception as e:
        print(e)
        return False, (item, str(e))

if __name__ == '__main__':
    root = './data/bingrad_music'
    items = [d for d in glob.glob(f'{root}/*/*') if os.path.isdir(d)]
    
    # 创建进度条
    with multiprocessing.Pool() as pool, \
         tqdm(total=len(items), desc="Processing", unit="item") as pbar:
        
        # 使用imap_unordered保持处理顺序灵活性
        results = pool.imap_unordered(process_item, items)
        
        success_count = 0
        failure_log = []
        
        # 实时更新进度条
        for status, data in results:
            if status:
                success_count += 1
                pbar.set_postfix_str(f"Success: {success_count}, Failed: {len(failure_log)}")
            else:
                failure_log.append(data)
                pbar.set_postfix_str(f"Success: {success_count}, Failed: {len(failure_log)}")
            pbar.update(1)  # 更新进度条

    # 打印最终结果
    print(f"\nProcessing completed: {success_count} succeeded, {len(failure_log)} failed")
    if failure_log:
        print("\nFailed items:")
        for item, error in failure_log:
            print(f"- {item}: {error}")




