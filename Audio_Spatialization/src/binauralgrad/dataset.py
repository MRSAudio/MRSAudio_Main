# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import os
import random
import torch
import torchaudio

from glob import glob
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import json
import time

class BinauralConditionalDataset(torch.utils.data.Dataset):
  def __init__(self, paths, binaural_type="", predict_mean_condition=False):
    super().__init__()
    self.mono, self.binaural, self.binaural_geowarp, self.view = [], [], [], []
    self.binaural_type = binaural_type
    self.predict_mean_condition = predict_mean_condition
    self.item_names = []
    self.paths = paths
    self.chunk_size = int(0.8 * 48000)
    if self.chunk_size % 400 > 0:
      self.chunk_size = self.chunk_size + 400 - self.chunk_size % 400
    items=json.load(open(f'{paths}.json' ,'r'))
    error_list = ['20250402-203650-6-xhk-ldd-《两杆大烟枪》电影剧本_part2_缩混_cut_807.687_843.022_B']
    for item in items:
      if item['item_name'] in error_list:
        continue
      self.item_names.append(item['item_name'])
    
    # for item_name in os.listdir(paths):
    #   tx_view = np.loadtxt(f"{paths}/{item_name}/tx_positions.txt")
    #   if tx_view.shape[0] * 400 > self.chunk_size:
    #     self.item_names.append(item_name)  

  def __len__(self):
    return len(self.item_names)

  def __getitem__(self, idx):
    idx = idx % len(self.item_names)
    item_name = self.item_names[idx]
    try:
      mono, _ = torchaudio.load(f"{self.paths}/{item_name}/mono.wav")
      orig_binaural, _ = torchaudio.load(f"{self.paths}/{item_name}/binaural.wav")
      binaural_geowarp, _ = torchaudio.load(f"{self.paths}/{item_name}/binaural_geowarp.wav")
      # receiver is fixed at origin in this dataset, so we only need transmitter view
      view = np.loadtxt(f"{self.paths}/{item_name}/tx_positions.txt").transpose().astype(np.float32)
      if mono.shape[1]<self.chunk_size:
        pad_audio = self.chunk_size - mono.shape[1]
        # 使用F.pad进行音频填充 (B, T) 格式
        mono = F.pad(mono, (0, pad_audio), mode='constant', value=0)           # (1, T) -> (1, chunk_size)
        binaural = F.pad(orig_binaural, (0, pad_audio), mode='constant', value=0)  # (2, T) -> (2, chunk_size)
        binaural_geowarp = F.pad(binaural_geowarp, (0, pad_audio), mode='constant', value=0)

        required_pos_length  = self.chunk_size // 400
        pad_pos = required_pos_length - view.shape[1]
        view = np.pad(view, [(0, 0), (0, pad_pos)], mode='edge')
      else:
        offset = random.randint(0, mono.shape[1]-self.chunk_size)
        mono = mono[:, offset:offset+self.chunk_size]
        binaural = orig_binaural[0:2, offset:offset+self.chunk_size]
        binaural_geowarp = binaural_geowarp[0:2, offset:offset+self.chunk_size]
        pos_offset = offset // 400
        pos_length = self.chunk_size // 400
        view = view[:, pos_offset:pos_offset+pos_length] 
      mean_condition = binaural[0:2, :].mean(0, keepdim=True)
      if not all(v.shape[1] == self.chunk_size for v in [mono, mean_condition, binaural, binaural_geowarp]):
        print(item_name, mono.shape[1], mean_condition.shape[1], binaural.shape[1], binaural_geowarp.shape[1], view.shape[1]) 
      return {
          'mono': mono,
          'binaural': binaural,
          'binaural_geowarp': binaural_geowarp,
          'view': view,
          'mean_condition': mean_condition,     
      }
    except Exception as e:
      # 打印详细的错误信息
      print(f"\nError processing item {item_name} (idx {idx}):")
      print(f"Error type: {type(e).__name__}")
      print(f"Error message: {str(e)}")
      return None  # 返回空值表示加载失败

# class BinauralConditionalDataset(torch.utils.data.Dataset):
#   def __init__(self, paths, binaural_type="", predict_mean_condition=False):
#     super().__init__()
#     self.mono, self.binaural, self.binaural_geowarp, self.view = [], [], [], []
#     self.binaural_type = binaural_type
#     self.predict_mean_condition = predict_mean_condition
#     for subject_id in range(8):
#       mono, _ = torchaudio.load(f"{paths}/subject{subject_id + 1}/mono.wav")
#       binaural, _ = torchaudio.load(f"{paths}/subject{subject_id + 1}/binaural.wav")
#       binaural_geowarp, _ = torchaudio.load(f"{paths}/subject{subject_id + 1}/binaural_geowarp.wav")
#       # receiver is fixed at origin in this dataset, so we only need transmitter view
#       tx_view = np.loadtxt(f"{paths}/subject{subject_id + 1}/tx_positions.txt").transpose()
#       self.mono.append(mono)
#       self.binaural.append(binaural)
#       self.binaural_geowarp.append(binaural_geowarp)
#       self.view.append(tx_view.astype(np.float32))
#     # ensure that chunk_size is a multiple of 400 to match audio (48kHz) and receiver/transmitter positions (120Hz)
#     self.chunk_size = 2000 * 48
#     if self.chunk_size % 400 > 0:
#       self.chunk_size = self.chunk_size + 400 - self.chunk_size % 400
#     # compute chunks
#     self.chunks = []
#     for subject_id in range(8):
#       last_chunk_start_frame = self.mono[subject_id].shape[-1] - self.chunk_size + 1
#       hop_length = int((1 - 0.5) * self.chunk_size)
#       for offset in range(0, last_chunk_start_frame, hop_length):
#         self.chunks.append({'subject': subject_id, 'offset': offset})    

#   def __len__(self):
#     return len(self.chunks)

#   def __getitem__(self, idx):
#     subject = self.chunks[idx]['subject']
#     offset = self.chunks[idx]['offset']
#     mono = self.mono[subject][:, offset:offset+self.chunk_size]
#     view = self.view[subject][:, offset//400:(offset+self.chunk_size)//400]    

#     binaural = self.binaural[subject][0:2, offset:offset+self.chunk_size]
#     binaural_geowarp = self.binaural_geowarp[subject][0:2, offset:offset+self.chunk_size]

#     mean_condition = self.binaural[subject][0:2, offset:offset+self.chunk_size].mean(0, keepdim=True)

#     return {
#         'mono': mono,
#         'binaural': binaural,
#         'binaural_geowarp': binaural_geowarp,
#         'view': view,
#         'mean_condition': mean_condition,     
#     }

class Collator:
  def __init__(self, params):
    self.params = params
  
  def collate_binaural(self, minibatch):
    minibatch = [record for record in minibatch if record is not None]
    valid_records = []
    clip_length = self.params.clip_length
    for record in minibatch:
      start_view = random.randint(0, record['mono'].shape[1] // 400 - clip_length // 400)
      start = start_view * 400
      end_view = start_view + clip_length // 400
      end = end_view * 400
      try:
        record['mono'] = record['mono'][:, start:end]
        record['mean_condition'] = record['mean_condition'][:, start:end]
        record['binaural'] = record['binaural'][:, start:end]
        record['binaural_geowarp'] = record['binaural_geowarp'][:, start:end]
        record['view'] = record['view'][:, start_view:end_view].T
        record['view'] = np.repeat(record['view'], 400, axis=0).T
        if all(v.shape[1] == clip_length for v in [record['mono'], record['mean_condition'], record['binaural'], record['binaural_geowarp'], record['view']]):
          valid_records.append(record)
        else:
          print(record['mono'].shape[1], record['mean_condition'].shape[1], record['binaural'].shape[1], record['binaural_geowarp'].shape[1], record['view'].shape[1]) 
      except Exception as e:
        print(f"Skipping record due to error: {str(e)}")
        continue
    if len(valid_records)==0:
      return {
          'mono': torch.zeros((5, 1, self.params.clip_length)),
          'mean_condition': torch.zeros((5, 1, self.params.clip_length)),
          'audio': torch.zeros((5, 2, self.params.clip_length)),
          'binaural_geowarp': torch.zeros((5, 2, self.params.clip_length)),
          'view': torch.zeros((5, 7, self.params.clip_length))
      }
    mono = np.stack([record['mono'] for record in valid_records if 'mono' in record])
    mean_condition = np.stack([record['mean_condition'] for record in valid_records if 'mean_condition' in record])
    binaural = np.stack([record['binaural'] for record in valid_records if 'binaural' in record])
    binaural_geowarp = np.stack([record['binaural_geowarp'] for record in valid_records if 'binaural_geowarp' in record])
    view = np.stack([record['view'] for record in valid_records if 'view' in record])

    assert binaural_geowarp.shape[0] == view.shape[0]

    return {
        'mono': torch.from_numpy(mono),
        'mean_condition': torch.from_numpy(mean_condition),
        'audio': torch.from_numpy(binaural),
        'binaural_geowarp': torch.from_numpy(binaural_geowarp),
        'view': torch.from_numpy(view),
    }

# def from_path(data_dirs, params, binaural_type="", is_distributed=False):
#   if binaural_type:
#     dataset = BinauralConditionalDataset(data_dirs[0], binaural_type, 
#       predict_mean_condition=getattr(params, "predict_mean_condition", False))
#   else:
#     raise ValueError("Unsupported binaural_type")
#   return torch.utils.data.DataLoader(
#       dataset,
#       batch_size=params.batch_size,
#       collate_fn=Collator(params).collate_binaural,
#       shuffle=not is_distributed,
#       num_workers=os.cpu_count(),
#       sampler=DistributedSampler(dataset) if is_distributed else None,
#       pin_memory=True,
#       drop_last=True)
def from_path(data_dirs, params, binaural_type="", is_distributed=False):
  if binaural_type:
    dataset = BinauralConditionalDataset(data_dirs[0], binaural_type, 
      predict_mean_condition=getattr(params, "predict_mean_condition", False))
  else:
    raise ValueError("Unsupported binaural_type")
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate_binaural,
      shuffle=not is_distributed,
      # num_workers=os.cpu_count(),
      num_workers=8,
      persistent_workers=True,
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)
if __name__ == '__main__':
  import binauralgrad.params as params_all
  params = params_all.params_single_drama
  data_dirs = './BinauralGrad/data/bingrad_drama'
  dataset = from_path(data_dirs, params, 'leftright', is_distributed=True)