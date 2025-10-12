# Copyright 2023 Sony Group Corporation.

import numpy as np
import pandas as pd
import soundfile as sf
import random
import json
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from util.func_seld_data_loader import select_time, get_label
from feature.feature import SpectralFeature

import librosa



def create_data_loader(args, object_detection):
    data_set = SELDDataSet(args, object_detection)
    return DataLoader(data_set, batch_size=args.batch_size, shuffle=True)


def wav_path2video_frame(wav_path, start_sec):
    mp4_path = wav_path.replace('mic', 'video_360x180').replace('foa', 'video_360x180').replace('binaural', 'video_360x180').replace('.wav', '.mp4')
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FPS) * start_sec))
    _, frame = cap.read()
    if frame is None:
        raise ValueError('video frame is None. wav_path: {}, start_sec: {}'.format(wav_path, start_sec))
    return frame


class SELDDataSet(Dataset):
    def __init__(self, args, object_detection):
        # wav on RAM
        self._args = args

        self._train_wav_list = []
        for train_wav in pd.read_table(self._args.train_wav_txt, header=None).values.tolist():
            self._train_wav_list.append(train_wav[0])

        # self._train_wav_dict = {}
        # self._time_array_dict = {}
        # for train_wav in pd.read_table(self._args.train_wav_txt, header=None).values.tolist():
        #     self._train_wav_dict[train_wav[0]] = sf.read(train_wav[0], dtype='float32', always_2d=True)
        #     real_csv = train_wav[0].replace('mic', 'metadata').replace('foa', 'metadata').replace('.wav', '.csv')
        #     self._time_array_dict[train_wav[0]] = pd.read_csv(real_csv, header=None).values

        with open(self._args.feature_config, 'r') as f:
            self._feature_config = json.load(f)

        self._object_detection = object_detection

    def __len__(self):
        return len(self._train_wav_list)  # dummy

    def __getitem__(self, idx):  # idx is dummy
        try:
            path, time_array, wav, fs, start = self._choice_wav(idx)
            input_wav = wav[start: start + round(self._args.train_wav_length * fs)]
            input_spec = self._wav2spec(input_wav)

            label = get_label(self._args.train_wav_length, time_array, start / fs, self._args.class_num)
            label_float = label.astype(np.float32)

            start_sec = start / fs
            frame = wav_path2video_frame(path, start_sec)
            frame_rgb_in = frame[:, :, [2, 1, 0]]  # BGR -> RGB
            box_in = self._object_detection.img2box(frame_rgb_in)
            frame_out = self._object_detection.box2dist(box_in)
            frame_out_float = frame_out.astype(np.float32)
        except Exception as e:
            print('Error in wav_path2video_frame: {}'.format(e))
            print('wav_path: {}'.format(path))
            print('start_sec: {}'.format(start_sec))
            print('wav_length: {}'.format(len(wav)))
            print('wav_path: {}'.format(path))
            print('wav_length: {}'.format(len(wav)))
            print('wav: {}'.format(wav))
            print('start: {}'.format(start))
            print('input_wav: {}'.format(input_wav))
            print('input_spec: {}'.format(input_spec))
            print('label: {}'.format(label))
            print('label_float: {}'.format(label_float))
            print('frame: {}'.format(frame))
            print('frame_rgb_in: {}'.format(frame_rgb_in))
            print('box_in: {}'.format(box_in))
            print('frame_out: {}'.format(frame_out))
            print('frame_out_float: {}'.format(frame_out_float))
            raise

        # input_spec = np.expand_dims(input_spec, axis=0)

        return input_spec, frame_out_float, label_float, '{}_{}'.format(path, start_sec)

    def _choice_wav(self, idx):
        path = self._train_wav_list[idx]
        wav_fs = sf.read(path, dtype='float32', always_2d=True)
        real_csv = path.replace('mic', 'metadata').replace('foa', 'metadata').replace('binaural', 'metadata').replace('.wav', '.csv')
        time_array = pd.read_csv(real_csv, header=None).values
        wav, fs = wav_fs
        if fs != self._args.sampling_frequency:
            wav = np.array([
                librosa.resample(channel, orig_sr=fs, target_sr=self._args.sampling_frequency)
                for channel in wav.T
            ]).T
            fs = self._args.sampling_frequency
        if len(wav) < self._args.train_wav_length * fs:
            raise ValueError('wav length is less than train_wav_length. wav_path: {}, wav_length: {}'.format(path, len(wav)))
        start = select_time(self._args.train_wav_length, wav, fs)
        return path, time_array, wav, fs, start

    def _wav2spec(self, input_wav):
        spec_feature = SpectralFeature(wav=input_wav,
                                       fft_size=self._args.fft_size,
                                       stft_hop_size=self._args.stft_hop_size,
                                       center=True,
                                       config=self._feature_config)
        if self._args.feature == 'amp_phasediff' or self._args.feature == 'binaural_phasediff':
            input_spec = np.concatenate((spec_feature.amplitude(),
                                         spec_feature.phasediff()))
        return input_spec
