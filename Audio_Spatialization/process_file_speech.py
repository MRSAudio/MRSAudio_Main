import math
import os
import glob
import json
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
def euler_to_quaternion(yaw_deg, pitch_deg, roll_deg):
    """
    将欧拉角（Z-X-Y顺序）转换为四元数。
    参数顺序：航向角(yaw, 绕Z轴), 俯仰角(pitch, 绕X轴), 滚转角(roll, 绕Y轴)，单位为度。
    返回四元数 (w, x, y, z)。
    """
    # 将角度转换为弧度
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    
    # 绕Z轴的旋转（yaw）
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)
    q_z = (cy, 0.0, 0.0, sy)  # (w, x, y, z)
    
    # 绕X轴的旋转（pitch）
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    q_x = (cp, sp, 0.0, 0.0)
    
    # 绕Y轴的旋转（roll）
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    q_y = (cr, 0.0, sr, 0.0)
    
    # 四元数乘法函数
    def multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return (w, x, y, z)
    
    # 按顺序相乘：q_y * q_x * q_z
    q_temp = multiply(q_x, q_z)
    q = multiply(q_y, q_temp)
    
    return q


meta_fn = 'data/_drama_metadata_all_with_prompt.json'
input_dir_wav = 'data/drama_splitspker'
input_dir_pos = 'data/drama_npy_new'
output_root = 'data/bingrad_drama'
fps = 120
rw, rx, ry, rz = euler_to_quaternion(0,0,0)
os.makedirs(output_root, exist_ok=True)
tgt_sr = 48000
items = json.load(open(meta_fn, 'r'))
for item in tqdm(items):
    output_dir = os.path.join(output_root, item)
    os.makedirs(output_dir, exist_ok=True)

    wav_fn = os.path.join(input_dir_wav, f'{item}.wav')
    # orig_sr = sf.info(wav_fn).samplerate
    binaural_audio, orig_sr = librosa.load(wav_fn, mono=False, sr=tgt_sr)  # 保持原始采样率

    sr = tgt_sr
    binaural_audio = binaural_audio.T
    clip_audio_samples = (binaural_audio.shape[0] // 400) * 400
    binaural_audio = binaural_audio[:clip_audio_samples, :]
    mono_audio = np.mean(binaural_audio, axis=1)

    sf.write(os.path.join(output_dir, 'binaural.wav'), binaural_audio, sr)
    sf.write(os.path.join(output_dir, 'mono.wav'), mono_audio, sr)

    position_length = clip_audio_samples // 400  # 确保为整数
    pos_fn = os.path.join(input_dir_pos, f'{item}_v.npy')
    position = np.load(pos_fn)[:,:7]
    
    # 插值处理
    T, D = position.shape
    L = int(position_length)  # 目标长度
    
    old_x = np.arange(T)
    new_x = np.linspace(0, T-1, L)  # 生成新的时间轴
    
    new_position = np.zeros((L, D))
    for d in range(D):
        new_position[:, d] = np.interp(new_x, old_x, position[:, d])
    
    txt_output_path = os.path.join(output_dir, 'tx_positions.txt')
    # 写入文件
    with open(txt_output_path, 'w') as f:
        for pos in new_position:
            line = f"{pos[0]:.7f} {pos[1]:.7f} {pos[2]:.7f} " \
                f"{pos[3]:.7f} {pos[4]:.7f} {pos[5]:.7f} {pos[6]:.7f}\n"
            f.write(line)
    
    txt_output_path = os.path.join(output_dir, 'rx_positions.txt')
    # 写入文件
    with open(txt_output_path, 'w') as f:
        for pos in new_position:
            line = f"{0:.7f} {0:.7f} {0:.7f} " \
                f"{rw:.7f} {rx:.7f} {ry:.7f} {rz:.7f}\n"
            f.write(line)
