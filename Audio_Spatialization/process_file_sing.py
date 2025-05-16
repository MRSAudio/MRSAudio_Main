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

json_file = 'data/MRSSing/sing_stat_meta.json'
input_dir_wav = 'data/MRSSing'
output_root = 'data/bingrad_sing'
fps = 120
rw, rx, ry, rz = euler_to_quaternion(0,0,0)
tw, tx, ty, tz = euler_to_quaternion(180,0,0)
os.makedirs(output_root, exist_ok=True)
tgt_sr = 48000
items = json.load(open(json_file, 'r'))
wav_files = glob.glob(f'{input_dir_wav}/*/*/*/*/spatial/*.wav')
for item in tqdm(items):
    item_name = item['item_name']
    wavfile = item['audio_path']
    output_dir = os.path.join(output_root, item_name)
    os.makedirs(output_dir, exist_ok=True)

    # orig_sr = sf.info(wav_fn).samplerate
    binaural_audio, orig_sr = librosa.load(wavfile, mono=False, sr=tgt_sr)  # 保持原始采样率

    sr = tgt_sr
    binaural_audio = binaural_audio.T
    clip_audio_samples = (binaural_audio.shape[0] // 400) * 400
    
    if not os.path.exists(os.path.join(output_dir, 'binaural.wav')):
        binaural_audio = binaural_audio[:clip_audio_samples, :]
        sf.write(os.path.join(output_dir, 'binaural.wav'), binaural_audio, sr)
    if not os.path.exists(os.path.join(output_dir, 'mono.wav')):
        mono_audio = np.mean(binaural_audio, axis=1)
        sf.write(os.path.join(output_dir, 'mono.wav'), mono_audio, sr)

    position_length = clip_audio_samples // 400  # 确保为整数

    source_pos = item['pos']['source_pos']
    ear_pos = item['pos']['ear_pos']
    ear_direction = item['pos']['ear_direction']
    assert ear_direction['x']==0 and ear_direction['y']==1 and ear_direction['z']==0, '耳朵朝向不对'
    position = np.array([source_pos['x']-ear_pos['x'], source_pos['y']-ear_pos['y'], source_pos['z']-ear_pos['z'], tw, tx, ty, tz])
    
    txt_output_path = os.path.join(output_dir, 'tx_positions.txt')
    # 写入文件
    with open(txt_output_path, 'w') as f:
        for i in range(position_length):
            line = f"{position[0]:.7f} {position[1]:.7f} {position[2]:.7f} " \
                f"{position[3]:.7f} {position[4]:.7f} {position[5]:.7f} {position[6]:.7f}\n"
            f.write(line)
    
    txt_output_path = os.path.join(output_dir, 'rx_positions.txt')
    # 写入文件
    with open(txt_output_path, 'w') as f:
        for i in range(position_length):
            line = f"{0:.7f} {0:.7f} {0:.7f} " \
                f"{rw:.7f} {rx:.7f} {ry:.7f} {rz:.7f}\n"
            f.write(line)
