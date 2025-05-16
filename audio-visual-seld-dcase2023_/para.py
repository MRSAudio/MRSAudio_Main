import json
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import subprocess
import librosa
import soundfile as sf
import sys
from multiprocessing import Pool, cpu_count
import traceback

# 配置路径
split_json = "./audio-visual-seld-dcase2023/_audio_metadata_all_split_with_prompt.json"
target_dir = "./audio-visual-seld-dcase2023/MRSAudio"
wav_target_dir = "./data/audio_bin"

def cut_foa(foa_path, start_time, duration, output_path):
    """
    精确切割Ambisonic音频（保持24bit/48kHz/4通道特性）
    
    参数：
    input_path: 输入WAV文件路径
    output_path: 输出WAV路径
    start_time: 切割开始时间
    duration: 切割时长
    """
    # 精确到采样点的切割参数
    cmd = [
        "ffmpeg",
        "-i", foa_path,
        "-ss", str(start_time),
        "-t", str(duration),
        "-c:a", "copy",  # 直接复制音频流
        "-map_metadata", "0",
        "-fflags", "+genpts",
        "-ac", "4",      # 强制保持4通道
        "-ar", "48000",  # 保持采样率
        "-sample_fmt", "s32",  # 保持24bit包装格式
        "-y",
        output_path
    ]
    
    cmd += [
        "-af", "apad=whole_dur={duration}".format(duration=duration),  # 精确填充
        "-fflags", "+igndts"  # 忽略解码时间戳
    ]
    
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        # print(f"音频切割成功：{output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"切割失败：{e.stderr.decode()}")
        return False

def cut_wav(wav_path, start_time, duration, output_path):
    print(f"Cutting wav from {start_time} to {start_time + duration}")
    return_code = subprocess.run([
        "ffmpeg",
        "-i", wav_path,          # 输入音频文件
        "-ss", str(start_time),  # 开始时间
        "-t", str(duration),     # 持续时间
        "-acodec", "copy",       # 保持原始音频编码
        output_path              # 输出文件路径
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode

    if return_code != 0:
        print(f"Error cutting audio: {wav_path}")
        sys.exit(1)
    print(f"Audio cut successfully: {output_path}")
    return return_code
   
def cut_vid(vid_path, start_time, duration, output_path):
    print(f"Cutting video from {start_time} to {start_time + duration}")
    return_code = subprocess.run([
        "ffmpeg",
        "-i", vid_path,
        "-ss", str(start_time),
        "-t", str(duration),
        "-vf", "scale=1920:1080",
        "-r", "29.97",
        "-c:v", "h264_nvenc",
        "-c:a", "copy",
        output_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode
    if return_code != 0:
        print(f"Error cutting video: {output_path}")
        return_code = subprocess.run([
            "ffmpeg",
            "-i", vid_path,
            "-ss", str(start_time),
            "-t", str(duration),
            "-r", "29.97",
            "-c:v", "libx264",
            "-c:a", "copy",
            output_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode
        if return_code != 0:
            print(f"Error cutting video with libx264: {output_path}")
            sys.exit(1)
    print(f"Video cut successfully: {output_path}")
    return return_code

def resample(file, target_sr=24000):
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
        
def downsample_foa_audio(audio_path, original_sr=48000, target_sr=24000):
    """
    将四声道的FOA音频从原始采样率降采样到目标采样率。

    参数：
        audio_array (np.ndarray): 输入的音频数组，形状需为 (n_samples, 4)
        original_sr (int): 原始采样率，默认为48000 Hz
        target_sr (int): 目标采样率，默认为24000 Hz

    返回：
        np.ndarray: 降采样后的音频数组，形状为 (new_n_samples, 4)
    """
    # 检查输入合法性
    audio_array, sr = sf.read(audio_path, dtype='float32')
    if original_sr <= target_sr:
        raise ValueError("目标采样率必须小于原始采样率")
    if audio_array.ndim != 2 or audio_array.shape[1] != 4:
        raise ValueError("输入音频应为四声道，形状需为 (n_samples, 4)")
    
    # 对每个声道进行重采样
    downsampled_channels = []
    for channel in range(4):
        channel_data = audio_array[:, channel]
        resampled = librosa.resample(
            channel_data,
            orig_sr=original_sr,
            target_sr=target_sr,
            # res_type="soxr_vhq"  # 高质量抗混叠滤波
        )
        downsampled_channels.append(resampled)
    
    # 合并声道并转置为 (n_samples, 4)
    downsampled_audio = np.vstack(downsampled_channels).T
    return downsampled_audio

def process_single_item(args):
    """并行处理单个音频视频项"""
    key, value = args
    try:
        date = value['wav_fn'].split('audio/')[1].split('/')[0]
        item_name = value['pos_fn'].split('audio_pos/')[1].replace('.npy', '')
        start_time = value['start']
        end_time = value['stop']
        duration = end_time - start_time

        # 处理FOA音频
        foa_path = value['foa_fn'].replace("./Spatial/MRSAudio/", "./data/")
        foa_dir = os.path.join(target_dir, "foa_dev")
        os.makedirs(foa_dir, exist_ok=True)
        foa_output_path = os.path.join(foa_dir, f"{item_name}.WAV")
        
        if not os.path.exists(foa_output_path):
            # 切割音频
            cut_foa(foa_path, start_time, duration, foa_output_path)
            # 降采样
            downsampled_audio = downsample_foa_audio(foa_output_path)
            sf.write(foa_output_path, downsampled_audio, 24000)

        # 处理视频
        vid_path = value['vid_fn'].replace("./Spatial/MRSAudio/", "./data/")
        vid_dir = os.path.join(target_dir, "video_dev")
        os.makedirs(vid_dir, exist_ok=True)
        vid_output_path = os.path.join(vid_dir, f"{item_name}.mp4")
        
        if not os.path.exists(vid_output_path):
            cut_vid(vid_path, start_time, duration, vid_output_path)
            
        # 处理音频
        wav_path = value['wav_fn'].replace("./Spatial/MRSAudio/", "./data/")
        wav_dir = os.path.join(wav_target_dir, "audio_dev")
        os.makedirs(wav_dir, exist_ok=True)
        wav_output_path = os.path.join(wav_dir, f"{item_name}.wav")
        
        if not os.path.exists(wav_output_path):
            cut_wav(wav_path, start_time, duration, wav_output_path)
            # 重采样
            resample(wav_output_path, target_sr=24000)

        return key, None
    except Exception as e:
        error_msg = f"{key} 处理失败: {str(e)}\n{traceback.format_exc()}"
        return key, error_msg

def main_parallel():
    # 加载数据
    with open(split_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 准备任务参数
    task_args = [(k, v) for k, v in data.items()]
    
    # 创建进程池
    workers = 24
    error_log = []
    
    with Pool(processes=workers) as pool:
        # 使用双进度条：总体进度和当前任务
        with tqdm(total=len(task_args), desc="总进度") as pbar:
            # 使用imap_unordered获取更快进度反馈
            results = pool.imap_unordered(process_single_item, task_args)
            
            for key, error in results:
                if error:
                    error_log.append(error)
                    pbar.write(f"错误: {error[:100]}...")  # 显示错误摘要
                pbar.update(1)
                pbar.set_postfix({"错误数": len(error_log)})

    # 输出统计信息
    print(f"\n处理完成: 成功 {len(task_args)-len(error_log)}/{len(task_args)}")
    if error_log:
        print(f"前5个错误详情:")
        for err in error_log[:5]:
            print("-"*50)
            print(err)

# 保持原有工具函数不变 (cut_foa, cut_vid, downsample_foa_audio等)

if __name__ == "__main__":
    main_parallel()