import os
import subprocess
import sys
import time
from tqdm import tqdm
from datetime import datetime, timedelta
import json
import numpy as np

def get_start_time_from_json(json_path):
    # "start_time": "20250319-194832",
    # "vid_time": "20250319-194816",
    # "log_time": "20250319-194828",
    with open(json_path, "r") as f:
        data = json.load(f)
    start_time = data["start_time"] # wav start time
    vid_time = data["vid_time"]     # vid start time
    log_time = data["log_time"]     # log start time
    return start_time, vid_time, log_time

def get_duration(log_path, wav_path, vid_path, log_cut, wav_cut, vid_cut):
    vid_duration = float(subprocess.run(["ffprobe", "-i", vid_path, "-show_entries", "format=duration", "-v", "quiet"], capture_output=True).stdout.decode("utf-8").replace("duration=", "").replace("\n", "").replace("[/FORMAT]", "").replace("[FORMAT]", "").strip()) - vid_cut.total_seconds()
    wav_duration = float(subprocess.run(["ffprobe", "-i", wav_path, "-show_entries", "format=duration", "-v", "quiet"], capture_output=True).stdout.decode("utf-8").replace("duration=", "").replace("\n", "").replace("[/FORMAT]", "").replace("[FORMAT]", "").strip()) - wav_cut.total_seconds()
    data = np.load(log_path, allow_pickle=True)
    end_time = data[-1, 3]
    start_time = data[0, 3]
    log_duration = (end_time - start_time)/1000 - log_cut.total_seconds()
    print(f"vid_duration: {vid_duration}, wav_duration: {wav_duration}, log_duration: {log_duration}")
    return min(wav_duration, vid_duration, log_duration)

def string2time(string):
    # "20250319-194832" -> HHMMSS
    time = string.split("-")
    year = int(time[0][:4])
    month = int(time[0][4:6])
    day = int(time[0][6:])
    hour = int(time[1][:2])
    minute = int(time[1][2:4])
    second = int(time[1][4:])
    return datetime(year, month, day, hour, minute, second)
    
def get_start_time(wav_time, vid_time, log_time):
    # get the latest start time
    start_time = max(wav_time, vid_time, log_time)
    return start_time

def cut_vid(vid_path, start_time, duration, output_path):
    print(f"Cutting video from {start_time} to {start_time.total_seconds() + duration}")
    return_code = subprocess.run([
        "ffmpeg",
        "-i", vid_path,
        "-ss", str(start_time),
        "-t", str(duration),
        "-vf", "scale=1920:1080",
        "-r", "24",
        "-c:v", "h264_nvenc",
        "-c:a", "copy",
        output_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode
    if return_code != 0:
        print(f"Error cutting video: {vid_path}")
        sys.exit(1)
    print(f"Video cut successfully: {output_path}")
    return return_code

def cut_wav(wav_path, start_time, duration, output_path):
    print(f"Cutting wav from {start_time} to {start_time.total_seconds() + duration}")
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

def cut_npy(npy_paths, start_time, duration):
    # print(start_time, start_time.total_seconds() + duration)
    for npy_path in npy_paths:
        data = np.load(npy_path, allow_pickle=True)
        # get start_index and end_index
        start_index = 0
        for i in range(data.shape[0]):
            if int(data[i, 3]) < start_time.total_seconds() * 1000:
                continue
            start_index = i
            break
        end_index = data.shape[0]-1
        for i in range(data.shape[0]):
            if int(data[i, 3]) < (start_time.total_seconds() * 1000 + duration * 1000):
                continue
            end_index = i
            break
        data = data[start_index:end_index, :]
        # save npy
        np.save(npy_path.replace(".npy", "_cut.npy"), data)
        print(f"Cutting npy from {start_index} to {end_index}, corresponding time {data[0, 3]} to {data[-1, 3]}")
        # print(f"npy cut successfully: {npy_path.replace('.npy', '_cut.npy')}")

def main():
    sum = 0
    base_dir = "./Spatial/MRSDrama3"
    for root, _dirs, files in os.walk(base_dir):
        for _dir in tqdm(sorted(_dirs)):
            if "2025" not in _dir or "20250424-141610-7-cg-lk-lyh-《教父》电影剧本_part1" in _dir:
                continue
            dir_path = os.path.join(root, _dir)
            print(f"Processing directory: {dir_path}")
            json_path = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".json")][0]
            wav_path = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".wav") and "缩混" in f  and "cut" not in f][0]
            vid_path = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".mp4") and "compressed" in f and "cut" not in f][0]
            log_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".npy") and "cut" not in f]
            wav_time, vid_time, log_time = get_start_time_from_json(json_path)
            wav_time, vid_time, log_time = string2time(wav_time), string2time(vid_time), string2time(log_time)
            start_time = get_start_time(wav_time, vid_time, log_time)
            duration = get_duration(log_paths[0], wav_path, vid_path, start_time - log_time, start_time - wav_time, start_time - vid_time)
            sum += duration
            vid_out = vid_path.replace(".mp4", "_cut.mp4")
            wav_out = wav_path.replace(".wav", "_cut.wav")
            log_out = log_paths[0].replace(".npy", "_cut.npy")
            if os.path.exists(vid_out) and os.path.exists(wav_out) and os.path.exists(log_out):
                print(f"文件已存在: {vid_out}, {wav_out}, {log_out}")
                continue
            print(f"开始时间: {start_time}, 持续时间: {duration}")
            if not os.path.exists(vid_out):
                cut_vid(vid_path, start_time - vid_time, duration, vid_path.replace(".mp4", "_cut.mp4"))
                print(f"视频前面截取长度: {start_time - vid_time}, 长度: {duration}")
            if not os.path.exists(wav_out):
                cut_wav(wav_path, start_time - wav_time, duration, wav_path.replace(".wav", "_cut.wav"))
                print(f"音频前面截取长度: {start_time - wav_time}, 长度: {duration}")
            if not os.path.exists(log_out):
                cut_npy(log_paths, start_time - log_time, duration)
                print(f"npy前面截取长度: {start_time - log_time}, 长度: {duration}")
            # break
    print(f"总时长: {sum}s")
    # translate second into HH:MM:SS
    hours = sum // 3600
    minutes = (sum % 3600) // 60
    seconds = sum % 60
    print(f"总时长: {int(hours)}小时{int(minutes)}分钟{int(seconds)}秒")
            
            
if __name__ == "__main__":
    main()