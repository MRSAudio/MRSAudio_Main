# read start time from json file 
# read duration from getting duration of wav and vid
# according to the latest start time and earliest end time
# cut video, wav with start time and duration by ffmpeg, cut npy with numpy
import os
import subprocess
import sys
import time
from tqdm import tqdm
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
import glob
from process_log import read_txt_data

def get_start_time_from_json(json_path):
    # "start_time": "20250319-194832",
    # "vid_time": "20250319-194816",
    # "log_time": "20250319-194828",
    with open(json_path, "r") as f:
        data = json.load(f)
    start_time = data["start_time"] # wav start time
    vid_time = data["vid_time"]     # vid start time
    log_time = data["log_time"]     # log start time
    foa_time = data["foa_time"]     # foa start time
    fov_time = data["fov_time"]     # fov start time
    return start_time, vid_time, log_time, foa_time, fov_time

def get_start_time_from_json_no_log(json_path):
    # "start_time": "20250319-194832",
    # "vid_time": "20250319-194816",
    # "log_time": "20250319-194828",
    with open(json_path, "r") as f:
        data = json.load(f)
    start_time = data["start_time"] # wav start time
    vid_time = data["vid_time"]     # vid start time
    foa_time = data["foa_time"]     # foa start time
    fov_time = data["fov_time"]     # fov start time
    return start_time, vid_time, foa_time, fov_time

def get_start_time_no_log(wav_time, vid_time, foa_time, fov_time):
    # get the latest start time
    start_time = max(wav_time, vid_time, foa_time, fov_time)
    return start_time

def get_duration_no_log(wav_path, vid_path, foa_path, fov_path, wav_cut, vid_cut, foa_cut, fov_cut):
    vid_duration = float(subprocess.run(["ffprobe", "-i", vid_path, "-show_entries", "format=duration", "-v", "quiet"], capture_output=True).stdout.decode("utf-8").replace("duration=", "").replace("\n", "").replace("[/FORMAT]", "").replace("[FORMAT]", "").strip()) - vid_cut.total_seconds()
    wav_duration = float(subprocess.run(["ffprobe", "-i", wav_path, "-show_entries", "format=duration", "-v", "quiet"], capture_output=True).stdout.decode("utf-8").replace("duration=", "").replace("\n", "").replace("[/FORMAT]", "").replace("[FORMAT]", "").strip()) - wav_cut.total_seconds()
    foa_duration = float(subprocess.run(["ffprobe", "-i", foa_path, "-show_entries", "format=duration", "-v", "quiet"], capture_output=True).stdout.decode("utf-8").replace("duration=", "").replace("\n", "").replace("[/FORMAT]", "").replace("[FORMAT]", "").strip()) - foa_cut.total_seconds()
    fov_duration = float(subprocess.run(["ffprobe", "-i", fov_path, "-show_entries", "format=duration", "-v", "quiet"], capture_output=True).stdout.decode("utf-8").replace("duration=", "").replace("\n", "").replace("[/FORMAT]", "").replace("[FORMAT]", "").strip()) - fov_cut.total_seconds()
    
    print(f"vid_duration: {vid_duration}, wav_duration: {wav_duration}, foa_duration: {foa_duration}, fov_duration: {fov_duration}")
    return min(wav_duration, vid_duration, foa_duration, fov_duration)

def get_duration(log_path, wav_path, vid_path, foa_path, fov_path, log_cut, wav_cut, vid_cut, foa_cut, fov_cut):
    vid_duration = float(subprocess.run(["ffprobe", "-i", vid_path, "-show_entries", "format=duration", "-v", "quiet"], capture_output=True).stdout.decode("utf-8").replace("duration=", "").replace("\n", "").replace("[/FORMAT]", "").replace("[FORMAT]", "").strip()) - vid_cut.total_seconds()
    wav_duration = float(subprocess.run(["ffprobe", "-i", wav_path, "-show_entries", "format=duration", "-v", "quiet"], capture_output=True).stdout.decode("utf-8").replace("duration=", "").replace("\n", "").replace("[/FORMAT]", "").replace("[FORMAT]", "").strip()) - wav_cut.total_seconds()
    foa_duration = float(subprocess.run(["ffprobe", "-i", foa_path, "-show_entries", "format=duration", "-v", "quiet"], capture_output=True).stdout.decode("utf-8").replace("duration=", "").replace("\n", "").replace("[/FORMAT]", "").replace("[FORMAT]", "").strip()) - foa_cut.total_seconds()
    fov_duration = float(subprocess.run(["ffprobe", "-i", fov_path, "-show_entries", "format=duration", "-v", "quiet"], capture_output=True).stdout.decode("utf-8").replace("duration=", "").replace("\n", "").replace("[/FORMAT]", "").replace("[FORMAT]", "").strip()) - fov_cut.total_seconds()
    
    log_duration = get_duration_from_txt(log_path) - log_cut.total_seconds()
    print(f"vid_duration: {vid_duration}, wav_duration: {wav_duration}, log_duration: {log_duration}, foa_duration: {foa_duration}, fov_duration: {fov_duration}")
    return min(wav_duration, vid_duration, log_duration, foa_duration, fov_duration)

def get_duration_from_txt(log_path):
    """从TXT文件读取原始数据"""
    with open (log_path, 'r') as f:
        log_time = f.readline().split(' ')[0].split('_')[1].split(':')[0:4]
        lines = f.readlines()
    log_time = float(log_time[0])*3600 + float(log_time[1])*60 + float(log_time[2]) + float(log_time[3])/1000
    
    x, y, z, t = [], [], [], []
    for line in lines:
        if f'TAG0' in line and "ANC" not in line and "nan" not in line:
            # print(line)
            try:
                t.append((float(line.split(' ')[0].split(':')[2]) + float(line.split(' ')[0].split(':')[1])*60 + float(line.split(' ')[0].split(':')[0])*3600 + float(line.split(' ')[0].split(':')[3])/1000 - log_time)*1000)
                x.append(float(line.split('[')[1].split(',')[0]))
                y.append(float(line.split(',')[1]))
                z.append(float(line.split(',')[2].split(']')[0]))
            except ValueError:
                print(f"数据格式错误: {line}")
                continue
    # check nan
    if any(np.isnan(i) for i in z):
        print(f"{log_path}数据中包含NaN: {line}")
            
    return round(((t[-1] - t[0])/1000), 3) # 返回最后一个时间戳减去第一个时间戳的差值

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
    
def get_start_time(wav_time, vid_time, log_time, foa_time, fov_time):
    # get the latest start time
    start_time = max(wav_time, vid_time, log_time, foa_time, fov_time)
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
        print(f"音频切割成功：{output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"切割失败：{e.stderr.decode()}")
        return False

def cut_fov(fov_path, start_time, duration, output_path):
    """
    增强版GoPro切割函数（兼容旧版ffmpeg）
    """
    cmd = [
        "ffmpeg",
        "-hwaccel", "cuda",          # NVIDIA显卡加速
        "-hwaccel_output_format", "cuda",
        "-i", fov_path,
        "-ss", str(start_time),            # 注意参数顺序优化
        "-t", str(duration),
        "-map", "0:v:0",              # 精确选择流
        "-map", "0:a:0",
        "-c:v", "h264_nvenc",         # HEVC硬件解码
        "-c:a", "copy",
        "-vsync", "0",                # 禁用帧同步
        "-y",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        print(f"视频切割成功：{output_path}")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode()
        if "Could not find tag for codec" in error_msg:
            # 特殊处理数据流问题
            return handle_special_streams(fov_path, output_path, start_time, duration)
        print(f"切割失败：{error_msg}")
        return False


def handle_special_streams(input_path, output_path, start, duration):
    """处理包含特殊数据流的情况"""
    backup_cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-map", "0",
        "-dn",               # 排除数据流
        "-c", "copy",
        "-f", "matroska",    # 使用更兼容的容器
        "-y",
        output_path.replace('.MP4', '.mkv')
    ]
    
    try:
        subprocess.run(backup_cmd, check=True)
        print(f"使用MKV容器保存：{output_path.replace('.MP4', '.mkv')}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"最终失败：{e.stderr.decode()}")
        return False

def cut_npy(txt_path, start_time, duration, out_path):
    x, y, z, t = read_txt_data(txt_path, "0")
    for i in range(len(t)):
        t[i] -= t[0]
    data = np.array([x, y, z, t]).T
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
    np.save(out_path, data)
    print(f"Cutting npy from {start_index} to {end_index}, corresponding time {data[0, 3]} to {data[-1, 3]}")
    
def check_single(file_list, info):
    if len(file_list) != 1:
        print(f"{info}文件数量不正确: {file_list}")
        return file_list
    return file_list[0]

def mmsssss2float(string):
    # "2:49.245" -> 2*60 + 49.245
    # possible formats: "1:07:26.286" -> 1*3600 + 7*60 + 26.286
    if string.count(":") == 1:
        minutes, seconds = string.split(":")
        seconds = seconds.split(".")
        if len(seconds) == 1:
            return int(minutes) * 60 + float(seconds[0])
        else:
            return int(minutes) * 60 + float(seconds[0]) + float(seconds[1]) / (10 ** len(seconds[1]))
    elif string.count(":") == 2:
        hours, minutes, seconds = string.split(":")
        seconds = seconds.split(".")
        if len(seconds) == 1:
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds[0])
        else:
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds[0]) + float(seconds[1]) / (10 ** len(seconds[1]))
    else:
        raise ValueError(f"Invalid time format: {string}")
    
def hhmmss2float(string):
    # "00:02:49" -> 2*60 + 49
    hours, minutes, seconds = string.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

def timedelta_to_hhmmss(td):
    """Convert a timedelta object to a string in HH:MM:SS format."""
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def cut_label(label_path, start_time, duration, output_path):
    data = pd.read_csv(label_path, encoding="utf-8", delimiter="\t")
    # print(data.columns)
    data["Start"] = data["Start"].apply(lambda x: mmsssss2float(x) - hhmmss2float(timedelta_to_hhmmss(start_time)))
    # save result
    data.to_csv(output_path, index=False, encoding="utf-8")
    
def main():
    sum = 0
    base_dir = "./Spatial/MRSAudio2"
    _dirs = glob.glob(os.path.join(base_dir, "*", "*"))
    _dirs = sorted(_dirs)
    for _dir in tqdm(sorted(_dirs)):
        # if "bell" in _dir or "doubletoneblock" in _dir:
        #     continue
        dir_path = _dir
        if "sports" not in dir_path:
            print(f"Processing directory: {dir_path}")
            json_path = check_single([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".json")], "json")
            wav_path = check_single([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".wav") and "缩混" in f  and "cut" not in f], "wav")
            vid_path = check_single([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".mp4") and "4k" in f and "cut" not in f], "vid")
            foa_path = check_single([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".WAV") and "FoA_merged" in f and "cut" not in f], "foa")
            fov_path = check_single([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".MP4") and "FOV_merged" in f and "cut" not in f], "fov")
            log_path = check_single([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith("_log.txt") and "cut" not in f], "log")
            label_path = check_single([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".csv") and "标记.csv" in f], "label")
            wav_time, vid_time, log_time, foa_time, fov_time = get_start_time_from_json(json_path)
            wav_time, vid_time, log_time, foa_time, fov_time = string2time(wav_time), string2time(vid_time), string2time(log_time), string2time(foa_time), string2time(fov_time)
            
            start_time = get_start_time(wav_time, vid_time, log_time, foa_time, fov_time)
            duration = get_duration(log_path, wav_path, vid_path, foa_path, fov_path, \
                        start_time - log_time, start_time - wav_time, start_time - vid_time, start_time - foa_time, start_time - fov_time, \
                        )
            # sum += duration
            vid_out = vid_path.replace(".mp4", "_cut.mp4")
            wav_out = wav_path.replace(".wav", "_cut.wav")
            log_out = "/".join(log_path.split("/")[:-1]) + "/tag0.npy"
            foa_out = foa_path.replace(".WAV", "_cut.WAV")
            fov_out = fov_path.replace(".MP4", "_cut.MP4")
            label_out = label_path.replace(".csv", "_cut.csv")
            
            if os.path.exists(vid_out) and os.path.exists(wav_out) and os.path.exists(log_out) and os.path.exists(foa_out) and os.path.exists(fov_out):
                print(f"文件已存在: {vid_out}, {wav_out}, {log_out}, {foa_out}, {fov_out}")
                continue
            print(f"开始时间: {start_time}, 持续时间: {duration}")
            if not os.path.exists(vid_out):
                cut_vid(vid_path, start_time - vid_time, duration, vid_path.replace(".mp4", "_cut.mp4"))
                print(f"视频前面截取长度: {start_time - vid_time}, 长度: {duration}")
            if not os.path.exists(wav_out):
                cut_wav(wav_path, start_time - wav_time, duration, wav_path.replace(".wav", "_cut.wav"))
                print(f"音频前面截取长度: {start_time - wav_time}, 长度: {duration}")
            if not os.path.exists(log_out):
                cut_npy(log_path, start_time - log_time, duration, log_out)
                print(f"npy文件{log_out}前面截取长度: {start_time - log_time}, 长度: {duration}")
            if not os.path.exists(foa_out):
                cut_foa(foa_path, start_time - foa_time, duration, foa_path.replace(".WAV", "_cut.WAV"))
                print(f"FoA前面截取长度: {start_time - foa_time}, 长度: {duration}")
            if not os.path.exists(fov_out):
                cut_fov(fov_path, start_time - fov_time, duration, fov_path.replace(".MP4", "_cut.MP4"))
                print(f"FOV前面截取长度: {start_time - fov_time}, 长度: {duration}")
            if not os.path.exists(label_out):
                cut_label(label_path, start_time - wav_time, duration, label_path.replace(".csv", "_cut.csv"))
                print(f"标记前面截取长度: {start_time - wav_time}, 长度: {duration}")
        else: # sports no log
            print(f"Processing directory: {dir_path}")
            json_path = check_single([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".json")], "json")
            wav_path = check_single([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".wav") and "缩混" in f  and "cut" not in f], "wav")
            vid_path = check_single([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".mp4") and "4k" in f and "cut" not in f], "vid")
            foa_path = check_single([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".WAV") and "FoA_merged" in f and "cut" not in f], "foa")
            fov_path = check_single([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".MP4") and "FOV_merged" in f and "cut" not in f], "fov")
            # log_path = check_single([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith("_log.txt") and "cut" not in f], "log")
            label_path = check_single([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".csv") and "标记.csv" in f], "label")
            wav_time, vid_time, foa_time, fov_time = get_start_time_from_json_no_log(json_path)
            wav_time, vid_time, foa_time, fov_time = string2time(wav_time), string2time(vid_time), string2time(foa_time), string2time(fov_time)
            
            start_time = get_start_time_no_log(wav_time, vid_time, foa_time, fov_time)
            duration = get_duration_no_log(wav_path, vid_path, foa_path, fov_path, \
                        start_time - wav_time, start_time - vid_time, start_time - foa_time, start_time - fov_time, \
                        )
            # sum += duration
            vid_out = vid_path.replace(".mp4", "_cut.mp4")
            wav_out = wav_path.replace(".wav", "_cut.wav")
            foa_out = foa_path.replace(".WAV", "_cut.WAV")
            fov_out = fov_path.replace(".MP4", "_cut.MP4")
            label_out = label_path.replace(".csv", "_cut.csv")
            
            if os.path.exists(vid_out) and os.path.exists(wav_out) and os.path.exists(foa_out) and os.path.exists(fov_out):
                print(f"文件已存在: {vid_out}, {wav_out}, {foa_out}, {fov_out}")
                continue
            print(f"开始时间: {start_time}, 持续时间: {duration}")
            if not os.path.exists(vid_out):
                cut_vid(vid_path, start_time - vid_time, duration, vid_path.replace(".mp4", "_cut.mp4"))
                print(f"视频前面截取长度: {start_time - vid_time}, 长度: {duration}")
            if not os.path.exists(wav_out):
                cut_wav(wav_path, start_time - wav_time, duration, wav_path.replace(".wav", "_cut.wav"))
                print(f"音频前面截取长度: {start_time - wav_time}, 长度: {duration}")
            if not os.path.exists(foa_out):
                cut_foa(foa_path, start_time - foa_time, duration, foa_path.replace(".WAV", "_cut.WAV"))
                print(f"FoA前面截取长度: {start_time - foa_time}, 长度: {duration}")
            if not os.path.exists(fov_out):
                cut_fov(fov_path, start_time - fov_time, duration, fov_path.replace(".MP4", "_cut.MP4"))
                print(f"FOV前面截取长度: {start_time - fov_time}, 长度: {duration}")
            if not os.path.exists(label_out):
                cut_label(label_path, start_time - wav_time, duration, label_path.replace(".csv", "_cut.csv"))
                print(f"标记前面截取长度: {start_time - wav_time}, 长度: {duration}")

            
if __name__ == "__main__":
    main()