import os
import re
import sys
import subprocess

def is_bottom_folder(path):
    return not any(os.path.isdir(os.path.join(path, d)) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))

def process_metadata(root):
    metadata_path = [f for f in os.listdir(root) if f.lower().endswith('metadata.json')][0]
    if os.path.exists(metadata_path):
        parent_dir = os.path.dirname(root)
        date_part = os.path.basename(parent_dir)
        folder_name = os.path.basename(root)
        new_name = f"{date_part}_{folder_name}_metadata.json"
        new_path = os.path.join(root, new_name)
        os.rename(metadata_path, new_path)
        print(f"Renamed metadata: {new_path}")

def process_videos(root, date_part, folder_name):
    mp4_files = []
    pattern = re.compile(r'GX(\d{2})\d+\.MP4', re.IGNORECASE)
    
    for f in os.listdir(root):
        if f.upper().startswith('GX') and f.upper().endswith('.MP4'):
            match = pattern.match(f.upper())
            if match:
                num = int(match.group(1))
                mp4_files.append((num, f))
    
    if not mp4_files:
        return
    
    mp4_files.sort()
    file_list = os.path.join(root, "concat_list.txt")
    with open(file_list, "w") as f:
        for _, filename in mp4_files:
            f.write(f"file '{os.path.join(root, filename)}'\n")
    
    output_file = os.path.join(root, f"{date_part}_{folder_name}_FOV_merged.MP4")
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists, skipping merge.")
        return
    
    try:
        subprocess.run([
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", file_list,
            "-c", "copy",
            output_file
        ], check=True)
        print(f"Merged video: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error merging videos in {root}: {e}")
    # finally:
    #     if os.path.exists(file_list):
    #         os.remove(file_list)

def process_audio(root, date_part, folder_name):
    wav_files = []
    pattern = re.compile(r'.*_(\d+)\.WAV', re.IGNORECASE)
    
    for f in os.listdir(root):
        if f.upper().endswith('.WAV') and '_' in f:
            match = pattern.match(f)
            if match:
                num = int(match.group(1))
                wav_files.append((num, f))
    
    if not wav_files:
        return
    
    wav_files.sort()
    file_list = os.path.join(root, "concat_list.txt")
    with open(file_list, "w") as f:
        for _, filename in wav_files:
            f.write(f"file '{os.path.join(root, filename)}'\n")
    
    output_file = os.path.join(root, f"{date_part}_{folder_name}_FoA_merged.WAV")
    
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists, skipping merge.")
        return
    
    try:
        subprocess.run([
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", file_list,
            "-c", "copy",
            output_file
        ], check=True)
        print(f"Merged audio: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error merging audio in {root}: {e}")

def process_folder(root):
    if not is_bottom_folder(root):
        return
    
    parent_dir = os.path.dirname(root)
    date_part = os.path.basename(parent_dir)
    folder_name = os.path.basename(root)
    print(f"Processing folder: {root}")
    
    # 处理metadata
    process_metadata(root)
    
    # 处理视频文件
    process_videos(root, date_part, folder_name)
    
    # 处理音频文件
    process_audio(root, date_part, folder_name)

def main(root_dir):
    for root, dirs, files in os.walk(root_dir):
        process_folder(root)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_files.py <root_directory>")
        sys.exit(1)
    
    main(sys.argv[1])