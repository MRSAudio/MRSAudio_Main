import json
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import subprocess
import shutil

audio_json = "./Spatial/_audio_metadata_all.json"
audio_pos_dir = "./Spatial/audio_pos"
split_json = "./Spatial/_audio_metadata_all_split.json"

if __name__ == "__main__":
    all_data = defaultdict(list)
    with open(audio_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for key, value in data.items():
        
        pos_fn = value['pos_fn']
        wav_fn = value['vid_fn']
        position = np.load(pos_fn, allow_pickle=True)
        date = pos_fn.split('audio/')[1].split('/')[0]
        
        wav_duration = float(subprocess.run(["ffprobe", "-i", wav_fn, "-show_entries", "format=duration", "-v", "quiet"], capture_output=True).stdout.decode("utf-8").replace("duration=", "").replace("\n", "").replace("[/FORMAT]", "").replace("[FORMAT]", "").strip())
    
        length = min(position.shape[0], int(wav_duration * 20))
        if position.shape[0] > wav_duration * 20:
            print(f"文件长度: {length}, wav_duration: {wav_duration*20}, position.shape[0]: {position.shape[0]}")
        
        for i in range(0, length, 200):
            # 确保 item_name 是独立生成的，不会累积之前的值
            item_name = f"{date}_{value['item_name']}_{i//200}"
            print(item_name)
            
            new_position = position[i:i+200]
            if new_position.shape[0] < 200:  # 如果不足 200 行，处理剩余片段
                print(f"最后片段不足 200 行: {new_position.shape[0]} 行")
            
            new_pos_fn = os.path.join(audio_pos_dir, f"{key}_{i//200}.npy")
            np.save(new_pos_fn, new_position)
            # print(f"保存位置数据: {new_pos_fn}")
            
            # 确保 all_data[item_name] 是独立的，不会影响原始数据
            all_data[item_name] = value.copy()
            all_data[item_name]['pos_fn'] = new_pos_fn
            all_data[item_name]['start'] = i / 20
            all_data[item_name]['stop'] = (i + new_position.shape[0]) / 20
            all_data[item_name]['item_name'] = item_name
            
    with open(split_json, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
        print(f"保存切割后的数据: {split_json}")