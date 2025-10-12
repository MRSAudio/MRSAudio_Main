import json
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import traceback

# 配置参数
split_json_path = "./audio-visual-seld-dcase2023/_audio_metadata_all_split_with_prompt.json"
csv_base_dir = "./audio-visual-seld-dcase2023/MRSAudio/metadata_dev"
EVENT2CLASS = {
    "toy_car": 1,
    "maracas": 2,
    "handbell": 3,
    "stickbell": 4,
    "gong": 5,
    "Rattle": 6,
    "slitdrum": 7,
    "woodenshaker": 8,
    "clash_cymbals": 9,
    "woodenclapper": 10,
    "chinesecymbals": 11,
    "rainstick": 12,
    "triangle": 13,
    "woodenfish": 14,
    "tambourine": 15,
    "bell": 16,
    "doubletoneblock": 17,
    "WoodenClapper": 18,
    "Claves": 19,
    "whistle": 20,
    "birdwhistle": 21,
    "squeakingrubberchicken": 22,
    "clap": 23,
    "rotatingclapperboard": 24,
    "thundertube": 25,
    "fanwithpaper": 26,
    "hairdryer": 27,
    "Cheerclappers": 28,
    "Tear_off_tape": 29,
    "toy_train": 30,
    "MultiPitchPercussionTube": 31,
}

source_number_index = 1
workers = 32  # 并行进程数

def get_azimuth_elevation_distance(x, y, z):
    """计算方位角、仰角和距离"""
    distance = np.sqrt(x**2 + y**2 + z**2) * 100
    azimuth = np.arctan2(y, x) * 180 / np.pi
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    return int(azimuth), int(elevation), int(distance)

def get_class_index(event):
    """获取事件类别索引"""
    return EVENT2CLASS.get(event, 0)

def npy2csv(npy_path, csv_path, class_index):
    """转换npy文件到csv格式"""
    data = np.load(npy_path)
    csv_data = []
    
    for i in range(0, data.shape[0], 2):
        if i + 1 < data.shape[0]:
            x, y, z, t = (data[i] + data[i+1]) / 2
        else:
            x, y, z, t = data[i]
            
        azimuth, elevation, distance = get_azimuth_elevation_distance(x, y, z)
        csv_data.append([i//2 + 1, class_index, source_number_index, azimuth, elevation, distance])
    
    np.savetxt(csv_path, csv_data, delimiter=",", fmt="%s")

def process_single_item(args):
    """处理单个数据项的worker函数"""
    key, value = args
    try:
        # 解析事件类型
        event = "_".join(value["item_name"].split("_")[1:-3]).translate(str.maketrans('', '', '12345'))
        class_index = get_class_index(event)
        if class_index == 0:
            return key, f"未知事件类型: {event}"
        
        # 准备文件路径
        item_name = value['pos_fn'].split('audio_pos/')[1].replace('.npy', '')
        pos_fn = value['pos_fn'].replace("./Spatial/", "./data/")
        csv_path = os.path.join(csv_base_dir, f"{item_name}.csv")
        
        # 创建目录并转换文件
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        npy2csv(pos_fn, csv_path, class_index)
        
        return key, None
    except Exception as e:
        error_msg = f"{key} 处理失败: {str(e)}\n{traceback.format_exc()}"
        return key, error_msg

def main_parallel():
    # 加载元数据
    with open(split_json_path, "r") as f:
        metadata = json.load(f)
    
    # 准备任务参数
    task_args = [(k, v) for k, v in metadata.items()]
    
    # 创建进程池
    error_log = []
    with Pool(processes=min(workers, len(task_args))) as pool:
        with tqdm(total=len(task_args), desc="转换进度") as pbar:
            # 使用imap_unordered获取更快进度反馈
            results = pool.imap_unordered(process_single_item, task_args)
            
            for key, error in results:
                if error:
                    error_log.append(f"{key}: {error.splitlines()[0]}")
                    pbar.write(f"错误: {error[:100]}...")  # 显示错误摘要
                pbar.update(1)
                pbar.set_postfix({"错误数": len(error_log)})

    # 输出统计信息
    print(f"\n处理完成: 成功 {len(task_args)-len(error_log)}/{len(task_args)}")
    if error_log:
        print(f"前5个错误:")
        for err in error_log[:5]:
            print(f" - {err}")

if __name__ == "__main__":
    main_parallel()