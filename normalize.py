import os
import glob
import soundfile as sf
import pyloudnorm as pyln
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import traceback

def process_single_file(args):
    """处理单个音频文件的worker函数"""
    file_path, target_lufs, overwrite = args
    
    # 构建输出路径
    if overwrite:
        out_path = file_path
    else:
        dir_name = os.path.dirname(file_path)
        new_dir = os.path.join(dir_name, "spatial_normalized")
        os.makedirs(new_dir, exist_ok=True)
        out_path = os.path.join(new_dir, os.path.basename(file_path))
    
    if os.path.exists(out_path):
        return file_path, None
    
    try:
        # 读取音频数据和采样率
        data, rate = sf.read(file_path)
        
        # 创建响度测量器
        meter = pyln.Meter(rate)
        
        # 计算当前响度
        loudness = meter.integrated_loudness(data)
        
        # 处理静音文件
        if loudness == float("-inf"):
            norm_data = data
            print(f"静音文件跳过: {file_path}")
        else:
            # 响度归一化
            norm_data = pyln.normalize.loudness(data, loudness, target_lufs)
            
            # 剪辑处理
            peak = np.max(np.abs(norm_data))
            if peak > 1.0:
                norm_data = norm_data / peak * 0.9999

        # 保存文件
        info = sf.info(file_path)
        sf.write(out_path, norm_data, rate, subtype=info.subtype)
        
        return file_path, None
    
    except Exception as e:
        error_msg = f"{file_path} 处理失败: {str(e)}\n{traceback.format_exc()}"
        return file_path, error_msg

def normalize_to_lufs_parallel(input_dir, target_lufs=-23.0, overwrite=True, workers=32):
    """
    并行响度归一化版本
    """
    # 查找音频文件
    pattern = os.path.join(input_dir, "*", "*_缩混_cut.wav")
    audio_files = sorted(glob.glob(pattern))
    print(f"找到 {len(audio_files)} 个音频文件")

    if not audio_files:
        print("未找到匹配文件")
        return

    # 准备任务参数
    task_args = [(f, target_lufs, overwrite) for f in audio_files]

    # 创建进程池
    error_log = []
    with Pool(processes=min(workers, len(audio_files))) as pool:
        # 使用tqdm显示进度
        with tqdm(total=len(audio_files), desc="音频归一化") as pbar:
            for file_path, error in pool.imap_unordered(process_single_file, task_args):
                if error:
                    error_log.append(error)
                    pbar.write(error)  # 实时显示错误
                pbar.update(1)
                pbar.set_postfix({"错误数": len(error_log)})

    # 输出统计信息
    print(f"\n处理完成: 成功 {len(audio_files)-len(error_log)}/{len(audio_files)}")
    if error_log:
        print(f"错误文件 ({len(error_log)} 个):")
        for err in error_log[:5]:  # 显示前5个错误
            print("-"*50)
            print(err)

if __name__ == "__main__":
    input_dir = "./Spatial/MRSDrama2"
    normalize_to_lufs_parallel(
        input_dir=input_dir,
        target_lufs=-23.0,
        overwrite=False,
        workers=32
    )