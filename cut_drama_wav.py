import json
import os
import librosa
import soundfile as sf
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from resample import resample
from collections import OrderedDict

json_base_dir = "./Spatial/drama_json"

def process_single_file(json_path):
    """处理单个JSON文件（并行安全版本）"""
    print(f"Processing {json_path}")
    try:
        with ope"n(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        origin_wav_name = os.path.basename(json_path.replace(".json", ".wav"))
        origin_wav_name = os.path.join(
            "./Spatial/MRSDrama2",
            origin_wav_name.replace("_缩混_cut.wav", ""),
            "spatial_normalized",
            origin_wav_name
        )
        
        if not os.path.exists(origin_wav_name):
            print(f"File not found: {origin_wav_name}")
            return {"status": "error", "file": json_path, "reason": "原始音频文件不存在"}
        
        # 加载音频文件（仅执行一次）
        with sf.SoundFile(origin_wav_name) as f:
            originalsubtype = f.subtype
            sr = f.samplerate
            
        y, sr = librosa.load(origin_wav_name, sr=sr, mono=False)
        if y.ndim == 1:
            raise ValueError("单声道音频不符合要求")
        y = y.T  # 转换为 (samples, channels)

        processed_items = 0
        error_items = 0
        
        # 处理所有音频片段
        # for item_key in data['items']:
        #     item = data['items'][item_key]
        for key, value in data.items():
            item = value
            try:
                # start = int(item['start_time'] * sr)
                # end = int(item['end_time'] * sr)
                start = int(item['start'] * sr)
                end = int(item['stop'] * sr)
                
                # 执行音频切割
                y_cut = y[start:end]
                
                # 确保输出目录存在
                os.makedirs(os.path.dirname(item['wav_fn']), exist_ok=True)
                
                # 保存音频片段
                sf.write(
                    item['wav_fn'],
                    y_cut,
                    sr,
                    subtype=originalsubtype
                )
                
                if sr != 48000:
                    resample(item['wav_fn'], target_sr=48000)
                    
                processed_items += 1
                # print(f"Processed {item['wav_fn']}")
                
            except Exception as e:
                error_items += 1
                print(f"Error processing {item_key}: {str(e)}")

        return {
            "status": "success",
            "file": json_path,
            "processed": processed_items,
            "errors": error_items
        }
        
    except Exception as e:
        return {
            "status": "error",
            "file": json_path,
            "reason": str(e)
        }

def process_wrapper(args):
    """包装函数用于异常捕获"""
    try:
        return process_single_file(args)
    except Exception as e:
        return {
            "status": "error",
            "file": args,
            "reason": str(e)
        }

include_list = [
    # '0331',
    # "0401",
    # "0402",
    # "0330",
    # "0322",
    # "0420"
    "0418"
]

def main():
    # 收集所有JSON文件
    json_files = []
    for root, _, files in os.walk(json_base_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    # json_files = json_files[:5]
    print(f"发现 {len(json_files)} 个待处理JSON文件")

    # 创建进程池（使用3/4 CPU核心）
    workers = max(1, int(cpu_count() * 0.75))
    total_processed = 0
    total_errors = 0
    file_errors = 0

    with Pool(processes=workers) as pool:
        # 使用tqdm显示进度
        with tqdm(total=len(json_files), desc="处理进度") as pbar:
            # 使用imap保持处理顺序（可选）
            results = []
            for result in pool.imap(process_wrapper, json_files):
                # 更新统计信息
                if result["status"] == "success":
                    total_processed += result["processed"]
                    total_errors += result["errors"]
                else:
                    file_errors += 1
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    "已处理片段": total_processed,
                    "文件错误": file_errors,
                    "片段错误": total_errors
                })
                
                results.append(result)

    # 输出最终统计
    print("\n处理完成:")
    print(f"成功处理文件: {len(json_files) - file_errors}/{len(json_files)}")
    print(f"成功处理片段: {total_processed}")
    print(f"文件级错误: {file_errors}")
    print(f"片段级错误: {total_errors}")

    # 输出错误日志（前10个）
    if any(r["status"] == "error" for r in results):
        print("\n错误详情（前10个）:")
        error_count = 0
        for result in results:
            if result["status"] == "error" and error_count < 10:
                print(f"文件: {result['file']}")
                print(f"原因: {result['reason']}")
                print("-" * 50)
                error_count += 1

if __name__ == "__main__":
    main()