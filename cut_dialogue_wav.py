import json
import os
import librosa
import soundfile as sf
from multiprocessing import Pool
from tqdm import tqdm
from collections import OrderedDict

json_base_dir = "./3D-Speaker/output"
# json_base_dir = "/home2/zhangyu/gwx/asr_fix"
wav_base_dir = "./Spatial/play"

def merge_segments(segments):
    """合并连续的同一说话人片段（支持任意数量连续片段）"""
    if not segments:
        return []

    # 按开始时间排序
    sorted_segments = sorted(segments, key=lambda x: x["start"])
    merged = []
    
    i = 0
    while i < len(sorted_segments):
        current = {
            "start": sorted_segments[i]["start"],
            "stop": sorted_segments[i]["stop"],
            "speaker": sorted_segments[i]["speaker"],
            "original_keys": [sorted_segments[i]["key"]]
        }
        
        # 合并后续可合并的片段
        j = i + 1
        while j < len(sorted_segments):
            next_seg = sorted_segments[j]
            # 合并条件：同一说话人且时间连续/重叠
            if (next_seg["speaker"] == current["speaker"] 
                and next_seg["start"] <= current["stop"] + 0.01):
                current["stop"] = max(current["stop"], next_seg["stop"])
                current["original_keys"].append(next_seg["key"])
                j += 1
            else:
                break
        
        merged.append(current)
        i = j  # 直接跳到未处理的位置
    
    return merged

def process_single_file(json_path):
    """处理单个JSON文件"""
    results = {
        'total_clips': 0,
        'merged_clips': 0,
        'success': 0,
        'skipped': 0,
        'errors': [],
        'file_error': None
    }
    
    try:
        # 基础信息解析
        base_name = os.path.basename(json_path)
        session_id = base_name.replace("_缩混_cut_new.json", "")
        wav_dir = os.path.join(wav_base_dir, session_id)
        
        # 原始音频路径
        original_wav_path = os.path.join(
            wav_base_dir, 
            os.path.basename(wav_dir),
            os.path.basename(wav_dir) + "_缩混_cut.wav" 
        )
        if not os.path.exists(original_wav_path):
            raise FileNotFoundError(f"原始音频不存在: {original_wav_path}")

        # 加载音频
        with sf.SoundFile(original_wav_path) as f:
            original_subtype = f.subtype
        y, sr = librosa.load(original_wav_path, sr=None, mono=False)
        if y.ndim == 1:
            raise ValueError("单声道音频不符合要求")
        y = y.T  # (samples, channels)

        # 加载并处理切割配置
        with open(json_path, 'r') as f:
            raw_data = json.load(f, object_pairs_hook=OrderedDict)
        
        # 准备合并数据（保留原始key）
        segments = []
        for key, value in raw_data.items():
            segments.append({
                "key": key,
                "start": value["start"],
                "stop": value["stop"],
                "speaker": value["speaker"],
                "text": value["text"]
            })
        
        # 合并片段
        merged_segments = merge_segments(segments)
        results['total_clips'] = len(segments)
        results['merged_clips'] = len(merged_segments)

        # 生成新配置
        new_config = OrderedDict()
        output_dir = os.path.join(wav_dir, "split_spker")
        os.makedirs(output_dir, exist_ok=True)

        for seg in merged_segments:
            # 生成新键名（保留原始时间精度）
            new_key = f"{session_id}_缩混_cut_{seg['start']:.3f}_{seg['stop']:.3f}_{seg['speaker']}"
            
            # 构建输出路径
            output_wav_path = os.path.join(
                output_dir,
                f"{new_key}.wav"
            )
            
            # 跳过已存在文件
            if os.path.exists(output_wav_path):
                results['skipped'] += 1
                new_config[new_key] = {
                    "start": seg['start'],
                    "stop": seg['stop'],
                    "speaker": seg['speaker'],
                    # "text": seg['text'] 
                }
                continue

            try:
                # 切割音频
                start_sample = int(seg["start"] * sr)
                end_sample = int(seg["stop"] * sr)
                y_cut = y[start_sample:end_sample]
                
                # 保存文件
                sf.write(output_wav_path, y_cut, sr, subtype=original_subtype)
                new_config[new_key] = {
                    "start": seg['start'],
                    "stop": seg['stop'],
                    "speaker": seg['speaker'],
                    # "text": seg['text'] 
                }
                results['success'] += 1
            except Exception as e:
                error_msg = f"{new_key} (原始keys: {seg['original_keys']}): {str(e)}"
                results['errors'].append(error_msg)

        # 保存新配置文件（覆盖原文件）
        new_json_path = os.path.join(wav_dir, base_name)
        # print(f"保存新配置到: {new_json_path}")
        # print(f"新配置: {new_config}")
        with open(new_json_path, 'w') as f:
            json.dump(new_config, f, indent=4, ensure_ascii=False)

    except Exception as e:
        results['file_error'] = f"{json_path}: {str(e)}"
    
    return results

def count_total_clips(json_files):
    """统计原始片段总数"""
    total = 0
    for path in json_files:
        try:
            with open(path, 'r') as f:
                total += len(json.load(f))
        except:
            continue
    return total

def main():
    # 收集JSON文件
    json_files = [
        os.path.join(json_base_dir, f)
        for f in os.listdir(json_base_dir)
        if f.endswith("_缩混_cut_new.json")
    ]
    
    # 预统计
    print("正在扫描任务...")
    total_original = count_total_clips(json_files)
    print(f"发现 {len(json_files)} 个文件，共 {total_original} 个原始片段")

    # 初始化统计
    stats = {
        'total_original': total_original,
        'total_merged': 0,
        'success': 0,
        'skipped': 0,
        'errors': [],
        'file_errors': []
    }

    # 进度条
    pbar = tqdm(total=len(json_files), desc="🔄 处理文件中")

    # 并行处理
    with Pool(32) as pool:
        for result in pool.imap_unordered(process_single_file, json_files):
            # 更新统计
            stats['total_merged'] += result['merged_clips']
            stats['success'] += result['success']
            stats['skipped'] += result['skipped']
            stats['errors'].extend(result['errors'])
            if result['file_error']:
                stats['file_errors'].append(result['file_error'])
            
            # 更新进度
            pbar.update(1)
            pbar.set_postfix({
                '合并率': f"{1 - stats['total_merged']/stats['total_original']:.1%}",
                '成功率': f"{stats['success']/stats['total_merged']:.1%}" if stats['total_merged'] else '0%'
            })

    pbar.close()

    # 输出报告
    print("\n处理报告:")
    print(f"原始片段总数: {stats['total_original']}")
    print(f"合并后片段数: {stats['total_merged']} (减少 {stats['total_original']-stats['total_merged']})")
    print(f"成功生成: {stats['success']}")
    print(f"跳过文件: {stats['skipped']}")
    print(f"失败片段: {len(stats['errors'])}")
    print(f"失败文件: {len(stats['file_errors'])}")

    # 错误日志
    if stats['errors'] or stats['file_errors']:
        for err in stats['file_errors'][:10]:
            print(f"文件错误: {err}")
        for err in stats['errors'][:10]:
            print(f"片段错误: {err}")
            
if __name__ == "__main__":
    main()