import json
import os
import librosa
import soundfile as sf
from multiprocessing import Pool
from tqdm import tqdm
from collections import OrderedDict

json_base_dir = "./3D-Speaker/output"
wav_base_dir = "./Spatial/play"

def merge_segments(segments):
    """Combine consecutive segments of the same speaker (supports any number of consecutive segments)"""
    if not segments:
        return []

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

        j = i + 1
        while j < len(sorted_segments):
            next_seg = sorted_segments[j]
            if (next_seg["speaker"] == current["speaker"] 
                and next_seg["start"] <= current["stop"] + 0.01):
                current["stop"] = max(current["stop"], next_seg["stop"])
                current["original_keys"].append(next_seg["key"])
                j += 1
            else:
                break
        
        merged.append(current)
        i = j
    
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
        base_name = os.path.basename(json_path)
        session_id = base_name.replace("_缩混_cut_new.json", "")
        wav_dir = os.path.join(wav_base_dir, session_id)
        
        original_wav_path = os.path.join(
            wav_base_dir, 
            os.path.basename(wav_dir),
            os.path.basename(wav_dir) + "_缩混_cut.wav" 
        )
        if not os.path.exists(original_wav_path):
            raise FileNotFoundError(f"原始音频不存在: {original_wav_path}")

        with sf.SoundFile(original_wav_path) as f:
            original_subtype = f.subtype
        y, sr = librosa.load(original_wav_path, sr=None, mono=False)
        if y.ndim == 1:
            raise ValueError("单声道音频不符合要求")
        y = y.T  # (samples, channels)

        with open(json_path, 'r') as f:
            raw_data = json.load(f, object_pairs_hook=OrderedDict)
        
        segments = []
        for key, value in raw_data.items():
            segments.append({
                "key": key,
                "start": value["start"],
                "stop": value["stop"],
                "speaker": value["speaker"],
                "text": value["text"]
            })
        
        # combine segments
        merged_segments = merge_segments(segments)
        results['total_clips'] = len(segments)
        results['merged_clips'] = len(merged_segments)

        # generate new config and cut audio
        new_config = OrderedDict()
        output_dir = os.path.join(wav_dir, "split_spker")
        os.makedirs(output_dir, exist_ok=True)

        for seg in merged_segments:
            new_key = f"{session_id}_缩混_cut_{seg['start']:.3f}_{seg['stop']:.3f}_{seg['speaker']}"
            
            output_wav_path = os.path.join(
                output_dir,
                f"{new_key}.wav"
            )
            
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
                # cut audio
                start_sample = int(seg["start"] * sr)
                end_sample = int(seg["stop"] * sr)
                y_cut = y[start_sample:end_sample]
                
                # save audio
                sf.write(output_wav_path, y_cut, sr, subtype=original_subtype)
                new_config[new_key] = {
                    "start": seg['start'],
                    "stop": seg['stop'],
                    "speaker": seg['speaker'],
                    # "text": seg['text'] 
                }
                results['success'] += 1
            except Exception as e:
                error_msg = f"{new_key} (origin keys: {seg['original_keys']}): {str(e)}"
                results['errors'].append(error_msg)

        # save new config
        new_json_path = os.path.join(wav_dir, base_name)
        with open(new_json_path, 'w') as f:
            json.dump(new_config, f, indent=4, ensure_ascii=False)

    except Exception as e:
        results['file_error'] = f"{json_path}: {str(e)}"
    
    return results

def count_total_clips(json_files):
    """count total clips in all JSON files"""
    total = 0
    for path in json_files:
        try:
            with open(path, 'r') as f:
                total += len(json.load(f))
        except:
            continue
    return total

def main():
    # collect all JSON files
    json_files = [
        os.path.join(json_base_dir, f)
        for f in os.listdir(json_base_dir)
        if f.endswith("_缩混_cut_new.json")
    ]
    
    # prepare for processing
    print("正在扫描任务...")
    total_original = count_total_clips(json_files)
    print(f"发现 {len(json_files)} 个文件，共 {total_original} 个原始片段")

    stats = {
        'total_original': total_original,
        'total_merged': 0,
        'success': 0,
        'skipped': 0,
        'errors': [],
        'file_errors': []
    }

    pbar = tqdm(total=len(json_files), desc="🔄 处理文件中")

    # parallel processing
    with Pool(32) as pool:
        for result in pool.imap_unordered(process_single_file, json_files):
            # update stats
            stats['total_merged'] += result['merged_clips']
            stats['success'] += result['success']
            stats['skipped'] += result['skipped']
            stats['errors'].extend(result['errors'])
            if result['file_error']:
                stats['file_errors'].append(result['file_error'])
            
            # update progress bar
            pbar.update(1)
            pbar.set_postfix({
                '合并率': f"{1 - stats['total_merged']/stats['total_original']:.1%}",
                '成功率': f"{stats['success']/stats['total_merged']:.1%}" if stats['total_merged'] else '0%'
            })

    pbar.close()

    # output
    print("\n处理报告:")
    print(f"原始片段总数: {stats['total_original']}")
    print(f"合并后片段数: {stats['total_merged']} (减少 {stats['total_original']-stats['total_merged']})")
    print(f"成功生成: {stats['success']}")
    print(f"跳过文件: {stats['skipped']}")
    print(f"失败片段: {len(stats['errors'])}")
    print(f"失败文件: {len(stats['file_errors'])}")

    # error log
    if stats['errors'] or stats['file_errors']:
        for err in stats['file_errors'][:10]:
            print(f"文件错误: {err}")
        for err in stats['errors'][:10]:
            print(f"片段错误: {err}")
            
if __name__ == "__main__":
    main()