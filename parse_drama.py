import os
import re
import json
from difflib import SequenceMatcher
from tqdm import tqdm
import numpy as np
import pandas as pd


def parse_textgrid(textgrid_path):
    intervals = []
    with open(textgrid_path, 'r', encoding='utf-8') as f:
        content = f.read()
        intervals_section = re.search(r'intervals: size = \d+((?:.|\n)*?)(?=\n\s*item|\Z)', content)
        if not intervals_section:
            raise ValueError("Invalid TextGrid format: intervals section not found")
        intervals_section = intervals_section.group(1)
        interval_pattern = re.compile(
            r'intervals \[\d+\]:\s+xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"', 
            re.DOTALL
        )
        matches = interval_pattern.findall(intervals_section)
        for match in matches:
            xmin = float(match[0])
            xmax = float(match[1])
            text = match[2].strip().replace('"', '')
            intervals.append({'xmin': xmin, 'xmax': xmax, 'text': text})
    return intervals

def parse_txt(txt_path):
    items = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            speaker_part, _, dialogue = line.partition(':')
            speaker = speaker_part.strip()
            if len(speaker) != 1 or not speaker.isupper():
                continue
            raw_txt = dialogue.strip()
            items.append({'speaker': speaker, 'raw_txt': raw_txt})
    return items

def find_best_match(intervals, target_chars, start_pos, max_scan=200):
    # 预处理：获取纯汉字序列
    target_clean = re.findall(r'[\u4e00-\u9fff]', target_chars)
    interval_texts = [t['text'] for t in intervals]
    
    # 动态规划矩阵初始化
    dp = [[0]*(len(interval_texts)+1) for _ in range(len(target_clean)+1)]
    path = [[None]*(len(interval_texts)+1) for _ in range(len(target_clean)+1)]
    
    # 填充动态规划矩阵
    max_score = 0
    best_end = 0
    for i in range(1, len(target_clean)+1):
        for j in range(start_pos, min(len(interval_texts), start_pos + max_scan)):
            if target_clean[i-1] == interval_texts[j]:
                dp[i][j+1] = dp[i-1][j] + 1
                path[i][j+1] = (i-1, j)
            else:
                # 考虑三种情况：跳过目标字符、跳过间隔字符、替换
                skip_target = dp[i-1][j+1]
                skip_interval = dp[i][j]
                replace = dp[i-1][j] - 0.5
                dp[i][j+1] = max(skip_target, skip_interval, replace)
                # 记录路径
                if dp[i][j+1] == skip_target:
                    path[i][j+1] = (i-1, j+1)
                elif dp[i][j+1] == skip_interval:
                    path[i][j+1] = (i, j)
                else:
                    path[i][j+1] = (i-1, j)
            
            if dp[i][j+1] > max_score:
                max_score = dp[i][j+1]
                best_end = j+1
    
    # 回溯最佳路径
    if max_score < len(target_clean)*0.5:  # 匹配阈值
        return None
    
    # 找到匹配结束位置
    match_length = 0
    i, j = len(target_clean), best_end
    while i > 0 and j > start_pos:
        prev_i, prev_j = path[i][j]
        if prev_i < i:  # 匹配或替换
            match_length += 1
        i, j = prev_i, prev_j
    
    start_idx = j
    end_idx = best_end - 1
    return (start_idx, end_idx)

def generate_json(textgrid_path, txt_path):
    intervals = parse_textgrid(textgrid_path)
    items = parse_txt(txt_path)
    
    current_pos = 0
    matched_items = []
    errors = []
    
    for idx, item in enumerate(items):
        target_chars = item['raw_txt']
        match_result = find_best_match(intervals, target_chars, current_pos)
        
        if not match_result:
            errors.append(f"匹配失败：第{idx+1}句 -> {target_chars}")
            continue
            
        start, end = match_result
        matched_intervals = intervals[start:end+1]
        
        if not matched_intervals:  # Check if matched_intervals is empty
            errors.append(f"匹配失败：第{idx+1}句 -> {target_chars} (无匹配间隔)")
            continue

        current_pos = end + 1

        # 生成对齐后的文本
        aligned_text = []
        target_ptr = 0
        target_clean = re.findall(r'[\u4e00-\u9fff]', target_chars)
        
        for interval in matched_intervals:
            text = interval['text']
            if text == target_clean[target_ptr] if target_ptr < len(target_clean) else None:
                aligned_text.append(text)
                target_ptr += 1
            else:
                aligned_text.append(text)
        
        # 处理剩余未匹配的目标字符
        if target_ptr < len(target_clean):
            errors.append(f"部分匹配：第{idx+1}句 -> 原句：{target_chars} | 匹配到：{''.join(aligned_text)}")
        
        # 构建结果项
        words = []
        word_dur = []
        for interval in matched_intervals:
            words.append('<SP>' if not interval['text'] else interval['text'])
            word_dur.append(interval['xmax'] - interval['xmin'])
        
        matched_items.append({
            'speaker': item['speaker'],
            'raw_txt': target_chars,
            'textgrid_txt': ''.join(aligned_text),
            'words': words,
            'word_dur': word_dur,
            'start': matched_intervals[0]['xmin'],
            'end': matched_intervals[-1]['xmax']
        })
    
    # 时间间隔调整（同前）
    for i in range(len(matched_items)-1):
        current = matched_items[i]
        next_item = matched_items[i+1]
        gap = next_item['start'] - current['end']
        if gap > 0:
            half_gap = gap / 2
            current['end'] += half_gap
            next_item['start'] -= half_gap
        if next_item['start'] < current['end']:
            next_item['start'] = current['end']
    
    # 构建最终输出
    output = {'items': {}, 'errors': errors}
    base_name = os.path.splitext(os.path.basename(textgrid_path))[0]
    
    for item in matched_items:
        start = item['start']
        end = item['end']
        speaker = item['speaker']
        item_name = f"{base_name}_{start:.3f}_{end:.3f}_{speaker}"
        
        output['items'][item_name] = {
            'speaker': speaker,
            # 'raw_txt': item['raw_txt'],
            'raw_txt': ''.join(re.findall(r'[\u4e00-\u9fff]', item['raw_txt'])),
            'textgrid_txt': item['textgrid_txt'],
            'word': item['words'],
            'word_dur': item['word_dur'],
            'start_time': start,
            'end_time': end,
            'item_name': item_name,
            'wav_fn': f'./Spatial/drama_splitspker/{item_name}.wav'
        }
    
    return json.dumps(output, ensure_ascii=False, indent=4, separators=(',', ': '))

include_list = []

def main():
    base_dir = "./Spatial/MRSDrama2"
    output_dir = "./Spatial/drama_json"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    process_list = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_metadata.json"):
                process_list.append(os.path.join(root, file))
    # print(len(process_list))
    for json_file in tqdm(sorted(process_list), desc="Processing files", unit="file"):
        try:
            with open (json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text_grid_path = data['tg_fn']
            txt_path = data['text']
            json_output = generate_json(text_grid_path, txt_path)
            json_file_name = os.path.splitext(os.path.basename(text_grid_path))[0] + ".json"
            json_file_path = os.path.join(output_dir, json_file_name)
            with open(json_file_path, 'w', encoding='utf-8') as json_f:
                json_f.write(json_output)
            print(f"Processed {json_file_path}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
            
if __name__ == "__main__":
    main()
        