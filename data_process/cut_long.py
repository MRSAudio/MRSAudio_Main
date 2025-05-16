import os
import re
from pydub import AudioSegment
from tqdm import tqdm
import librosa
import soundfile as sf
import json
import yaml
from mfa_preprocess import mfa_lab_prepare

config_path = './Spatial/drama.yaml'
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
txt_process = config['preprocess_args']['txt_process']

long_json = "./Spatial/long_audio.json"

def parse_filename(path):
    filename = os.path.basename(path)
    pattern = r'_cut_(\d+\.\d+)_(\d+\.\d+)_'
    match = re.search(pattern, filename)
    if not match:
        raise ValueError("Invalid filename format")
    
    return float(match.group(1)), float(match.group(2))

def parse_name(name):
    pattern = r'_cut_(\d+\.\d+)_(\d+\.\d+)_'
    match = re.search(pattern, name)
    if not match:
        raise ValueError("Invalid filename format")
    
    return float(match.group(1)), float(match.group(2))
    
def find_subsequence_indices(phs, cut_phs):
    """
    在列表 phs 中找到子序列 cut_phs 的开始索引和结束索引。
    如果子序列不存在，则抛出异常。
    """
    phs_len = len(phs)
    cut_phs_len = min(len(cut_phs), 5)
    print(f"phs: {phs}")
    print(f"cut_phs: {cut_phs}")
    for i in range(phs_len - cut_phs_len + 1):
        if phs[i:i + cut_phs_len] == cut_phs[:cut_phs_len]:
            start_index = i
            end_index = i + len(cut_phs) - 1
            return start_index, end_index

    raise ValueError("子序列 cut_phs 不存在于 phs 中")

def fix_json(tg_path, original_tg_path):
    print(original_tg_path)
    origin_name = os.path.basename(original_tg_path).replace(".TextGrid", "") 
    tg_name = os.path.basename(tg_path).replace(".TextGrid", "") 
    print(f"tg_name: {tg_name}")
    print(f"origin_name: {origin_name}")
    json_path = "_".join(tg_path.replace(".TextGrid", ".json").split("_")[:-3]).replace("drama_splitspker", "drama_json").replace("drama3_splitspker", "drama_json").replace("split_test", "drama_json")+ ".json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tg_txt = data['items'][origin_name]['textgrid_txt']
    cut_phs = (parse_textgrid(tg_path)["ph"])
    cut_phs = [i for i in cut_phs if i != "<SP>"]
    words, phs, ph2word, ph_gb_word_nosil = mfa_lab_prepare(tg_txt, txt_process)
    print(f"words: {words}")
    # 在phs里面找到子序列cut_phs,保证存在
    start, end = find_subsequence_indices(phs, cut_phs)
    cut_words = "".join(words[ph2word[start]-1:ph2word[end]])
    # print(f"cut_words: {cut_words}")
    item = data['items'][origin_name]
    item['textgrid_txt'] = cut_words
    item['item_name'] = tg_name
    item['start_time'], item['end_time'] = parse_name(tg_name)
    item['word'] = words[ph2word[start]-1:ph2word[end]]
    item['word_dur'] = []
    item['wav_fn'] = tg_path.replace(".TextGrid", ".wav")
    # print(item)
    with open(long_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data[tg_name] = item
    with open(long_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
def parse_textgrid(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]

    items = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('item [') and 'item []' not in line:
            item_idx = int(line.split('[')[1].split(']')[0])
            current_item = {
                'index': item_idx,
                'class': None,
                'name': None,
                'xmin': None,
                'xmax': None,
                'intervals': []
            }
            i += 1
            while i < len(lines):
                inner_line = lines[i]
                if inner_line.startswith('class ='):
                    current_item['class'] = inner_line.split('=')[1].strip().strip('"')
                    i += 1
                elif inner_line.startswith('name ='):
                    current_item['name'] = inner_line.split('=')[1].strip().strip('"')
                    i += 1
                elif inner_line.startswith('xmin ='):
                    current_item['xmin'] = float(inner_line.split('=')[1].strip())
                    i += 1
                elif inner_line.startswith('xmax ='):
                    current_item['xmax'] = float(inner_line.split('=')[1].strip())
                    i += 1
                elif inner_line.startswith('intervals: size ='):
                    size = int(inner_line.split('=')[1].strip())
                    current_item['intervals_size'] = size
                    i += 1
                    intervals = []
                    for interval_num in range(1, size + 1):
                        while i < len(lines) and not lines[i].startswith(f'intervals [{interval_num}]:'):
                            i += 1
                        if i >= len(lines):
                            break
                        interval = {'xmin': None, 'xmax': None, 'text': ''}
                        i += 1
                        attrs_collected = 0
                        while i < len(lines) and attrs_collected < 3:
                            attr_line = lines[i]
                            if attr_line.startswith('xmin ='):
                                interval['xmin'] = float(attr_line.split('=')[1].strip())
                                attrs_collected += 1
                                i += 1
                            elif attr_line.startswith('xmax ='):
                                interval['xmax'] = float(attr_line.split('=')[1].strip())
                                attrs_collected += 1
                                i += 1
                            elif attr_line.startswith('text ='):
                                text = attr_line.split('=', 1)[1].strip().strip('"')
                                interval['text'] = text
                                attrs_collected += 1
                                i += 1
                            else:
                                break
                        intervals.append(interval)
                    current_item['intervals'] = intervals
                    break
                else:
                    i += 1
            items.append(current_item)
        else:
            i += 1

    words_tier = next((item for item in items if item.get('name') == 'words'), None)
    phones_tier = next((item for item in items if item.get('name') == 'phones'), None)

    if not words_tier or not phones_tier:
        raise ValueError("Required tiers 'words' or 'phones' not found.")

    def process_tier(tier):
        texts = []
        durations = []
        for interval in tier['intervals']:
            text = interval['text'].strip()
            texts.append("<SP>" if text == "" else text)
            durations.append(round(interval['xmax'] - interval['xmin'], 3))
        return texts, durations

    word_list, word_dur = process_tier(words_tier)
    ph_list, ph_dur = process_tier(phones_tier)

    return {
        "word": word_list,
        "word_dur": word_dur,
        "ph": ph_list,
        "ph_dur": ph_dur
    }

def process_audio_segment(wav_path, start_time, end_time, output_path):
    """
    使用soundfile和librosa处理音频切割
    保持原始格式和采样率
    """
    long_start, long_end = parse_filename(wav_path)
    start_time -= long_start
    end_time -= long_start
    print(f"Processing {wav_path} from {start_time} to {end_time} to {output_path}")
    # 读取音频元数据
    with sf.SoundFile(wav_path) as f:
        sr = f.samplerate
        subtype = f.subtype
        channels = f.channels
    
    # 计算采样点
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    # print(f"start_sample: {start_sample}, end_sample: {end_sample}")
    
    # 使用librosa加载（保持原始采样率）
    y, sr = librosa.load(wav_path, sr=sr, mono=False, offset=start_time, duration=end_time-start_time)
    # print(f"y.shape: {y.shape}, sr: {sr}")
    
    # 保持多声道格式
    if channels > 1:
        y = y.T  # librosa返回(chn, samples)，需要转置为(samples, chn)
    
    # 保持原始格式写入
    sf.write(output_path, y, sr, subtype=subtype, format='WAV')

    
def split_audio_and_textgrid(wav_path, tg_path, base_tg):
    # 解析文件名中的起始和结束时
    start_time, end_time = parse_filename(wav_path)
    duration = end_time - start_time
    
    # 如果时长不超过30秒，直接返回
    if duration <= 30:
        return [(wav_path, tg_path)]
    
    # 解析TextGrid
    tg_data = parse_textgrid(tg_path)
    
    # 重构intervals（补充xmin/xmax信息）
    def reconstruct_intervals(texts, durations):
        intervals = []
        xmin = 0.0
        for text, dur in zip(texts, durations):
            xmax = xmin + dur
            intervals.append({'xmin': xmin, 'xmax': xmax, 'text': text})
            xmin = xmax
        return intervals
    
    words_intervals = reconstruct_intervals(tg_data['word'], tg_data['word_dur'])
    phones_intervals = reconstruct_intervals(tg_data['ph'], tg_data['ph_dur'])
    
    # 寻找切割点（最后一个在30秒前的<SP>）
    def find_cut_point(intervals):
        candidates = [i['xmax'] for i in intervals 
                    if i['text'] == '<SP>' and i['xmax'] <= 30]
        if not candidates:
            raise ValueError("No valid cut point found")
        return max(candidates)
    
    cut_time = find_cut_point(words_intervals)
    
    # 生成新的文件名
    def new_path(original, new_start, new_end):
        base = os.path.splitext(original)[0]
        new_base = re.sub(r'_cut_(\d+\.\d+)_(\d+\.\d+)_', 
                        f'_cut_{new_start:.3f}_{new_end:.3f}_', base)
        return f"{new_base}.wav".replace("too_long", "split_test").replace("drama3_splitspker", "split_test"), f"{new_base}.TextGrid".replace("too_long", "split_test").replace("drama3_splitspker", "split_test")
    
    # 前段文件
    front_end = start_time + cut_time
    front_wav, front_tg = new_path(wav_path, start_time, front_end)
    print(front_wav)
    
    # 切割音频
    def export_audio_segment(input_path, output_path, start, end):
        process_audio_segment(input_path, start, end, output_path)
        
        
    export_audio_segment(wav_path, front_wav, start_time, front_end)
    
    # 生成前段TextGrid
    def generate_textgrid(word_intervals, phones_intervals, cut_time, duration):
        content = [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            '',
            f'xmin = 0',
            f'xmax = {duration}',
            'tiers? <exists>',
            'size = 2',
            'item []:',
            build_tier([i for i in word_intervals if i['xmax'] <= cut_time], "words", 1),
            build_tier([i for i in phones_intervals if i['xmax'] <= cut_time], "phones", 2)
        ]
        return '\n'.join(content)
    
    def build_tier(intervals, name, index):
        tier = [
            f'    item [{index}]:',
            '        class = "IntervalTier"',
            f'        name = "{name}"',
            '        xmin = 0',
            f'        xmax = {intervals[-1]["xmax"] if intervals else 0}',
            f'        intervals: size = {len(intervals)}'
        ]
        for idx, interval in enumerate(intervals, 1):
            tier.extend([
                f'        intervals [{idx}]:',
                f'            xmin = {interval["xmin"]}',
                f'            xmax = {interval["xmax"]}',
                f'            text = "{interval["text"]}"'
            ])
        return '\n'.join(tier)
    current = generate_textgrid(words_intervals, phones_intervals, cut_time, cut_time)
    
    with open(front_tg, 'w', encoding='utf-8') as f:
        f.write(current)
    # fix_json for current
    fix_json(front_tg, base_tg)
    # 处理剩余部分
    remaining_start = front_end
    remaining_wav, remaining_tg = new_path(wav_path, remaining_start, end_time)
    print(f"Processing remaining {remaining_wav} from {remaining_start} to {end_time}")
    
    export_audio_segment(wav_path, remaining_wav, remaining_start, end_time)
    remaining_duration = end_time - remaining_start
    
    # 生成剩余TextGrid
    def generate_remaining(intervals, cut_time, duration):
        remaining = []
        for i in intervals:
            if i['xmin'] >= cut_time:
                remaining.append({
                    'xmin': i['xmin'] - cut_time,
                    'xmax': i['xmax'] - cut_time,
                    'text': i['text']
                })
        return remaining
    
    remaining_words = generate_remaining(words_intervals, cut_time, remaining_duration)
    remaining_phones = generate_remaining(phones_intervals, cut_time, remaining_duration)
    
    remain = generate_textgrid(remaining_words, remaining_phones, remaining_duration, remaining_duration)
    with open(remaining_tg, 'w', encoding='utf-8') as f:
        f.write(remain)
    fix_json(remaining_tg, base_tg)
    # 递归处理
    result = [(front_wav, front_tg)]
    if remaining_duration > 30:
        result += split_audio_and_textgrid(remaining_wav, remaining_tg, base_tg)
    else:
        result.append((remaining_wav, remaining_tg))
    return result

# 使用示例
with open("./Spatial/long_audio_new.json", "r") as f:
    data = json.load(f)
for key, value in tqdm(reversed(list(data.items()))):
    wav_path = value['wav_fn'].replace("drama_splitspker", "too_long")
    tg_path = wav_path.replace("drama_splitspker", "too_long").replace(".wav", ".TextGrid").replace("drama3_splitspker", "drama3_mfa")
    base_tg = tg_path
    try:
        result = split_audio_and_textgrid(wav_path, tg_path, base_tg)
        print(result)
    except Exception as e:
        print(f"Error processing {key}: {e}")