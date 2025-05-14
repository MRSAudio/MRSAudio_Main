import whisperx
import librosa
import re
import os
from audio_cut import check_single
import json
from tqdm import tqdm
from whisperx.alignment import *

def get_audio_duration(file_path):
    duration = librosa.get_duration(filename=file_path)
    return duration

def get_lyric(input_file):
    # 匹配对话行的正则表达式
    pattern = re.compile(r'^([A-Z]):\s+(.*)$')
    result = []
    
    # 汉字Unicode范围：\u4e00-\u9fff
    # 扩展正则表达式包含所有非汉字字符（包括标点、字母、数字等）
    chinese_only_pattern = re.compile(r'[^\u4e00-\u9fff]')

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行

            # 匹配对话行
            match = pattern.match(line)
            if match:
                content = match.group(2)  # 提取对话内容
                
                # 删除所有非汉字字符（包括标点符号）
                cleaned_content = chinese_only_pattern.sub('', content)
                
                if cleaned_content:  # 只保留有效内容
                    result.append(cleaned_content)
    
    # 合并所有内容为单个字符串
    return ''.join(result)

def align_long_overlap(
    transcript: Iterable[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor],
    device: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.
    """
    
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    
    MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    # 1. Preprocess to keep only characters in dictionary
    total_segments = len(transcript)
    for sdx, segment in enumerate(transcript):
        # strip spaces at beginning / end, but keep track of the amount.
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")
            
        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # split into words
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = text

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
            char_ = char.lower()
            # wav2vec2 models use "|" character to represent spaces
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                char_ = char_.replace(" ", "|")
            
            # ignore whitespace at beginning and end of transcript
            if cdx < num_leading:
                pass
            elif cdx > len(text) - num_trailing - 1:
                pass
            elif char_ in model_dictionary.keys():
                clean_char.append(char_)
                clean_cdx.append(cdx)

        clean_wdx = []
        for wdx, wrd in enumerate(per_word):
            if any([c in model_dictionary.keys() for c in wrd]):
                clean_wdx.append(wdx)

                
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment["clean_char"] = clean_char
        segment["clean_cdx"] = clean_cdx
        segment["clean_wdx"] = clean_wdx
        segment["sentence_spans"] = sentence_spans
    
    aligned_segments: List[SingleAlignedSegment] = []
    
    # 2. Get prediction matrix from alignment model & align
    for sdx, segment in enumerate(transcript):
        
        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]

        aligned_seg: SingleAlignedSegment = {
            "start": t1,
            "end": t2,
            "text": text,
            "words": [],
        }

        if return_char_alignments:
            aligned_seg["chars"] = []

        # check we can align
        if len(segment["clean_char"]) == 0:
            print(f'Failed to align segment ("{segment["text"]}"): no characters in this segment found in model dictionary, resorting to original...')
            aligned_segments.append(aligned_seg)
            continue

        if t1 >= MAX_DURATION:
            print(f'Failed to align segment ("{segment["text"]}"): original start time longer than audio duration, skipping...')
            aligned_segments.append(aligned_seg)
            continue

        text_clean = "".join(segment["clean_char"])
        tokens = [model_dictionary[c] for c in text_clean]

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)

        # TODO: Probably can get some speedup gain with batched inference here
        waveform_segment = audio[:, f1:f2]
        chunk_seconds = 20  # 主块长度（秒）
        overlap_seconds = 2  # 重叠秒数
        downsample_rate = 320
        # 确保分块长度是下采样率的整数倍
        chunk_samples = int(chunk_seconds * SAMPLE_RATE)
        chunk_samples = (chunk_samples // downsample_rate) * downsample_rate
        
        overlap_samples = int(overlap_seconds * SAMPLE_RATE)
        overlap_samples = (overlap_samples // downsample_rate) * downsample_rate

        emissions_list = []
        total_samples = waveform_segment.shape[-1]
        with torch.inference_mode():
            for start_idx in range(0, total_samples, chunk_samples):
                end_idx = min(start_idx + chunk_samples, total_samples)
                # 提取带重叠的音频块
                chunk_start = max(0, start_idx - overlap_samples)
                chunk = waveform_segment[:, chunk_start:end_idx]
                pad_length = (downsample_rate - (chunk.size(-1) % downsample_rate)) % downsample_rate
                if pad_length > 0:
                    chunk = torch.nn.functional.pad(chunk, (0, pad_length))
                if model_type == "torchaudio":
                    chunk_emissions, _ = model(chunk.to(device))
                elif model_type == "huggingface":
                    # 处理最小输入长度要求
                    chunk_emissions = model(chunk.to(device)).logits
                else:
                    raise NotImplementedError(f"Align model of type {model_type} not supported.")
                # 计算需要保留的输出帧数（关键修正）
                valid_output_frames = min(chunk_samples, total_samples-start_idx) // downsample_rate
                
                # 截取有效输出（考虑重叠部分）
                if start_idx > 0:
                    # 前向重叠部分的输出帧数
                    overlap_frames = overlap_samples // downsample_rate
                    chunk_emissions = chunk_emissions[:, overlap_frames:]
                
                # 确保不超过实际需要的帧数
                chunk_emissions = chunk_emissions[:, :valid_output_frames]
                # print(chunk_emissions.shape)
                emissions_list.append(chunk_emissions.cpu())
            # 合并所有分块结果并移回GPU
            emissions = torch.cat(emissions_list, dim=1).to(device)
            emissions = torch.log_softmax(emissions, dim=-1)
        emission = emissions[0].cpu().detach()
        blank_id = 0
        for char, code in model_dictionary.items():
            if char == '[pad]' or char == '<pad>':
                blank_id = code

        trellis = get_trellis(emission, tokens, blank_id)
        path = backtrack(trellis, emission, tokens, blank_id)

        if path is None:
            print(f'Failed to align segment ("{segment["text"]}"): backtrack failed, resorting to original...')
            aligned_segments.append(aligned_seg)
            continue

        char_segments = merge_repeats(path, text_clean)

        duration = t2 -t1
        ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)

        # assign timestamps to aligned characters
        char_segments_arr = []
        word_idx = 0
        for cdx, char in enumerate(text):
            start, end, score = None, None, None
            if cdx in segment["clean_cdx"]:
                char_seg = char_segments[segment["clean_cdx"].index(cdx)]
                start = round(char_seg.start * ratio + t1, 3)
                end = round(char_seg.end * ratio + t1, 3)
                score = round(char_seg.score, 3)

            char_segments_arr.append(
                {
                    "char": char,
                    "start": start,
                    "end": end,
                    "score": score,
                    "word-idx": word_idx,
                }
            )

            # increment word_idx, nltk word tokenization would probably be more robust here, but us space for now...
            if model_lang in LANGUAGES_WITHOUT_SPACES:
                word_idx += 1
            elif cdx == len(text) - 1 or text[cdx+1] == " ":
                word_idx += 1
            
        char_segments_arr = pd.DataFrame(char_segments_arr)

        aligned_subsegments = []
        # assign sentence_idx to each character index
        char_segments_arr["sentence-idx"] = None
        for sdx, (sstart, send) in enumerate(segment["sentence_spans"]):
            curr_chars = char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)]
            char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send), "sentence-idx"] = sdx
        
            sentence_text = text[sstart:send]
            sentence_start = curr_chars["start"].min()
            end_chars = curr_chars[curr_chars["char"] != ' ']
            sentence_end = end_chars["end"].max()
            sentence_words = []

            for word_idx in curr_chars["word-idx"].unique():
                word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
                word_text = "".join(word_chars["char"].tolist()).strip()
                if len(word_text) == 0:
                    continue

                # dont use space character for alignment
                word_chars = word_chars[word_chars["char"] != " "]

                word_start = word_chars["start"].min()
                word_end = word_chars["end"].max()
                word_score = round(word_chars["score"].mean(), 3)

                # -1 indicates unalignable 
                word_segment = {"word": word_text}

                if not np.isnan(word_start):
                    word_segment["start"] = word_start
                if not np.isnan(word_end):
                    word_segment["end"] = word_end
                if not np.isnan(word_score):
                    word_segment["score"] = word_score

                sentence_words.append(word_segment)
            
            aligned_subsegments.append({
                "text": sentence_text,
                "start": sentence_start,
                "end": sentence_end,
                "words": sentence_words,
            })

            if return_char_alignments:
                curr_chars = curr_chars[["char", "start", "end", "score"]]
                curr_chars.fillna(-1, inplace=True)
                curr_chars = curr_chars.to_dict("records")
                curr_chars = [{key: val for key, val in char.items() if val != -1} for char in curr_chars]
                aligned_subsegments[-1]["chars"] = curr_chars

        aligned_subsegments = pd.DataFrame(aligned_subsegments)
        aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"], method=interpolate_method)
        aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"], method=interpolate_method)
        # concatenate sentences with same timestamps
        agg_dict = {"text": " ".join, "words": "sum"}
        if model_lang in LANGUAGES_WITHOUT_SPACES:
            agg_dict["text"] = "".join
        if return_char_alignments:
            agg_dict["chars"] = "sum"
        aligned_subsegments= aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
        aligned_subsegments = aligned_subsegments.to_dict('records')
        aligned_segments += aligned_subsegments

    # create word_segments list
    word_segments: List[SingleWordSegment] = []
    for segment in aligned_segments:
        word_segments += segment["words"]

    return {"segments": aligned_segments, "word_segments": word_segments}

def align_long(
    transcript: Iterable[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor],
    device: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.
    """
    
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    
    MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    # 1. Preprocess to keep only characters in dictionary
    total_segments = len(transcript)
    for sdx, segment in enumerate(transcript):
        # strip spaces at beginning / end, but keep track of the amount.
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")
            
        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # split into words
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = text

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
            char_ = char.lower()
            # wav2vec2 models use "|" character to represent spaces
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                char_ = char_.replace(" ", "|")
            
            # ignore whitespace at beginning and end of transcript
            if cdx < num_leading:
                pass
            elif cdx > len(text) - num_trailing - 1:
                pass
            elif char_ in model_dictionary.keys():
                clean_char.append(char_)
                clean_cdx.append(cdx)

        clean_wdx = []
        for wdx, wrd in enumerate(per_word):
            if any([c in model_dictionary.keys() for c in wrd]):
                clean_wdx.append(wdx)

                
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment["clean_char"] = clean_char
        segment["clean_cdx"] = clean_cdx
        segment["clean_wdx"] = clean_wdx
        segment["sentence_spans"] = sentence_spans
    
    aligned_segments: List[SingleAlignedSegment] = []
    
    # 2. Get prediction matrix from alignment model & align
    for sdx, segment in enumerate(transcript):
        
        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]

        aligned_seg: SingleAlignedSegment = {
            "start": t1,
            "end": t2,
            "text": text,
            "words": [],
        }

        if return_char_alignments:
            aligned_seg["chars"] = []

        # check we can align
        if len(segment["clean_char"]) == 0:
            print(f'Failed to align segment ("{segment["text"]}"): no characters in this segment found in model dictionary, resorting to original...')
            aligned_segments.append(aligned_seg)
            continue

        if t1 >= MAX_DURATION:
            print(f'Failed to align segment ("{segment["text"]}"): original start time longer than audio duration, skipping...')
            aligned_segments.append(aligned_seg)
            continue

        text_clean = "".join(segment["clean_char"])
        tokens = [model_dictionary[c] for c in text_clean]

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)

        # TODO: Probably can get some speedup gain with batched inference here
        waveform_segment = audio[:, f1:f2]
        # 分块处理长音频
        chunk_length = 30 * SAMPLE_RATE  # 30秒的样本数（假设16kHz采样率）
        emissions_list = []
        total_samples = waveform_segment.shape[-1]
        with torch.inference_mode():
            for start in range(0, total_samples, chunk_length):
                end = min(start + chunk_length, total_samples)
                chunk = waveform_segment[:, start:end]
                chunk_samples = chunk.size(1)
                if chunk_samples <400:
                    chunk = torch.nn.functional.pad(chunk, (0, 400 - chunk_samples))
                    lengths = torch.as_tensor([chunk_samples]).to(device)
                else:
                    lengths = None
                if model_type == "torchaudio":
                    chunk_emissions, _ = model(chunk.to(device), lengths=lengths)
                elif model_type == "huggingface":
                    # 处理最小输入长度要求
                    chunk_emissions = model(chunk.to(device)).logits
                else:
                    raise NotImplementedError(f"Align model of type {model_type} not supported.")
                # print(chunk_emissions.shape)
                emissions_list.append(chunk_emissions.cpu())  # 转移至CPU保存
            # 合并所有分块结果并移回GPU
            emissions = torch.cat(emissions_list, dim=1).to(device)
            emissions = torch.log_softmax(emissions, dim=-1)
        emission = emissions[0].cpu().detach()
        blank_id = 0
        for char, code in model_dictionary.items():
            if char == '[pad]' or char == '<pad>':
                blank_id = code

        trellis = get_trellis(emission, tokens, blank_id)
        path = backtrack(trellis, emission, tokens, blank_id)

        if path is None:
            print(f'Failed to align segment ("{segment["text"]}"): backtrack failed, resorting to original...')
            aligned_segments.append(aligned_seg)
            continue

        char_segments = merge_repeats(path, text_clean)

        duration = t2 -t1
        ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)

        # assign timestamps to aligned characters
        char_segments_arr = []
        word_idx = 0
        for cdx, char in enumerate(text):
            start, end, score = None, None, None
            if cdx in segment["clean_cdx"]:
                char_seg = char_segments[segment["clean_cdx"].index(cdx)]
                start = round(char_seg.start * ratio + t1, 3)
                end = round(char_seg.end * ratio + t1, 3)
                score = round(char_seg.score, 3)

            char_segments_arr.append(
                {
                    "char": char,
                    "start": start,
                    "end": end,
                    "score": score,
                    "word-idx": word_idx,
                }
            )

            # increment word_idx, nltk word tokenization would probably be more robust here, but us space for now...
            if model_lang in LANGUAGES_WITHOUT_SPACES:
                word_idx += 1
            elif cdx == len(text) - 1 or text[cdx+1] == " ":
                word_idx += 1
            
        char_segments_arr = pd.DataFrame(char_segments_arr)

        aligned_subsegments = []
        # assign sentence_idx to each character index
        char_segments_arr["sentence-idx"] = None
        for sdx, (sstart, send) in enumerate(segment["sentence_spans"]):
            curr_chars = char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)]
            char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send), "sentence-idx"] = sdx
        
            sentence_text = text[sstart:send]
            sentence_start = curr_chars["start"].min()
            end_chars = curr_chars[curr_chars["char"] != ' ']
            sentence_end = end_chars["end"].max()
            sentence_words = []

            for word_idx in curr_chars["word-idx"].unique():
                word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
                word_text = "".join(word_chars["char"].tolist()).strip()
                if len(word_text) == 0:
                    continue

                # dont use space character for alignment
                word_chars = word_chars[word_chars["char"] != " "]

                word_start = word_chars["start"].min()
                word_end = word_chars["end"].max()
                word_score = round(word_chars["score"].mean(), 3)

                # -1 indicates unalignable 
                word_segment = {"word": word_text}

                if not np.isnan(word_start):
                    word_segment["start"] = word_start
                if not np.isnan(word_end):
                    word_segment["end"] = word_end
                if not np.isnan(word_score):
                    word_segment["score"] = word_score

                sentence_words.append(word_segment)
            
            aligned_subsegments.append({
                "text": sentence_text,
                "start": sentence_start,
                "end": sentence_end,
                "words": sentence_words,
            })

            if return_char_alignments:
                curr_chars = curr_chars[["char", "start", "end", "score"]]
                curr_chars.fillna(-1, inplace=True)
                curr_chars = curr_chars.to_dict("records")
                curr_chars = [{key: val for key, val in char.items() if val != -1} for char in curr_chars]
                aligned_subsegments[-1]["chars"] = curr_chars

        aligned_subsegments = pd.DataFrame(aligned_subsegments)
        aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"], method=interpolate_method)
        aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"], method=interpolate_method)
        # concatenate sentences with same timestamps
        agg_dict = {"text": " ".join, "words": "sum"}
        if model_lang in LANGUAGES_WITHOUT_SPACES:
            agg_dict["text"] = "".join
        if return_char_alignments:
            agg_dict["chars"] = "sum"
        aligned_subsegments= aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
        aligned_subsegments = aligned_subsegments.to_dict('records')
        aligned_segments += aligned_subsegments

    # create word_segments list
    word_segments: List[SingleWordSegment] = []
    for segment in aligned_segments:
        word_segments += segment["words"]

    return {"segments": aligned_segments, "word_segments": word_segments}

device = "cuda" 
align_model = '/home2/zhangyu/gwx/MRSAudio/data_process/ckpts/wav2vec2-large-xlsr-53-chinese-zh-cn'

def align(audio_file, lyric_file, target_tg):
    lyric = get_lyric(lyric_file)
    # print(lyric)
    audio = whisperx.load_audio(audio_file)
    # 2. Align whisper output

    xmax = get_audio_duration(audio_file)
    input_format=[{'text': lyric, 'start': 0., 'end': xmax}]
    model_a, metadata = whisperx.load_align_model(model_name=align_model, language_code='zh', device=device)
    result = align_long_overlap(input_format, model_a, metadata, audio, device, return_char_alignments=False)

    print(result["segments"][0].keys()) # after alignment
    from textgrid import TextGrid, IntervalTier, Interval
    from collections import defaultdict

    # 创建TextGrid对象
    tg = TextGrid(maxTime=xmax)  # 关键参数设置
    intervals = defaultdict(list)
    tier = IntervalTier(name="words", minTime=0, maxTime=xmax)
    prev_end = 0.0
    for idx, word in enumerate(result["segments"][0]['words']):
        text = word['word']
        if 'start' not in word:
            continue
        start = word['start']
        end = word['end']
        if start > prev_end:
            blank = Interval(prev_end, start, "")
            tier.addInterval(blank)  # 正确调用方式
        # 添加实际语音间隔
        speech = Interval(start, end, text)
        tier.addInterval(speech)
        prev_end = end
    # 尾部空白处理
    if prev_end < xmax:
        tail_blank = Interval(prev_end, xmax, "")
        tier.addInterval(tail_blank)
    tg.append(tier)
    # 保存TextGrid文件
    tg.write(target_tg)
    
    
include_list = []
def main():
    base_dir = "./Spatial/MRSDrama"
    process_list = []
    ignore_list = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # if file.endswith("_缩混_cut.wav") and not any(ignore in file for ignore in ignore_list) and (any (x in root for x in include_list)):
            if file.endswith("_缩混_cut.wav") and not any(ignore in file for ignore in ignore_list):
                process_list.append(os.path.join(root, file))
                
    for file in tqdm(sorted(process_list)):
        _dir = os.path.dirname(file)
        print(_dir)
        root = _dir
        json_file = check_single([os.path.join(root, f) for f in os.listdir(_dir) if f.endswith("_metadata.json")], "json")
        # print(json_file)
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        audio_file = os.path.join(root, file)
        text_file = check_single([os.path.join(root, f) for f in os.listdir(_dir) if f.endswith(".txt") and "log" not in f], "text")
        audio_file = json_data['wav_fn']
        text_file = json_data['text']
        if not os.path.exists(json_data['tg_fn']):
            print('tg_fn:', json_data['tg_fn'])
            align(audio_file, text_file, json_data['tg_fn'])
                
if __name__ == "__main__":
    main()