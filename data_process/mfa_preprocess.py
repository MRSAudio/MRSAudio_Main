import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import os
import yaml
from utils.text.zh_text_norm import NSWNormalizer
import re
from utils.text.text_encoder import PUNCS
from utils.text.text_encoder import is_sil_phoneme
from pypinyin import pinyin, Style
import librosa
from pydub import AudioSegment
import torchaudio
import pyloudnorm as pyln
import torch
import json
import glob
from multiprocessing import Pool, cpu_count
from functools import partial


ERROR_AUDIO = []

class TxtProcessor:
    table = {ord(f): ord(t) for f, t in zip(
        u'：，。！？【】（）％＃＠＆１２３４５６７８９０',
        u':,.!?[]()%#@&1234567890')}

    @classmethod
    def postprocess(cls, txt_struct, preprocess_args):
        if preprocess_args['with_phsep']:
            txt_struct = cls.add_bdr(txt_struct)
        if preprocess_args['add_eos_bos']:
            # remove sil phoneme in head and tail
            while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[0][0]):
                txt_struct = txt_struct[1:]
            while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[-1][0]):
                txt_struct = txt_struct[:-1]
            txt_struct = [["<BOS>", ["<BOS>"]]] + txt_struct + [["<EOS>", ["<EOS>"]]]
        return txt_struct

    @classmethod
    def add_bdr(cls, txt_struct):
        txt_struct_ = []
        for i, ts in enumerate(txt_struct):
            txt_struct_.append(ts)
            if i != len(txt_struct) - 1 and \
                    not is_sil_phoneme(txt_struct[i][0]) and not is_sil_phoneme(txt_struct[i + 1][0]):
                txt_struct_.append(['|', ['|']])
        return txt_struct_

    @staticmethod
    def sp_phonemes():
        return ['|', '#']

    @staticmethod
    def preprocess_text(text):
        text = text.translate(TxtProcessor.table)
        text = NSWNormalizer(text).normalize(remove_punc=False).lower()
        text = re.sub("[\'\"()]+", "", text)
        text = re.sub("[-]+", " ", text)
        text = re.sub(f"[^ A-Za-z\u4e00-\u9fff{PUNCS}]", "", text)
        text = re.sub(f"([{PUNCS}])+", r"\1", text)  # !! -> !
        text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        text = re.sub(rf"\s+", r"", text)
        text = re.sub(rf"[A-Za-z]+", r"$", text)
        return text

    @classmethod
    def pinyin_with_en(cls, txt, style):
        x = pinyin(txt, style)
        x = [t[0] for t in x]
        x_ = []
        for t in x:
            if '$' not in t:
                x_.append(t)
            else:
                x_ += list(t)
        x_ = [t if t != '$' else 'LANG-ENG' for t in x_]
        x_ = [t if t != '&' else 'BREATHE' for t in x_]
        x_ = [t if t != '@' else '<SEP>' for t in x_]
        return x_

    @classmethod
    def process(cls, txt, preprocess_args):
        txt = cls.preprocess_text(txt)

        # https://blog.csdn.net/zhoulei124/article/details/89055403
        shengmu = cls.pinyin_with_en(txt, style=Style.INITIALS)
        yunmu = cls.pinyin_with_en(txt, style=Style.FINALS_TONE3 if preprocess_args['use_tone'] else Style.FINALS)
        assert len(shengmu) == len(yunmu)
        phs = []
        for a, b in zip(shengmu, yunmu):
            if a == b:
                phs += [a]
            else:
                phs += [a + "%" + b]
        if preprocess_args['use_char_as_word']:
            words = list(txt)
        else:
            words = jieba.cut(txt)
        txt_struct = [[w, []] for w in words]
        i_ph = 0
        for ts in txt_struct:
            ts[1] = [ph for char_pinyin in phs[i_ph:i_ph + len(ts[0])]
                     for ph in char_pinyin.split("%") if ph != '']
            i_ph += len(ts[0])
        txt_struct = cls.postprocess(txt_struct, preprocess_args)
        return txt_struct, txt
def process_content(content):
    processed_content = content.strip().replace('嗯', '蒽').replace('$', '').replace('哟', '优')
    processed_content = re.sub(r'[，。！？.,!?]', '', processed_content)
    return processed_content

def mfa_lab_prepare(text, txt_process):
    # text 姑 少 爷 您 还 到 哪 儿 去 不 早 
    # g_u sh_ao ie n_i
    txt_struct, txt = TxtProcessor.process(text, txt_process)
    phs = [p for w in txt_struct for p in w[1]]
    ph_gb_word = ["_".join(w[1]) for w in txt_struct]
    words = [w[0] for w in txt_struct]
    ph2word = [w_id + 1 for w_id, w in enumerate(txt_struct) for _ in range(len(w[1]))]
    ph = " ".join(phs) 
    word = " ".join(words)
    ph_gb_word = " ".join(ph_gb_word)
    ph_gb_word_nosil = " ".join(["_".join([p for p in w.split("_") if not is_sil_phoneme(p)])
                                    for w in ph_gb_word.split(" ") if not is_sil_phoneme(w)])
    # mfa是否使用声调
    if not txt_process['mfa_use_tone']:
        ph_gb_word_nosil = re.sub(r'\d', '', ph_gb_word_nosil)

    return words, phs, ph2word, ph_gb_word_nosil

def prepare_align_worker(wav_file, config):
    """处理单个文件的worker函数"""
    try:
        in_dir = config["path"]["corpus_path"]
        mfa_input = config["path"]["mfa_input"]
        sampling_rate = config["preprocess_args"]["audio"]["sampling_rate"]
        target_loudness = config["preprocess_args"]["audio"]["target_loudness"]
        txt_process = config['preprocess_args']['txt_process']
        mfa_dict_fn = config["mfa"]["dict_fn"]

        with open(mfa_dict_fn, encoding="utf-8") as f:
            mfa_list = set(line.split("\t")[0] for line in f)

        dir_name = os.path.dirname(os.path.dirname(wav_file))
        item_name = os.path.basename(wav_file).split('.')[0]
        
        json_files = glob.glob(f'{dir_name}/*cut_new.json')
        if len(json_files) != 1:
            return {"status": "error", "file": wav_file, "reason": "JSON文件数量异常"}
        
        json_file = json_files[0]
        data = json.load(open(json_file,'r'))
        key_name = os.path.basename(wav_file).replace('.wav', '')
        
        if key_name not in data:
            return {"status": "error", "file": wav_file, "reason": "JSON中缺少对应key"}
        
        content = data[key_name]['text']
        speaker = data[key_name]['speaker']

        # 预处理文本内容
        english_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        if any(c in english_chars for c in content) or not content.strip():
            return {"status": "skipped", "file": wav_file, "reason": "包含英文字符或内容为空"}
        
        # 文本处理流程
        processed_content = process_content(content)
        
        # 生成音素标签
        _, _, _, ph_gb_word_nosil = mfa_lab_prepare(processed_content, txt_process)
        
        # 保存标签文件
        lab_path = os.path.join(mfa_input, f"{key_name}.lab")
        with open(lab_path, "w") as f:
            f.write(ph_gb_word_nosil)
        
        # 处理音频文件
        target_wav_path = os.path.join(mfa_input, f"{key_name}.wav")
        if not os.path.exists(target_wav_path):
            normalize_loudness(wav_file, target_loudness, target_wav_path, sampling_rate)
            
            return {"status": "success", "file": wav_file}
        else:
            return {"status": "skipped", "file": wav_file, "reason": "音频文件已存在"}

    except Exception as e:
        return {"status": "error", "file": wav_file, "reason": str(e)}

def prepare_align(config):
    """并行处理主函数"""
    in_dir = config["path"]["corpus_path"]
    mfa_input = config["path"]["mfa_input"]
    os.makedirs(mfa_input, exist_ok=True)

    # 获取待处理文件列表
    wav_files = glob.glob(os.path.join(in_dir, '**', 'split_spker', '*.wav'), recursive=True)
    wav_files = [f for f in wav_files if os.path.basename(f).split('_')[0] not in ERROR_AUDIO]
    print(f"发现 {len(wav_files)} 个待处理音频文件")

    # 创建进程池
    workers = max(1, int(cpu_count() * 0.75))  # 使用75%的CPU核心
    results = []
    
    with Pool(processes=workers) as pool:
        # 准备并行任务
        task = partial(prepare_align_worker, config=config)
        
        # 使用imap_unordered获取更快进度反馈
        with tqdm(total=len(wav_files), desc="处理进度") as pbar:
            for result in pool.imap_unordered(task, wav_files):
                results.append(result)
                pbar.update(1)
                pbar.set_postfix({
                    "成功": sum(1 for r in results if r["status"] == "success"),
                    "跳过": sum(1 for r in results if r["status"] == "skipped"),
                    "错误": sum(1 for r in results if r["status"] == "error")
                })

    # 输出统计信息
    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    
    print("\n处理完成:")
    print(f"成功: {success} 个文件")
    print(f"跳过: {skipped} 个文件")
    print(f"错误: {errors} 个文件")

    # 输出错误日志
    if errors > 0:
        print("\n错误详情（前20个）:")
        for r in results[:20]:
            if r["status"] == "error":
                print(f"文件: {r['file']}\n原因: {r['reason']}\n{'-'*50}")


def normalize_loudness(wav_path, target_loudness, save_path, target_sample_rate=None):
    # 加载 WAV 文件
    wav, sr = torchaudio.load(wav_path)
    
    # 如果目标采样率不为空，重新采样
    if target_sample_rate is not None and sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        wav = resampler(wav)
        sr = target_sample_rate  # 更新采样率
        
    if wav.shape[0] != 1:  # 如果是立体声，转换为单声道
        wav = wav.mean(0, keepdim=True)
    
    # 将 tensor 转换为 numpy 格式
    wav_np = wav.squeeze(0).cpu().numpy()

    # 创建一个用于响度计算的 meter 对象
    meter = pyln.Meter(sr)  # sr 是采样率

    # 计算当前 WAV 文件的整体响度 (LUFS)
    loudness = meter.integrated_loudness(wav_np)

    # 将音频响度归一化到目标响度
    normalized_wav_np = pyln.normalize.loudness(wav_np, loudness, target_loudness)

    loudness = meter.integrated_loudness(normalized_wav_np)
    # 转换回 torch tensor 格式
    normalized_wav = torch.tensor(normalized_wav_np).unsqueeze(0)

    # 保存归一化后的音频
    torchaudio.save(save_path, normalized_wav, sr)
    
if __name__ == "__main__":
    config_path = './Spatial/play.yaml'
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    prepare_align(config)