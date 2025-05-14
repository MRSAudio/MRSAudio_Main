import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import os
import yaml
from utils.text.zh_text_norm import NSWNormalizer
import re
from utils.text.text_encoder import PUNCS
from utils.text.text_encoder import is_sil_phoneme
import os
from pypinyin import pinyin, Style
import librosa
from pydub import AudioSegment
import torchaudio
import pyloudnorm as pyln
import torch
import json
import glob

ERROR_AUDIO = [
]
json_base_dir = './Spatial/drama3_json'
actor_list = ["A:", "B:", "C:", "D:"]
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

def clean_content(content):
    content = content.strip().replace('嗯', '蒽').replace('$', '').replace('b', '比').replace('B', '比').replace('R','啊').replace('r','啊').replace('h','诶去').replace('H','诶去').replace('a','诶').replace('A','诶').replace('o','欧').replace('O','欧').replace('i','爱').replace('I','爱')
    content = content.replace('m','爱慕').replace('M','爱慕').replace('n','恩').replace('N','恩').replace('g','寄').replace('G','寄').replace('d','地').replace('D','地').replace('e','艺').replace('E','艺').replace('c','吸').replace('C','吸').replace('呣', '嗯').replace('唷', '雨')
    content = content.strip().replace('A', '诶').replace('a', '诶').replace('B', '比').replace('b', '比').replace('C', '吸').replace('c', '吸').replace('D', '地').replace('d', '地').replace('E', '艺').replace('e', '艺').replace('F', '爱抚').replace('f', '爱抚').replace('G', '寄').replace('g', '寄').replace('H','诶去').replace('h','诶去').replace('I','爱').replace('i','爱').replace('J','杰').replace('j','杰').replace('K','开').replace('k','开').replace('L','了').replace('l','了').replace('M','爱慕').replace('m','爱慕').replace('N','恩').replace('n','恩').replace('O','欧').replace('o','欧').replace('P', '皮').replace('p','皮').replace('Q','去').replace('q','去').replace('R','啊').replace('r','啊').replace('S','爱思').replace('s','爱思').replace('T','提').replace('t','提').replace('U','优').replace('u','优').replace('V','微').replace('v','微').replace('W','维').replace('w','维').replace('X','克斯').replace('x','克斯').replace('Y','外').replace('y','外').replace('Z','贼').replace('z','贼')
    content = content.strip().replace('嗯', '蒽').replace('$', '').replace('哟', '优').replace('唷', '雨')
    content = content.replace('，', '').replace('。', '').replace('！', '').replace('？', '').replace(".", '').replace("！", '').replace("？", '').replace('!', '').replace('？', '').replace('，', '').replace('。', '').replace('；', '').replace(';', '').replace(':', '').replace('：', '')
    return content

def prepare_align(config):
    mfa_input = "./Spatial/drama_mfa"
    sampling_rate = config["preprocess_args"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocess_args"]["audio"]["max_wav_value"]
    target_loudness = config["preprocess_args"]["audio"]["target_loudness"]  # 可以调整为你希望的响度值，例如 -23.0 等
    txt_process = config['preprocess_args']['txt_process']
    os.makedirs(mfa_input, exist_ok=True)
    
    mfa_dict_fn = config["mfa"]["dict_fn"]
    mfa_list = []
    with open(mfa_dict_fn, encoding="utf-8") as f:
        for line in tqdm(f):
            text = line.split("\t")[0]
            mfa_list.append(text)
    
    mfa_dict = set()
    items = list()
    spk_info=dict()

    json_files = glob.glob(os.path.join(json_base_dir, '*.json'))
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data['items']:
            try:
                item_name = data['items'][item]['item_name']
                wav_file = data['items'][item]['wav_fn']
                text = clean_content(data['items'][item]["textgrid_txt"])
                txt, ph, ph2word, ph_gb_word_nosil = mfa_lab_prepare(text, txt_process)
                for w in ph_gb_word_nosil.split(" "):
                    if w not in mfa_list:
                        print('---------------------------------------------------')
                        print(f'{wav_file}里面的句子,')
                        print(text)
                        print(f'{txt}{ph_gb_word_nosil}#{w}#不存在于字典中')                    
                    mfa_dict.add(f"{w} {w.replace('_', ' ')}")
                data['items'][item]['word'] = ph_gb_word_nosil
                data['items'][item]['ph'] = ph
                
                with open(os.path.join(mfa_input, "{}.lab".format(item_name)),"w") as f:    
                    f.write(''.join(ph_gb_word_nosil))
                target_wav_fn = os.path.join(mfa_input, "{}.wav".format(item_name))
                normalize_loudness(wav_file, target_loudness=target_loudness, save_path=target_wav_fn, target_sample_rate=sampling_rate)
            except Exception as e:
                print(f"Error processing {item}: {str(e)}")
                continue
        
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
    config_path = './Spatial/drama.yaml'
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    prepare_align(config)