import os
import glob
import json
import wave
import webrtcvad
import numpy as np
import soundfile as sf
from scipy import signal
import librosa
import tqdm
import random


def get_info_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    car_type = data['type']
    event = data['event']
    loc_list = data['loc_list']
    metric = data['final_metric']
    return car_type, event, loc_list, metric

def gen_meta(audio_dir, out_meta_dir):
    meta_list = []
    audio_files = sorted(glob.glob(os.path.join(audio_dir, '**', '*.wav'), recursive=True))
    for audio_file in tqdm.tqdm(audio_files, desc=f'Processing {audio_dir}'):
        file_name = os.path.basename(audio_file)
        # json_file = audio_file.replace('.wav', '.json')
        # try:
        #     car_type, event, loc_list, metric = get_info_from_json(json_file)
        # except Exception as e:
        #     print(f'Error in {json_file}, {e}')
        #     assert False
        # /home2/zhangyu/Spatial-AST/dataset/vehicleaudio/segments/seg_data/室内/室内1/室内1_1/室内1_1_0.wav
        folder = audio_file.split('/')[-2]
        item_name = audio_file.split('/')[-1].replace('.wav','')
        # metric['overall_score'] *= 20
        # metric['quality_score'] *= 20
        # metric['spatial_score'] *= 20
        # metric['localization_score'] *= 20
        meta = {
            "id": item_name,
            "folder": folder,
            "original_audio": file_name,
            # "car_type": car_type,
            # "event": event,
            # "loc_list": loc_list,
            # "metric": metric
        }
        meta_list.append(meta)
    
    # generate train.json and eval.json
    # train_meta_list = []
    # eval_meta_list = []
    
    # random.seed(0)
    # random.shuffle(meta_list)
    # for i, meta in enumerate(meta_list):
    #     if i % 10:
    #         train_meta_list.append(meta)
    #     else:
    #         eval_meta_list.append(meta)
    
    # with open(os.path.join(out_meta_dir, 'train_class.json'), 'w') as f:
    #     json.dump(train_meta_list, f, indent=4, ensure_ascii=False)
    # with open(os.path.join(out_meta_dir, 'eval_class.json'), 'w') as f:
    #     json.dump(eval_meta_list, f, indent=4, ensure_ascii=False)
    with open(os.path.join(out_meta_dir, 'FireRed_meta.json'), 'w') as f:
        json.dump(meta_list, f, indent=4, ensure_ascii=False)
        
if __name__ == "__main__":
    audio_dir = '/home/panchanghao/2025-ICML/evaluation/test-data/test_FireRedTTS'
    out_meta_dir = '/home/panchanghao/2025-ICML/evaluation/metadata'
    gen_meta(audio_dir, out_meta_dir)
        
        
        