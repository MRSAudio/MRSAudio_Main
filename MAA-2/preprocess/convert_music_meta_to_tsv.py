import os
import glob
import json
import argparse
import tqdm

INSTRUMENT_DICT = {
    '电子琴': 'electronic keyboard',
    '小提琴': 'violin',
    '中提琴': 'viola',
    '大提琴': 'cello',
    '低音提琴': 'double bass',
    '小号': 'trumpet',
    '长号': 'trombone',
    '上低音号': 'euphonium',
    '长笛': 'flute',
    '单簧管': 'clarinet',
    '双簧管': 'oboe',
    '萨克斯': 'saxophone',
    '贝斯': 'bass-guitar',
    '吉他': 'guitar',
    '电吉他': 'electric guitar',
    '二胡': 'erhu',
    '琵琶': 'pipa',
    '竹笛': 'bamboo flute',
    '巴乌': 'bawu',
    '箫': 'xiao',
    '中音笙': 'alto sheng',
    '马头琴': 'morin khuur',
    '古筝': 'guzheng',
    'violin': 'violin'
}


def convert_json_to_tsv(input_meta_path, audio_dir, output_tsv):
    """
    Convert JSON metadata and audio directory into a TSV caption file.
    """
    # 加载 JSON 元数据
    with open(input_meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    # 用 item_name 构建索引，加速检索
    meta_dict = {item['item_name']: item for item in meta_data}

    # 获取所有音频文件
    audio_files = glob.glob(os.path.join(audio_dir, 'testset', '*.wav'))

    new_meta = []

    for audio_file in tqdm.tqdm(audio_files, desc="Processing audio files"):
        audio_item_name = os.path.splitext(os.path.basename(audio_file))[0]

        if audio_item_name not in meta_dict:
            continue

        item = meta_dict[audio_item_name]
        instrument_en = INSTRUMENT_DICT.get(item['instrument'], item['instrument'])

        caption = (
            f"A player playing {instrument_en}, the midi pitch sequence is {item['note']}, "
            f"the pitch duration list is {item['note_durs']}, and the velocity of the note is {item['velocity']}."
        )

        new_meta.append({
            'name': item['item_name'],
            'dataset': 'mrsmuic',
            'audio_path': audio_file,
            'caption': caption
        })

    # 写入 TSV 文件
    with open(output_tsv, 'w', encoding='utf-8') as f:
        f.write('name\tdataset\taudio_path\tcaption\n')
        for item in tqdm.tqdm(new_meta, desc="Writing to TSV"):
            f.write(f"{item['name']}\t{item['dataset']}\t{item['audio_path']}\t{item['caption']}\n")

    print('✅ TSV file created successfully!')


def main():
    parser = argparse.ArgumentParser(description="Convert music metadata from JSON to caption TSV format.")
    parser.add_argument('--input_meta', type=str, required=True, help='Path to the input metadata JSON file')
    parser.add_argument('--audio_dir', type=str, required=True, help='Path to the audio directory (contains subfolders with .wav files)')
    parser.add_argument('--output_tsv', type=str, required=True, help='Path to output TSV file')
    
    args = parser.parse_args()
    
    convert_json_to_tsv(args.input_meta, args.audio_dir, args.output_tsv)


if __name__ == '__main__':
    main()
