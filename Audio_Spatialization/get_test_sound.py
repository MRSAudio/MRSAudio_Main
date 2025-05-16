# 添加在原有代码末尾的分割数据集部分
import shutil
import os
from tqdm import tqdm
import json
import numpy as np
import glob

output_root = 'data/bingrad_audio'
test_files = glob.glob('data/bingrad_audio/testset/*')
test_items = []
for test_file in test_files:
    item_name = test_file.split('/')[-1]
    test_items.append(
        {
            'item_name': item_name,
            'wav_file': test_file
        }
    )

train_files = glob.glob('data/bingrad_audio/trainset/*')
train_items = []
for train_file in train_files:
    item_name = train_file.split('/')[-1]
    train_items.append(
        {
            'item_name': item_name,
            'wav_file': train_file
        }
    )

# 生成对应的JSON文件
testset_json_path = os.path.join(output_root, 'testset.json')
trainset_json_path = os.path.join(output_root, 'trainset.json')

with open(testset_json_path, 'w') as f:
    json.dump(test_items, f, indent=4)

with open(trainset_json_path, 'w') as f:
    json.dump(train_items, f, indent=4)

print("Dataset splitting completed successfully!")