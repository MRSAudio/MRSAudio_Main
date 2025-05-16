# 添加在原有代码末尾的分割数据集部分
import shutil
import os
from tqdm import tqdm
import json
import numpy as np

output_root = 'data/bingrad_drama'
meta_fn = 'data/_drama_metadata_all_with_prompt.json'

# 创建testset和trainset目录
testset_dir = os.path.join(output_root, 'testset')
trainset_dir = os.path.join(output_root, 'trainset')
os.makedirs(testset_dir, exist_ok=True)
os.makedirs(trainset_dir, exist_ok=True)
items_dict = json.load(open(meta_fn, 'r'))
items_list = []
for key, value in items_dict.items():
    items_list.append(value)
# 获取所有item的总数
total_items = len(items_list)

# 生成等间距的400个测试索引
indices = np.linspace(0, total_items - 1, num=400, dtype=int)
test_items = [items_list[i] for i in indices]
train_items = [item for idx, item in enumerate(items_list) if idx not in indices]

# 移动测试集item到testset目录
for item in tqdm(test_items, desc='Moving test items'):
    item_name = item['item_name']
    src = os.path.join(output_root, item_name)
    dst = os.path.join(testset_dir, item_name)
    if os.path.exists(src):
        shutil.move(src, dst)
    else:
        print(f"Warning: {src} does not exist. Skipping...")

# 移动训练集item到trainset目录
for item in tqdm(train_items, desc='Moving train items'):
    item_name = item['item_name']
    src = os.path.join(output_root, item_name)
    dst = os.path.join(trainset_dir, item_name)
    if os.path.exists(src):
        shutil.move(src, dst)
    else:
        print(f"Warning: {src} does not exist. Skipping...")

# 生成对应的JSON文件
testset_json_path = os.path.join(output_root, 'testset.json')
trainset_json_path = os.path.join(output_root, 'trainset.json')

with open(testset_json_path, 'w') as f:
    json.dump(test_items, f, indent=4)

with open(trainset_json_path, 'w') as f:
    json.dump(train_items, f, indent=4)

print("Dataset splitting completed successfully!")