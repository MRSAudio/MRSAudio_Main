import os
import random
import shutil

# 配置参数
base_dir = "./audio-visual-seld-dcase2023/MRSAudio"
seed = 2023  # 随机种子，可修改
test_size = 800

# 创建需要的目录结构
sub_dirs = ['foa_dev', 'metadata_dev', 'video_dev']
for sub_dir in sub_dirs:
    for split in ['dev-test', 'dev-train']:
        os.makedirs(os.path.join(base_dir, sub_dir, split), exist_ok=True)

# 获取所有音频文件的基本名称
audio_files = [f for f in os.listdir(os.path.join(base_dir, 'foa_dev')) if f.endswith('.wav')]
basenames = [os.path.splitext(f)[0] for f in audio_files]
print(f"找到 {len(basenames)} 个音频文件")

# 设置随机种子并分割数据集
random.seed(seed)
test_basenames = set(random.sample(basenames, test_size))
train_basenames = set(basenames) - test_basenames

# 移动文件并生成路径列表的函数
def organize_dataset(split_basenames, split_name):
    audio_paths = []
    
    for bn in split_basenames:
        # 处理音频文件
        src = os.path.join(base_dir, 'foa_dev', f"{bn}.wav")
        dest_dir = os.path.join(base_dir, 'foa_dev', split_name)
        shutil.move(src, os.path.join(dest_dir, f"{bn}.wav"))
        audio_paths.append(os.path.join('foa_dev', split_name, f"{bn}.wav"))

        # 处理metadata
        src_meta = os.path.join(base_dir, 'metadata_dev', f"{bn}.csv")
        dest_meta_dir = os.path.join(base_dir, 'metadata_dev', split_name)
        shutil.move(src_meta, os.path.join(dest_meta_dir, f"{bn}.csv"))

        # 处理视频文件
        src_video = os.path.join(base_dir, 'video_dev', f"{bn}.mp4")
        dest_video_dir = os.path.join(base_dir, 'video_dev', split_name)
        shutil.move(src_video, os.path.join(dest_video_dir, f"{bn}.mp4"))
    
    return audio_paths

# 处理测试集和训练集
test_paths = organize_dataset(test_basenames, 'dev-test')
train_paths = organize_dataset(train_basenames, 'dev-train')

# 写入list_dataset
list_dataset_dir = os.path.join(base_dir, 'list_dataset')
with open(os.path.join(list_dataset_dir, 'dcase2023t3_foa_devtest.txt'), 'w') as f:
    f.write('\n'.join(sorted(test_paths)))

with open(os.path.join(list_dataset_dir, 'dcase2023t3_foa_devtrain.txt'), 'w') as f:
    f.write('\n'.join(sorted(train_paths)))

print(f"数据集分割完成！测试集{test_size}条，训练集{len(basenames)-test_size}条")
print(f"文件列表已生成在: {list_dataset_dir}")