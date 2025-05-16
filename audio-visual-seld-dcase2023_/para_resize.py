import os
import glob
from tqdm import tqdm
import moviepy.editor
from moviepy.video.io.VideoFileClip import VideoFileClip
from multiprocessing import Pool, cpu_count
import traceback

def process_single_video(args):
    """处理单个视频的worker函数"""
    path_input, path_output, new_width = args
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(path_output), exist_ok=True)
        
        # 处理视频
        with VideoFileClip(path_input) as clip:
            resized_clip = clip.resize(width=new_width)
            resized_clip.write_videofile(path_output, verbose=False)
        return path_input, None
    except Exception as e:
        error_msg = f"{path_input} 处理失败: {str(e)}\n{traceback.format_exc()}"
        return path_input, error_msg

def main_parallel():
    # 配置参数
    path_dir = "./MRSAudio/video_dev"
    old_dir_name = "/video_dev/"
    new_dir_name = "/video_360x180_dev/"
    new_width = 360
    workers = 32  # 可根据CPU核心数调整

    # 获取文件列表
    path_files = glob.glob(f"{path_dir}/**/*.mp4", recursive=True)
    path_files = sorted(path_files)
    print(f"发现 {len(path_files)} 个待处理视频")

    # 准备任务参数
    task_args = []
    for path_input in path_files:
        path_output = path_input.replace(old_dir_name, new_dir_name)
        task_args.append((path_input, path_output, new_width))

    # 创建进程池
    error_log = []
    with Pool(processes=min(workers, len(path_files))) as pool:
        # 使用tqdm显示进度
        with tqdm(total=len(task_args), desc="视频处理") as pbar:
            for path, error in pool.imap_unordered(process_single_video, task_args):
                if error:
                    error_log.append(error)
                    pbar.write(f"错误: {error[:100]}...")  # 显示错误摘要
                pbar.update(1)
                pbar.set_postfix({"错误数": len(error_log)})

    # 输出统计信息
    print(f"\n处理完成: 成功 {len(path_files)-len(error_log)}/{len(path_files)}")
    if error_log:
        print(f"前5个错误详情:")
        for err in error_log[:5]:
            print("-"*50)
            print(err)

if __name__ == "__main__":
    main_parallel()