import os
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def get_position_and_orientation(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
        position = data["ear_position"]
        orientation = data["ear_orientation"]
        angle = np.arctan2(orientation[1], orientation[0]) - np.pi / 2
    return position, angle

def process(npy_path, position, orientation):
    data = np.load(npy_path, allow_pickle=True)
    
    # 坐标系变换
    data[:, :3] -= position
    x_new = data[:, 0] * np.cos(orientation) + data[:, 1] * np.sin(orientation)
    y_new = - data[:, 0] * np.sin(orientation) + data[:, 1] * np.cos(orientation)
    data[:, 0], data[:, 1] = x_new, y_new
    
    # 时间处理
    data[:, 3] -= data[0, 3]
    data = data[data[:, 3].argsort()]
    
    # 插值处理
    original_times = data[:, 3]
    x_vals, y_vals, z_vals = data[:, 0], data[:, 1], data[:, 2]
    max_time = original_times[-1]
    new_times = np.arange(0, max_time + 1e-9, 50)
    
    new_data = np.column_stack((
        np.interp(new_times, original_times, x_vals),
        np.interp(new_times, original_times, y_vals),
        np.interp(new_times, original_times, z_vals),
        new_times,
        np.full_like(new_times, data[0, 4])
    ))
    
    np.save(npy_path.replace(".npy", "_final.npy"), new_data)
    return True

def process_task(npy_path):
    try:
        dir_path = os.path.dirname(npy_path)
        # 查找metadata.json文件
        json_files = [f for f in os.listdir(dir_path) if f.endswith("metadata.json")]
        if not json_files:
            raise FileNotFoundError(f"Metadata not found for {npy_path}")
        
        json_path = os.path.join(dir_path, json_files[0])
        position, orientation = get_position_and_orientation(json_path)
        process(npy_path, position, orientation)
        return (npy_path, True)
    except Exception as e:
        print(f"\nError processing {npy_path}: {str(e)}")
        return (npy_path, False)

def main():
    # base_dir = "./Spatial/play"
    base_dir = "./Spatial/MRSDrama2"
    
    # 收集所有需要处理的文件路径
    task_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_cut.npy"):
                task_paths.append(os.path.join(root, file))
    
    # 创建进程池（根据CPU核心数自动设置）
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # 提交任务并使用tqdm显示进度
        futures = [executor.submit(process_task, path) for path in task_paths]
        
        # 进度条配置
        progress_bar = tqdm(
            total=len(task_paths),
            desc="Processing files",
            unit="file",
            dynamic_ncols=True
        )
        
        # 结果处理
        success_count = 0
        for future in futures:
            try:
                path, status = future.result()
                if status:
                    success_count += 1
            except Exception as e:
                print(f"\nUnexpected error: {str(e)}")
            finally:
                progress_bar.update(1)
        
        progress_bar.close()
        print(f"\nProcessing complete. Success: {success_count}/{len(task_paths)}")

if __name__ == "__main__":
    main()