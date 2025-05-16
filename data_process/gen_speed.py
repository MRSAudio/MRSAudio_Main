import numpy as np
from math import sqrt
from tqdm import tqdm
import os
import sys
import json
import math
import traceback
from euler_to_quaternion import euler_to_quaternion
from multiprocessing import Pool
import subprocess

# 全局配置
sample_rate = 48000
hop_size = 256
hop_time = hop_size / float(sample_rate)  # 每帧对应时间（秒）

def process_data_3d(file_path):
    # 加载数据
    data = np.load(file_path)
    n = data.shape[0]
    result = np.zeros((n, 12))
    
    # 参考点坐标
    ref_left = np.array([-0.11, 0.0, 0.0])
    ref_right = np.array([0.11, 0.0, 0.0])
    
    for i in range(n):
        # 原始坐标和时间
        current_pos = data[i, :3]
        x, y, z = current_pos
        
        # 速度计算 ------------------------------------------------------
        start = max(0, i-10)
        end = min(n-1, i+10)
        delta_pos = data[end, :3] - data[start, :3]
        delta_t_ms = data[end, 3] - data[start, 3]
        
        if delta_t_ms <= 0:
            velocity = np.zeros(3)
        else:
            velocity = delta_pos / (delta_t_ms / 1000.0)  # m/s
        
        speed = np.linalg.norm(velocity)
        
        # 欧拉角转换 ----------------------------------------------------
        if speed < 1e-6:
            yaw = pitch = roll = 0.0
        else:
            direction = velocity / speed
            yaw = np.arctan2(direction[1], direction[0]) * 180 / math.pi  # XY平面投影方向
            pitch = np.arcsin(direction[2]) * 180 / math.pi                # Z方向夹角
            roll = 0.0  # 假设无滚转
            
        quat = euler_to_quaternion(yaw, pitch, roll)  # 四元数转换
        
        # 速度分解计算 ---------------------------------------------------
        def calc_components(pos, ref_point):
            """计算相对于参考点的径向/法向速度分量"""
            # 生成当前点到参考点的向量
            vec_to_ref = pos - ref_point
            dist = np.linalg.norm(vec_to_ref)
            
            if dist < 1e-6:
                return 0.0, 0.0  # 重合时无法计算
            
            # 径向单位向量
            radial_unit = vec_to_ref / dist
            
            # 径向速度分量
            v_radial = np.dot(velocity, radial_unit)
            
            # 法向速度大小（勾股定理）
            v_normal = sqrt(max(speed**2 - v_radial**2, 0.0))
            
            return v_radial, v_normal
        
        # 左右参考点分量计算
        v_r_left, v_n_left = calc_components(current_pos, ref_left)
        v_r_right, v_n_right = calc_components(current_pos, ref_right)
        
        # 结果组装
        result[i] = [
            x, y, z,                   # 原始坐标
            *quat,                     # 四元数
            speed,                     # 总速度
            v_r_left, v_n_left,        # 左参考点分量
            v_r_right, v_n_right       # 右参考点分量
        ]
    
    return result

def process_static_scene(file_path, orient):
    data = np.load(file_path)
    if len(data.shape)==2:
        processed_data = np.zeros((data.shape[0], 12))
        processed_data[:, :3] = data[:, :3]
        processed_data[:, 3:7] = np.tile(orient, (data.shape[0], 1))
    else:
        processed_data = np.zeros((1, 12))
        processed_data[0, :3] = data[:3]
        processed_data[0, 3:7] = orient
    return processed_data

def process_single_file(args):
    """处理单个文件的worker函数"""
    key, value = args
    try:
        npy_file = value["pos_fn"].replace('drama_splitspker', 'drama_pos_recut')
        output_file = npy_file.replace('drama_splitspker', 'drama_pos_recut').replace(".npy", "_v.npy")
        if os.path.exists(output_file):
            return key, {"status": "success", "path": output_file}
        if "DYNAMIC" in value["spatial_prompt"] or "Dynamic" in value["spatial_prompt"]:
            processed_data = process_data_3d(npy_file)
            # np.save(output_file, processed_data)
        elif "STATIC" in value["spatial_prompt"] or "Static" in value["spatial_prompt"]:
            scene = value["scene"]
            if scene == 5:
                orient = euler_to_quaternion(-180, 0, 0)
            elif scene == 6:
                data = np.load(npy_file)
                if len(data.shape)==2:
                    x_mean = np.mean(data[:, 0])
                    y_mean = np.mean(data[:, 1])
                else:
                    x_mean = data[0]
                    y_mean = data[1]
                orient = euler_to_quaternion(math.atan2(y_mean, x_mean) * 180 / math.pi, 0, 0)
            elif scene == 7:
                orient = euler_to_quaternion(-90, 0, 0)
            else:
                print(f"未知场景{scene}: {key}: {value['spatial_prompt']}")
                orient = euler_to_quaternion(0, 0, 0)
            # 处理静态场景
            processed_data = process_static_scene(npy_file, orient)
        else:
            print(f"未知场景: {key}: {value['spatial_prompt']}")
        np.save(output_file, np.hstack((processed_data[:, :7], processed_data[:, [8]], processed_data[:, [10]])))
        
        return key, {"status": "success", "path": output_file}
    
    except Exception as e:
        error_msg = f"{traceback.format_exc()}"
        return key, {"status": "error", "error": error_msg}

def main_parallel(json_path):
    # 加载元数据
    with open(json_path, "r") as f:
        metadata = json.load(f)
    
    # 准备任务参数
    task_args = [(k, v) for k, v in metadata.items()]
    
    # 创建进程池
    workers = 32  # 根据实际情况调整
    results = {}
    
    with Pool(processes=workers) as pool:
        # 使用tqdm显示进度
        with tqdm(total=len(task_args), desc="处理3D数据") as pbar:
            for key, result in pool.imap_unordered(process_single_file, task_args):
                results[key] = result
                
                # 更新进度和显示状态
                pbar.update(1)
                pbar.set_postfix({
                    "成功": sum(1 for v in results.values() if v["status"] == "success"),
                    "跳过": sum(1 for v in results.values() if v["status"] == "skipped"),
                    "错误": sum(1 for v in results.values() if v["status"] == "error")
                })
                
                # 打印错误信息
                if result["status"] == "error":
                    pbar.write(f"错误处理 {key}:\n{result['error']}")

    # 输出最终统计
    success = sum(1 for v in results.values() if v["status"] == "success")
    skipped = sum(1 for v in results.values() if v["status"] == "skipped")
    errors = sum(1 for v in results.values() if v["status"] == "error")
    
    print("\n处理完成:")
    print(f"总文件数: {len(task_args)}")
    print(f"成功处理: {success}")
    print(f"跳过已存在: {skipped}")
    print(f"处理失败: {errors}")

if __name__ == "__main__":
    drama_json = './Spatial/_MRSDrama.json'
    main_parallel(drama3_json)