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


sample_rate = 48000
hop_size = 256
hop_time = hop_size / float(sample_rate)  

def process_data_3d(file_path):
    
    data = np.load(file_path)
    n = data.shape[0]
    result = np.zeros((n, 12))
    
    
    ref_left = np.array([-0.11, 0.0, 0.0])
    ref_right = np.array([0.11, 0.0, 0.0])
    
    for i in range(n):
        
        current_pos = data[i, :3]
        x, y, z = current_pos
        
        
        start = max(0, i-10)
        end = min(n-1, i+10)
        delta_pos = data[end, :3] - data[start, :3]
        delta_t_ms = data[end, 3] - data[start, 3]
        
        if delta_t_ms <= 0:
            velocity = np.zeros(3)
        else:
            velocity = delta_pos / (delta_t_ms / 1000.0)  
        
        speed = np.linalg.norm(velocity)
        
        
        if speed < 1e-6:
            yaw = pitch = roll = 0.0
        else:
            direction = velocity / speed
            yaw = np.arctan2(direction[1], direction[0]) * 180 / math.pi  
            pitch = np.arcsin(direction[2]) * 180 / math.pi                
            roll = 0.0  
            
        quat = euler_to_quaternion(yaw, pitch, roll)  
        
        
        def calc_components(pos, ref_point):
            """calculate radial and normal components of velocity"""
            
            vec_to_ref = pos - ref_point
            dist = np.linalg.norm(vec_to_ref)
            
            if dist < 1e-6:
                return 0.0, 0.0  
            
            
            radial_unit = vec_to_ref / dist
            
            
            v_radial = np.dot(velocity, radial_unit)
            
            
            v_normal = sqrt(max(speed**2 - v_radial**2, 0.0))
            
            return v_radial, v_normal
        
        
        v_r_left, v_n_left = calc_components(current_pos, ref_left)
        v_r_right, v_n_right = calc_components(current_pos, ref_right)
        
        
        result[i] = [
            x, y, z,                   
            *quat,                     
            speed,                     
            v_r_left, v_n_left,        
            v_r_right, v_n_right       
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
    """function to process a single file, suitable for parallel execution"""
    key, value = args
    try:
        npy_file = value["pos_fn"].replace('drama_splitspker', 'drama_pos_recut')
        output_file = npy_file.replace('drama_splitspker', 'drama_pos_recut').replace(".npy", "_v.npy")
        if os.path.exists(output_file):
            return key, {"status": "success", "path": output_file}
        if "DYNAMIC" in value["spatial_prompt"] or "Dynamic" in value["spatial_prompt"]:
            processed_data = process_data_3d(npy_file)
            
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
                print(f"unknown scene {scene}: {key}: {value['spatial_prompt']}")
                orient = euler_to_quaternion(0, 0, 0)
            
            processed_data = process_static_scene(npy_file, orient)
        else:
            print(f"unknown scene {key}: {value['spatial_prompt']}")
        np.save(output_file, np.hstack((processed_data[:, :7], processed_data[:, [8]], processed_data[:, [10]])))
        
        return key, {"status": "success", "path": output_file}
    
    except Exception as e:
        error_msg = f"{traceback.format_exc()}"
        return key, {"status": "error", "error": error_msg}

def main_parallel(json_path):
    
    with open(json_path, "r") as f:
        metadata = json.load(f)
    
    
    task_args = [(k, v) for k, v in metadata.items()]
    
    
    workers = 32  
    results = {}
    
    with Pool(processes=workers) as pool:
        
        with tqdm(total=len(task_args), ) as pbar:
            for key, result in pool.imap_unordered(process_single_file, task_args):
                results[key] = result
                
                
                pbar.update(1)
                pbar.set_postfix({
                    "success": sum(1 for v in results.values() if v["status"] == "success"),
                    "skipped": sum(1 for v in results.values() if v["status"] == "skipped"),
                    "error": sum(1 for v in results.values() if v["status"] == "error")
                })
                
                
                if result["status"] == "error":
                    pbar.write(f"error: {key}:\n{result['error']}")

    
    success = sum(1 for v in results.values() if v["status"] == "success")
    skipped = sum(1 for v in results.values() if v["status"] == "skipped")
    errors = sum(1 for v in results.values() if v["status"] == "error")
    
    print("finished")
    print(f"success: {success}, skipped: {skipped}, error: {errors}")

if __name__ == "__main__":
    drama_json = './Spatial/_MRSDrama.json'
    main_parallel(drama3_json)