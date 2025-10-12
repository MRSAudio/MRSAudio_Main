import os
import glob
import soundfile as sf
import pyloudnorm as pyln
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import traceback

def process_single_file(args):
    file_path, target_lufs, overwrite = args
    
    
    if overwrite:
        out_path = file_path
    else:
        dir_name = os.path.dirname(file_path)
        new_dir = os.path.join(dir_name, "spatial_normalized")
        os.makedirs(new_dir, exist_ok=True)
        out_path = os.path.join(new_dir, os.path.basename(file_path))
    
    if os.path.exists(out_path):
        return file_path, None
    
    try:
        
        data, rate = sf.read(file_path)
        
        
        meter = pyln.Meter(rate)
        
        
        loudness = meter.integrated_loudness(data)
        
        
        if loudness == float("-inf"):
            norm_data = data
            print(f"skipped: {file_path}")
        else:
            
            norm_data = pyln.normalize.loudness(data, loudness, target_lufs)
            
            
            peak = np.max(np.abs(norm_data))
            if peak > 1.0:
                norm_data = norm_data / peak * 0.9999

        
        info = sf.info(file_path)
        sf.write(out_path, norm_data, rate, subtype=info.subtype)
        
        return file_path, None
    
    except Exception as e:
        error_msg = f"{file_path} fail: {str(e)}\n{traceback.format_exc()}"
        return file_path, error_msg

def normalize_to_lufs_parallel(input_dir, target_lufs=-23.0, overwrite=True, workers=32):
    """
    normalize all *_缩混_cut.wav files in input_dir subdirectories to target_lufs
    and save to spatial_normalized subdirectory (or overwrite if specified)
    """
    
    pattern = os.path.join(input_dir, "*", "*_缩混_cut.wav")
    audio_files = sorted(glob.glob(pattern))
    print(f"found {len(audio_files)} files to process.")

    if not audio_files:
        print("no audio files found, exiting.")
        return

    
    task_args = [(f, target_lufs, overwrite) for f in audio_files]

    
    error_log = []
    with Pool(processes=min(workers, len(audio_files))) as pool:
        
        with tqdm(total=len(audio_files)) as pbar:
            for file_path, error in pool.imap_unordered(process_single_file, task_args):
                if error:
                    error_log.append(error)
                    pbar.write(error)  
                pbar.update(1)
                pbar.set_postfix({"error": len(error_log)})

    
    print(f"\nfinished processing {len(audio_files)} files.")
    if error_log:
        print(f"error files: ({len(error_log)} 个):")
        for err in error_log[:5]:  
            print("-"*50)
            print(err)

if __name__ == "__main__":
    input_dir = "./Spatial/MRSDrama2"
    normalize_to_lufs_parallel(
        input_dir=input_dir,
        target_lufs=-23.0,
        overwrite=False,
        workers=32
    )