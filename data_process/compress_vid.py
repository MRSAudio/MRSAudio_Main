import os
import subprocess
import argparse
from pathlib import Path
import logging
from tqdm import tqdm

def setup_logger():
    """配置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("video_compress.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

def compress_video(input_path, output_path, gpu_device=0, preset="p7", cq=23):
    """使用GPU压缩单个视频文件"""
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists, skipping.")
        return True
    try:
        cmd = [
            "ffmpeg", "-y", 
            "-i", str(input_path),
            "-c:v", "h264_nvenc",
            "-preset", "8",          # 使用数字预设
            "-cq", str(cq),           # 注意这里改为 -cq 而不是之前的 -crf
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            "-filter:v", "fps=fps=24,format=yuv420p,scale=1920:1080",
            "-vsync", "vfr",
            "-colorspace", "bt709",   # 添加色彩空间转换
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            str(output_path)
        ]

        print("CUDA_VISIBLE_DEVICES=0 " + " ".join(cmd))
        result = subprocess.run(
            "CUDA_VISIBLE_DEVICES=0 " + " ".join(cmd),
            # cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Success: {input_path} -> {output_path}")
            return True
        return False

    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing {input_path}: {e.output}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error with {input_path}: {str(e)}")
        return False

def process_directory(root_dir, **kwargs):
    """处理目录及其子目录中的所有MP4文件（单线程）"""
    processed = 0
    failed = 0
    
    # 收集所有需要处理的文件
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(".mp4") and not f.endswith("_compressed.mp4") and not f.endswith("_compressed_cut.mp4") and "animation" not in f:
                input_path = Path(dirpath) / f
                output_path = input_path.with_name(f"{input_path.stem}_compressed.mp4")
                
                if not output_path.exists():
                    files.append((input_path, output_path))

    # 单线程顺序处理
    with tqdm(total=len(files), desc="Processing videos") as pbar:
        for input_path, output_path in files:
            if not os.path.exists(output_path):
                success = compress_video(input_path, output_path, **kwargs)
                if success:
                    processed += 1
                else:
                    failed += 1
            else:
                logger.warning(f"Output file {output_path} already exists, skipping.")
                processed += 1
            pbar.update(1)

    logger.info(f"\nProcessing complete! Success: {processed}, Failed: {failed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="批量视频压缩工具 (NVIDIA GPU加速)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input", 
        required=True,
        help="输入目录路径"
    )
    parser.add_argument(
        "-g", "--gpu", 
        type=int, 
        default=0,
        help="使用的GPU设备ID（已移除并行，此参数保留但不生效）"
    )
    parser.add_argument(
        "-p", "--preset", 
        choices=["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        default="p7",
        help="NVENC编码预设 (p1最快-p7最慢)"
    )
    parser.add_argument(
        "-q", "--cq", 
        type=int,
        default=23,
        help="质量参数 (0-51，值越小质量越高)"
    )

    args = parser.parse_args()

    # 参数验证
    if not Path(args.input).is_dir():
        logger.error("输入路径不存在或不是目录！")
        exit(1)

    # 开始处理（强制单线程）
    process_directory(
        root_dir=args.input,
        gpu_device=args.gpu,
        preset=args.preset,
        cq=args.cq
    )
