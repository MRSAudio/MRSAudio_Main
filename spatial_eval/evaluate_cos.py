import os
import glob
import numpy as np
import tqdm

def matrix_cosine_similarity(matrix_a, matrix_b, epsilon=1e-8):
    """
    计算两个矩阵的加权余弦相似度（考虑IPD/ILD的特殊性）
    
    参数：
    matrix_a : ndarray - 参考矩阵（真实值）
    matrix_b : ndarray - 对比矩阵（预测值）
    epsilon : float - 防止除零的小量
    
    返回：
    float - 范围在[0,1]的相似度得分（1表示完全相同）
    """
    # 矩阵对齐
    min_rows = min(matrix_a.shape[0], matrix_b.shape[0])
    min_cols = min(matrix_a.shape[1], matrix_b.shape[1])
    # assert min_cols == 128
    a = matrix_a[:min_rows, :min_cols].flatten()
    b = matrix_b[:min_rows, :min_cols].flatten()
    
    # 相位周期修正（针对IPD）
    if np.any(a > np.pi):  # 检测是否为IPD矩阵（相位差范围[-π, π]）
        a = np.angle(np.exp(1j*a))  # 规范到[-π, π]
        b = np.angle(np.exp(1j*b))
        
    
    
    # 能量归一化（针对ILD）
    a_norm = a / (np.linalg.norm(a) + epsilon)
    b_norm = b / (np.linalg.norm(b) + epsilon)
    
    # print(a_norm.shape, b_norm.shape)
    
    # 带符号的余弦相似度
    raw_score = np.dot(a_norm, b_norm)
    # print(raw_score)
    
    return raw_score


if __name__ == "__main__":
    audio_dir = "./outputs/infer/npy"
    dis_npy_files = sorted(glob.glob(os.path.join(audio_dir, "*_dis.npy")))
    gt_dis_npy_files = dis_npy_files
    
    dis_scores = []
    azi_scores = []
    for gt_dis_npy_file in tqdm.tqdm(gt_dis_npy_files,desc='evaluate distance cos'):
        # print(gt_dis_npy_file)
        pred_dis_npy_file = gt_dis_npy_file.replace('outputs/infer', 'outputs/gt')
        assert os.path.exists(pred_dis_npy_file)
        gt_dis_tokens = np.load(gt_dis_npy_file, allow_pickle=True)
        pred_dis_tokens = np.load(pred_dis_npy_file, allow_pickle=True)        
        dis_score = matrix_cosine_similarity(gt_dis_tokens, pred_dis_tokens)
        dis_scores.append(dis_score)
    
    azi_npy_files = sorted(glob.glob(os.path.join(audio_dir, "*_azi.npy")))
    gt_azi_npy_files = azi_npy_files
    
    for gt_azi_npy_file in tqdm.tqdm(gt_azi_npy_files,desc='evaluate azimuth cos'):
        pred_azi_npy_file = gt_azi_npy_file.replace('outputs/infer', 'outputs/gt')
        gt_azi_tokens = np.load(gt_azi_npy_file, allow_pickle=True)
        pred_azi_tokens = np.load(pred_azi_npy_file, allow_pickle=True)
        
        azi_score = matrix_cosine_similarity(gt_azi_tokens, pred_azi_tokens)
        azi_scores.append(azi_score)

    dis_cos = np.mean(dis_scores)
    azi_cos = np.mean(azi_scores)

    print(f"dis Cosine Similarity: {dis_cos:.4f} (-1~1 scale)")
    print(f"azi Cosine Similarity: {azi_cos:.4f} (-1~1 scale)")