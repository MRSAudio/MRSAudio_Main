import os
import glob
import numpy as np
import tqdm

# 读取npy文件中的IPD和ILD

def read_IPD_ILD_from_npy(npy_path):
    """
    从.npy文件中读取IPD和ILD矩阵
    
    参数:
    npy_path: str - 存储了字典的.npy文件路径，字典包含'IPD'和'ILD'两个键
    
    返回:
    (IPD_mel, ILD_mel) - 包含两个二维numpy数组的元组
    """
    # 加载数据（允许pickle）
    data = np.load(npy_path, allow_pickle=True)
    # 转换为字典并提取数据
    data_dict = data.item()
    IPD_mel = data_dict['IPD']
    ILD_mel = data_dict['ILD']
    return IPD_mel, ILD_mel

def calculate_mae(matrix_a, matrix_b):
    """
    计算两个矩阵之间的平均绝对误差（MAE）
    自动截取较大矩阵的左上角部分与较小矩阵对齐
    
    参数：
    matrix_a : array_like
        第一个输入矩阵（二维数组）
    matrix_b : array_like
        第二个输入矩阵（二维数组）
        
    返回：
    float
        对齐后矩阵的MAE值
    """
    # 转换为NumPy数组
    a = np.asarray(matrix_a)
    b = np.asarray(matrix_b)
    
    # 获取最小维度
    min_rows = min(a.shape[0], b.shape[0])
    min_cols = min(a.shape[1], b.shape[1])
    
    # 截取矩阵的左上角部分
    a_trunc = a[:min_rows, :min_cols]
    b_trunc = b[:min_rows, :min_cols]
    
    # assert a_trunc.shape[0] == 128
    
    return np.mean(np.abs(a_trunc - b_trunc))

def calculate_mae_time(matrix_a, matrix_b):
    mae_value = calculate_mae(matrix_a, matrix_b)
    return mae_value / min(np.asarray(matrix_a).shape[0], np.asarray(matrix_b).shape[0])

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


def frequency_band_similarity(matrix_a, matrix_b, sr=16000, n_fft=512):
    """
    分频带计算加权余弦相似度
    """
    # 对齐矩阵
    min_rows = min(matrix_a.shape[0], matrix_b.shape[0])
    min_cols = min(matrix_a.shape[1], matrix_b.shape[1])
    a = matrix_a[:min_rows, :min_cols]
    b = matrix_b[:min_rows, :min_cols]
    
    # 频带划分（根据听觉临界频带）
    freq_bands = [
        (0, 500),    # 低频带
        (500, 2000), # 中频带
        (2000, 20000) # 高频带
    ]
    # 频率轴计算
    freqs = np.linspace(0, sr/2, n_fft//2 + 1)
    bin_indices = [np.where((freqs >= low) & (freqs < high))[0] 
                  for low, high in freq_bands]
    
    # 频带权重（根据听觉灵敏度）
    weights = [0.2, 0.5, 0.3]  # 中高频赋予更高权重
    total_score = 0
    for idx, w in zip(bin_indices, weights):
        a_band = a[:, idx].flatten()
        b_band = b[:, idx].flatten()
        
        # 跳过全零频带
        if np.all(a_band == 0) and np.all(b_band == 0):
            continue
            
        # 计算带内相似度
        score = matrix_cosine_similarity(a_band, b_band)
        total_score += w * score
    
    return total_score / sum(weights)


if __name__ == "__main__":
    audio_dir = "/home/panchanghao/2025-ICML/evaluation/test-data/test_FireRedTTS"
    feature_npy_files = sorted(glob.glob(os.path.join(audio_dir, "*_feature.npy")))
    gt_npy_files = [x for x in feature_npy_files if '[gt]' in x]
    
    IPD_MAE_LIST = []
    ILD_MAE_LIST = []
    
    ipd_scores = []
    ild_scores = []
    
    for gt_npy_file in tqdm.tqdm(gt_npy_files,desc='evaluate IPD and ILD'):
        pred_npy_file = gt_npy_file.replace('[gt]', '[pred]')
        gt_IPD, gt_ILD = read_IPD_ILD_from_npy(gt_npy_file)
        # print(f'gt_IPD维度 {gt_IPD.shape}, gt_ILD维度 {gt_ILD.shape}')
        pred_IPD, pred_ILD = read_IPD_ILD_from_npy(pred_npy_file)
        # print(f'pred_IPD维度{pred_IPD.shape}, pred_ILD维度 {pred_ILD.shape}')
        IPD_mae = calculate_mae_time(gt_IPD, pred_IPD)
        ILD_mae = calculate_mae_time(gt_ILD, pred_ILD)
        IPD_MAE_LIST.append(IPD_mae)
        ILD_MAE_LIST.append(ILD_mae)
        # 计算IPD余弦相似度（带相位修正）
        ipd_score = matrix_cosine_similarity(gt_IPD, pred_IPD)
        
        # 计算ILD余弦相似度（带能量归一化）
        ild_score = matrix_cosine_similarity(gt_ILD, pred_ILD)
        
        ipd_scores.append(ipd_score)
        ild_scores.append(ild_score)
    
    ipd_mae_loss = round( sum(IPD_MAE_LIST) / len(IPD_MAE_LIST) * 100 , 4)
    ild_mae_loss = round( sum(ILD_MAE_LIST) / len(ILD_MAE_LIST) * 100 , 4)
    
    print(f'IPD mae(x100): {ipd_mae_loss}')
    print(f'ILD mae(x100): {ild_mae_loss}')
    
    avg_ipd = np.mean(ipd_scores)
    avg_ild = np.mean(ild_scores)
    
    print(f"IPD Cosine Similarity: {avg_ipd:.4f} (-1~1 scale)")
    print(f"ILD Cosine Similarity: {avg_ild:.4f} (-1~1 scale)")
    
    dis_maes = []
    azi_maes = []
    
    dis_scores = []
    azi_scores = []
    
    dis_npy_files = sorted(glob.glob(os.path.join(audio_dir, "*_dis.npy")))
    gt_dis_npy_files = [x for x in dis_npy_files if '[gt]' in x]
    
    for gt_dis_npy_file in tqdm.tqdm(gt_dis_npy_files,desc='evaluate distance sos'):
        # print(gt_dis_npy_file)
        pred_dis_npy_file = gt_dis_npy_file.replace('[gt]', '[pred]')
        assert os.path.exists(pred_dis_npy_file)
        gt_dis_tokens = np.load(gt_dis_npy_file, allow_pickle=True)
        pred_dis_tokens = np.load(pred_dis_npy_file, allow_pickle=True)
        
        # print(gt_dis_tokens.shape, pred_dis_tokens.shape)
        
        dis_score = matrix_cosine_similarity(gt_dis_tokens, pred_dis_tokens)
        dis_mae = calculate_mae(gt_dis_tokens, pred_dis_tokens)
        dis_scores.append(dis_score)
        dis_maes.append(dis_mae)
        
    azi_npy_files = sorted(glob.glob(os.path.join(audio_dir, "*_azi.npy")))
    gt_azi_npy_files = [x for x in azi_npy_files if '[gt]' in x]
    
    for gt_azi_npy_file in tqdm.tqdm(gt_azi_npy_files,desc='evaluate azimuth cos'):
        pred_azi_npy_file = gt_azi_npy_file.replace('[gt]', '[pred]')
        gt_azi_tokens = np.load(gt_azi_npy_file, allow_pickle=True)
        pred_azi_tokens = np.load(pred_azi_npy_file, allow_pickle=True)
        
        azi_score = matrix_cosine_similarity(gt_azi_tokens, pred_azi_tokens)
        azi_scores.append(azi_score)
        azi_mae =calculate_mae(gt_azi_tokens, pred_azi_tokens)
        azi_maes.append(azi_mae)
        
    dis_cos = np.mean(dis_scores)
    azi_cos = np.mean(azi_scores)
    dis_mae = np.mean(dis_maes)
    azi_mae = np.mean(azi_maes)
    print(f'Distance mae: {dis_mae}')
    print(f'Azimuth mae: {azi_mae}')
    
    print(f"dis Cosine Similarity: {dis_cos:.4f} (-1~1 scale)")
    print(f"azi Cosine Similarity: {azi_cos:.4f} (-1~1 scale)")