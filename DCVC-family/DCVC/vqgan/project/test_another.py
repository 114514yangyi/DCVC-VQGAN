import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import subprocess
from scipy.interpolate import interp1d
from tqdm import tqdm

# ================= 配置区域 =================
# 神经网络结果 CSV 字典
# - 键: "模型名"
# - 值: 该模型在多个压缩点下的 CSV 文件列表
#   每个 CSV 内容格式: 原始路径, 重建路径, 压缩大小(Byte)
#   示例:
# NEURAL_DATA_SOURCES = {
#     "Ours_VQVAE": [
#         "csv_data/ours_q1.csv",
#         "csv_data/ours_q2.csv",
#         "csv_data/ours_q3.csv",
#     ],
#     "DVC": [
#         "csv_data/dvc_q1.csv",
#         "csv_data/dvc_q2.csv",
#     ],
# }

NEURAL_DATA_SOURCES = {
     "DVC": [
        "/data/huyang/DVC/video_comparison_L256.csv",
        "/data/huyang/DVC/video_comparison_L512.csv",
        "/data/huyang/DVC/video_comparison_L1024.csv",
        "/data/huyang/DVC/video_comparison_L2048.csv"
    ]
}

OUTPUT_DIR = "./paper_results"
CRF_VALUES = [ 20, 30, 40, 50] # 传统方法对比点
DEVICE = 'cuda:7' 
RAW_VIDEO_DIR = "/data/huyang/data/test"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 如果已经存在汇总好的指标 CSV，则优先复用其中“已计算过的方法”，避免重复计算
USE_CACHED_METRICS_IF_AVAILABLE = True

# ================= 核心计算函数 =================

def get_video_info(path):
    cap = cv2.VideoCapture(path)
    w, h, cnt = int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
    cap.release()
    return w, h, cnt

def calculate_metrics(real_p, dist_p):
    """计算指标：PSNR, SSIM, FID, IpIPS(LPIPS), 逐帧PSNR（用于mp4视频）"""
    cap_r = cv2.VideoCapture(real_p)
    cap_d = cv2.VideoCapture(dist_p)
    psnrs, ssims, ipips_list = [], [], []
    # 修正FID：使用标准2048维特征（Inception v3最后一层）
    fid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(DEVICE)
    
    while True:
        ret1, f1 = cap_r.read()
        ret2, f2 = cap_d.read()
        if not ret1 or not ret2: break
        
        # PSNR计算：确保使用float类型
        mse = np.mean((f1.astype(float) - f2.astype(float)) ** 2)
        psnrs.append(10 * np.log10(255.0**2 / mse) if mse > 0 else 100)
        
        # 修正SSIM：计算彩色SSIM（关键修正）
        # 使用channel_axis=2指定通道维度，data_range=255指定8位图像范围
        ssim_val = ssim(f1, f2, channel_axis=2, data_range=255)
        ssims.append(ssim_val)
        
        # FID 数据准备
        t1 = torch.from_numpy(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).to(DEVICE)
        t2 = torch.from_numpy(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).to(DEVICE)
        fid_metric.update(t1, real=True)
        fid_metric.update(t2, real=False)

        # IpIPS (LPIPS) 数据准备：归一化到 [0, 1]（LPIPS内部会处理到[-1,1]）
        t1_lp = t1.float() / 255.0
        t2_lp = t2.float() / 255.0
        ipips_val = lpips_metric(t1_lp, t2_lp).item()
        ipips_list.append(ipips_val)
        
    cap_r.release(); cap_d.release()
    return (
        np.mean(psnrs),
        np.mean(ssims),
        fid_metric.compute().item(),
        np.mean(ipips_list) if len(ipips_list) > 0 else 0.0,
        psnrs,
    )

def calculate_metrics_from_yuv(yuv_orig, yuv_compressed, w, h, frame_count):
    """从YUV文件计算指标：PSNR, SSIM, FID, IpIPS(LPIPS), 逐帧PSNR"""
    # YUV420P格式：每帧大小为 w*h*1.5 (Y平面: w*h, U平面: w*h/4, V平面: w*h/4)
    y_size = w * h
    uv_size = w * h // 4
    frame_size = y_size + uv_size * 2
    
    psnrs, ssims, ipips_list = [], [], []
    # 修正FID：使用标准2048维特征
    fid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(DEVICE)
    
    with open(yuv_orig, 'rb') as f_orig, open(yuv_compressed, 'rb') as f_comp:
        for frame_idx in range(frame_count):
            # 读取Y平面
            y_data_orig = f_orig.read(y_size)
            y_data_comp = f_comp.read(y_size)
            
            if len(y_data_orig) < y_size or len(y_data_comp) < y_size:
                break  # 文件读取完毕
                
            y_orig = np.frombuffer(y_data_orig, dtype=np.uint8).reshape(h, w)
            y_comp = np.frombuffer(y_data_comp, dtype=np.uint8).reshape(h, w)
            
            # 读取U和V平面（需要用于重建完整RGB图像）
            u_data_orig = f_orig.read(uv_size)
            v_data_orig = f_orig.read(uv_size)
            u_data_comp = f_comp.read(uv_size)
            v_data_comp = f_comp.read(uv_size)
            
            if len(u_data_orig) < uv_size or len(v_data_orig) < uv_size:
                break
            
            # 重建U和V平面（从420到444，即上采样到原始尺寸）
            u_orig = np.frombuffer(u_data_orig, dtype=np.uint8).reshape(h//2, w//2)
            v_orig = np.frombuffer(v_data_orig, dtype=np.uint8).reshape(h//2, w//2)
            u_comp = np.frombuffer(u_data_comp, dtype=np.uint8).reshape(h//2, w//2)
            v_comp = np.frombuffer(v_data_comp, dtype=np.uint8).reshape(h//2, w//2)
            
            # 上采样U和V到原始尺寸（使用CUBIC插值可能更准确）
            u_orig_upsampled = cv2.resize(u_orig, (w, h), interpolation=cv2.INTER_CUBIC)
            v_orig_upsampled = cv2.resize(v_orig, (w, h), interpolation=cv2.INTER_CUBIC)
            u_comp_upsampled = cv2.resize(u_comp, (w, h), interpolation=cv2.INTER_CUBIC)
            v_comp_upsampled = cv2.resize(v_comp, (w, h), interpolation=cv2.INTER_CUBIC)
            
            # 合并YUV为BGR图像
            # 注意：YUV420P通常使用YCrCb颜色空间，使用COLOR_YCrCb2BGR可能更准确
            yuv_orig_frame = np.stack([y_orig, u_orig_upsampled, v_orig_upsampled], axis=2)
            yuv_comp_frame = np.stack([y_comp, u_comp_upsampled, v_comp_upsampled], axis=2)
            # 尝试使用YCrCb转换（YCrCb就是YUV的一种变体）
            frame_orig = cv2.cvtColor(yuv_orig_frame, cv2.COLOR_YCrCb2BGR)
            frame_comp = cv2.cvtColor(yuv_comp_frame, cv2.COLOR_YCrCb2BGR)
            
            # PSNR计算
            mse = np.mean((frame_orig.astype(float) - frame_comp.astype(float)) ** 2)
            psnrs.append(10 * np.log10(255.0**2 / mse) if mse > 0 else 100)
            
            # 修正SSIM：计算彩色SSIM（关键修正）
            ssim_val = ssim(frame_orig, frame_comp, channel_axis=2, data_range=255)
            ssims.append(ssim_val)
            
            # FID 数据准备
            t1 = torch.from_numpy(cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).to(DEVICE)
            t2 = torch.from_numpy(cv2.cvtColor(frame_comp, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).to(DEVICE)
            fid_metric.update(t1, real=True)
            fid_metric.update(t2, real=False)

            # IpIPS (LPIPS) 数据准备：归一化到 [0, 1]（LPIPS内部会处理到[-1,1]）
            t1_lp = t1.float() / 255.0
            t2_lp = t2.float() / 255.0
            ipips_val = lpips_metric(t1_lp, t2_lp).item()
            ipips_list.append(ipips_val)
    
    return (
        np.mean(psnrs),
        np.mean(ssims),
        fid_metric.compute().item(),
        np.mean(ipips_list) if len(ipips_list) > 0 else 0.0,
        psnrs,
    )

def validate_metrics_calculation():
    """验证指标计算是否正确"""
    print(">>> 开始验证指标计算...")
    
    # 创建测试图像
    test_img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    test_img2 = test_img1.copy()
    test_img2[100:150, 100:150] = 128  # 添加一些失真
    
    # 计算PSNR
    mse = np.mean((test_img1.astype(float) - test_img2.astype(float)) ** 2)
    psnr_manual = 10 * np.log10(255.0**2 / mse) if mse > 0 else 100
    
    # 使用OpenCV的PSNR函数验证
    psnr_opencv = cv2.PSNR(test_img1, test_img2)
    
    print(f"手动计算PSNR: {psnr_manual:.2f} dB")
    print(f"OpenCV PSNR: {psnr_opencv:.2f} dB")
    print(f"差异: {abs(psnr_manual - psnr_opencv):.6f}")
    
    # 验证SSIM
    ssim_val = ssim(test_img1, test_img2, channel_axis=2, data_range=255)
    print(f"彩色SSIM: {ssim_val:.4f}")
    
    # 验证完全相同图像的PSNR和SSIM
    psnr_same = cv2.PSNR(test_img1, test_img1)
    ssim_same = ssim(test_img1, test_img1, channel_axis=2, data_range=255)
    print(f"\n完全相同图像:")
    print(f"  PSNR: {psnr_same:.2f} dB (应该接近无穷大)")
    print(f"  SSIM: {ssim_same:.4f} (应该为1.0)")
    
    is_valid = abs(psnr_manual - psnr_opencv) < 0.01
    print(f"\n验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
    return is_valid

def bd_rate(rate_a, dist_a, rate_t, dist_t):
    """计算 BD-Rate (以 H.265 为基准)"""
    try:
        # 转换为numpy数组并过滤无效值
        rate_a = np.array(rate_a)
        dist_a = np.array(dist_a)
        rate_t = np.array(rate_t)
        dist_t = np.array(dist_t)
        
        # 检查数据点数量
        n_a = len(dist_a)
        n_t = len(dist_t)
        
        if n_a < 2 or n_t < 2:
            return 0.0
        
        # 检查是否有有效的数据范围
        if np.all(rate_a <= 0) or np.all(rate_t <= 0):
            return 0.0
        
        l_ra, l_rt = np.log(rate_a), np.log(rate_t)
        
        # 根据数据点数量选择多项式次数（避免条件不良）
        # 需要至少 degree+1 个点才能进行多项式拟合
        degree_a = min(3, n_a - 1)
        degree_t = min(3, n_t - 1)
        
        # 如果数据点太少，使用线性拟合
        if degree_a < 1:
            degree_a = 1
        if degree_t < 1:
            degree_t = 1
        
        # 使用加权最小二乘法，添加条件检查
        p_a = np.polyfit(dist_a, l_ra, degree_a)
        p_t = np.polyfit(dist_t, l_rt, degree_t)
        
        # 计算积分区间
        m = max(min(dist_a), min(dist_t))
        M = min(max(dist_a), max(dist_t))
        
        # 检查积分区间是否有效
        if M <= m or abs(M - m) < 1e-10:
            return 0.0
        
        # 计算积分
        v_a = np.polyint(p_a)
        v_t = np.polyint(p_t)
        i_a = np.polyval(v_a, M) - np.polyval(v_a, m)
        i_t = np.polyval(v_t, M) - np.polyval(v_t, m)
        
        # 计算 BD-Rate
        bd_rate_val = (np.exp((i_t - i_a) / (M - m)) - 1) * 100
        
        # 检查结果是否合理
        if not np.isfinite(bd_rate_val):
            return 0.0
        
        return bd_rate_val
    except Exception as e:
        # 静默处理错误，返回0.0
        return 0.0

# ================= 主评估流程 =================

class VideoPaperVisualizer:
    def __init__(self):
        self.results = []
        self.temporal_samples = {} # 存一份用于画时间曲线和误差图的逐帧数据

    def run(self):
        # 0. 如果已有 metrics_summary.csv，则对“已存在的方法”进行按方法级别缓存
        metrics_csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
        cached_df = None
        cached_methods = set()
        if USE_CACHED_METRICS_IF_AVAILABLE and os.path.exists(metrics_csv_path):
            cached_df = pd.read_csv(metrics_csv_path)
            cached_methods = set(cached_df["Method"].unique())
            print(f">>> 检测到已存在指标文件: {metrics_csv_path}，将对这些方法复用缓存: {sorted(cached_methods)}")

        # 1. 处理所有神经网络 CSV 数据（按方法级别判断是否已在缓存中）
        for method, csv_list in NEURAL_DATA_SOURCES.items():
            if method in cached_methods:
                print(f">>> 方法 {method} 已在缓存中，跳过重新计算。")
                continue
            # 兼容: 如果用户仍然提供单个字符串，则转成列表
            if isinstance(csv_list, str):
                csv_list = [csv_list]

            
            for csv_p in tqdm(csv_list):
                print(f">>> 正在处理 CSV: {csv_p}")
                if not os.path.exists(csv_p):
                    print(f"[警告] 找不到 CSV 文件: {csv_p}，已跳过。")
                    continue

                print(f">>> 正在处理模型: {method}, CSV: {os.path.basename(csv_p)}")
                df = pd.read_csv(csv_p, names=['orig', 'recon', 'size'], header=None)

                # 对这个 CSV 里的所有样本求平均，形成该模型在一个压缩点上的 RD 点
                bpps, psnrs, ssims, fids, ipips_vals = [], [], [], [], []
                sample_for_temporal = None

                for _, row in tqdm(df.iterrows()):
                    if not (os.path.exists(row['orig']) and os.path.exists(row['recon'])):
                        continue
                    w, h, cnt = get_video_info(row['orig'])
                    bpp = (row['size'] * 8) / (w * h * cnt)
                    p, s, f, ip, p_list = calculate_metrics(row['orig'], row['recon'])

                    bpps.append(bpp)
                    psnrs.append(p)
                    ssims.append(s)
                    fids.append(f)
                    ipips_vals.append(ip)

                    # 记录一条样本用于时间曲线和误差图（优先 Ours）
                    if sample_for_temporal is None and len(p_list) > 0:
                        sample_for_temporal = (row['orig'], row['recon'], p_list)

                    # 为该压缩点记录 IpIPS
                    # 注意：这里存的是该 CSV 中所有样本的 LPIPS 平均值
                    # （越小越好）
                    if 'IpIPS' not in locals():
                        IpIPS = []
                    IpIPS.append(ip)

                if len(bpps) == 0:
                    continue

                # 将该 CSV（一个压缩点）的平均结果加入 RD 曲线
                self.results.append({
                    'Method': method,
                    'BPP': np.mean(bpps),
                    'PSNR': np.mean(psnrs),
                    'SSIM': np.mean(ssims),
                    'FID': np.mean(fids),
                    'IpIPS': float(np.mean(ipips_vals)) if len(ipips_vals) > 0 else 0.0,
                })

                # 如果是 Ours 且还没有 temporal_samples，就用这一份
                if "Ours" in method and sample_for_temporal is not None and not self.temporal_samples:
                    orig_p, recon_p, p_list = sample_for_temporal
                    self.temporal_samples = {
                        'p_list': p_list,
                        'orig': orig_p,
                        'recon': recon_p,
                        'label': method
                    }

        # 2. 处理传统基准 H.264/H.265（同样遵循方法级别缓存）
        print(f">>> 正在扫描原始目录: {RAW_VIDEO_DIR}")
        raw_videos = [os.path.join(RAW_VIDEO_DIR, f) for f in os.listdir(RAW_VIDEO_DIR) if f.endswith('.mp4')]
        
        for name, codec in [("H.264", "libx264"), ("H.265", "libx265")]:
            if name in cached_methods:
                print(f">>> 基准方法 {name} 已在缓存中，跳过重新计算。")
                continue
            print(f">>> 正在运行传统编码器: {name}")
            for crf in CRF_VALUES:
                # 为了计算平均值，我们需要在这个 CRF 下跑完所有视频
                temp_bpps, temp_psnrs, temp_ssims, temp_fids, temp_ipips = [], [], [], [], []
                
                for video_p in tqdm(raw_videos):
                    # 直接压缩：与“以前直接压缩 mp4”的方式一致（不做 mp4<->yuv 来回转换）
                    base_name = os.path.splitext(os.path.basename(video_p))[0]
                    w, h, cnt = get_video_info(video_p)
                    
                    # 步骤1: 直接对原始 mp4 编码输出 mp4
                    tmp_out_mp4 = f"tmp_{name}_{crf}_{base_name}.mp4"
                    subprocess.run([
                        'ffmpeg', '-y', '-i', video_p,
                        '-c:v', codec, '-crf', str(crf),
                        '-preset', 'slow', tmp_out_mp4
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # 步骤2: 计算压缩后文件大小（使用 mp4 文件）
                    bpp = (os.path.getsize(tmp_out_mp4) * 8) / (w * h * cnt)
                    
                    # 步骤3: 直接对比“原始 mp4 vs 压缩 mp4”计算指标
                    p, s, f, ip, _ = calculate_metrics(video_p, tmp_out_mp4)
                    
                    temp_bpps.append(bpp)
                    temp_psnrs.append(p)
                    temp_ssims.append(s)
                    temp_fids.append(f)
                    temp_ipips.append(ip)
                    
                    # 清理临时文件
                    os.remove(tmp_out_mp4)

                # 将该 CRF 下所有视频的平均值存入结果
                self.results.append({
                    'Method': name, 
                    'BPP': np.mean(temp_bpps), 
                    'PSNR': np.mean(temp_psnrs), 
                    'SSIM': np.mean(temp_ssims), 
                    'FID': np.mean(temp_fids),
                    'IpIPS': float(np.mean(temp_ipips)) if len(temp_ipips) > 0 else 0.0,
                })
                print(f"    [{name}] CRF {crf} 完成: Avg BPP={np.mean(temp_bpps):.4f}, PSNR={np.mean(temp_psnrs):.2f}")

        # 如果前面没有从 Ours 拿到逐帧样本，这里从传统编码结果中选一个样本来画最后两张图
        if not self.temporal_samples and raw_videos:
            sample_v = raw_videos[0]
            sample_name, sample_codec = "H.265", "libx265"
            crf_sample = CRF_VALUES[len(CRF_VALUES) // 2]  # 取中间的一个 CRF，例如 32
            tmp_out = f"tmp_sample_{sample_name}_{crf_sample}_{os.path.basename(sample_v)}"

            print(f">>> 未找到 Ours 的逐帧数据，使用 {sample_name} (CRF={crf_sample}) 的一个样本来绘制时间曲线和误差图")
            subprocess.run([
                'ffmpeg', '-y', '-i', sample_v,
                '-c:v', sample_codec, '-crf', str(crf_sample),
                '-preset', 'medium', tmp_out
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            _, _, _, _, p_list = calculate_metrics(sample_v, tmp_out)
            self.temporal_samples = {
                'p_list': p_list,
                'orig': sample_v,
                'recon': tmp_out,
                'label': f"{sample_name}_CRF{crf_sample}",
            }

        # 3. 合并“缓存的方法结果”和“本次新计算的方法结果”，并保存为 CSV
        if cached_df is not None and not cached_df.empty:
            new_df = pd.DataFrame(self.results) if len(self.results) > 0 else None
            if new_df is not None and not new_df.empty:
                self.df = pd.concat([cached_df, new_df], ignore_index=True)
            else:
                self.df = cached_df
        else:
            self.df = pd.DataFrame(self.results)

        # 将所有方法在各个压缩点下的指标保存为 CSV，便于后续分析/画图
        metrics_csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
        self.df.to_csv(metrics_csv_path, index=False)
        print(f">>> 已将所有方法的指标保存到: {metrics_csv_path}")

        self.generate_plots()


        # sample_v = df.iloc[0]['orig']
        # for name, codec in [("H.264", "libx264"), ("H.265", "libx265")]:
        #     print(f">>> 正在处理基准: {name}")
        #     for crf in CRF_VALUES:
        #         tmp = "tmp.mp4"
        #         subprocess.run(['ffmpeg', '-y', '-i', sample_v, '-c:v', codec, '-crf', str(crf), tmp], stderr=-1)
        #         w, h, cnt = get_video_info(sample_v)
        #         bpp = (os.path.getsize(tmp) * 8) / (w * h * cnt)
        #         p, s, f, _ = calculate_metrics(sample_v, tmp)
        #         self.results.append({'Method': name, 'BPP': bpp, 'PSNR': p, 'SSIM': s, 'FID': f})
        #         os.remove(tmp)

        # self.df = pd.DataFrame(self.results)
        # self.df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)
        # self.generate_plots()

    def generate_plots(self):
        # --- 图 1: RD Curves (PSNR/SSIM/FID/IpIPS) ---
        for m in ['PSNR', 'SSIM', 'FID', 'IpIPS']:
            plt.figure(figsize=(7, 5))
            for method in self.df['Method'].unique():
                sub = self.df[self.df['Method'] == method].sort_values('BPP')
                plt.plot(sub['BPP'], sub[m], 'o-', label=method, markersize=4)
            plt.title(f'Rate-Distortion Curve ({m})')
            plt.xlabel('bpp'); plt.ylabel(m); plt.legend(); plt.grid(True)
            # 越小越好：FID / IpIPS
            if m in ['FID', 'IpIPS']:
                plt.gca().invert_yaxis()
            plt.savefig(os.path.join(OUTPUT_DIR, f"RD_Curve_{m}.png"), dpi=300)

        # --- 图 2: BD-Rate 柱状图 ---
        plt.figure(figsize=(7, 5))
        methods, bdr_vals = [], []
        h265 = self.df[self.df['Method'] == 'H.265'].sort_values('BPP')
        
        # 检查 H.265 是否有足够的数据点
        if len(h265) < 2:
            print("警告: H.265 数据点不足，无法计算 BD-Rate")
        else:
            for m in self.df['Method'].unique():
                if m in ['H.265', 'H.264']: 
                    continue
                test = self.df[self.df['Method'] == m].sort_values('BPP')
                
                # 检查测试方法是否有足够的数据点
                if len(test) < 2:
                    print(f"警告: 方法 {m} 数据点不足，跳过 BD-Rate 计算")
                    continue
                
                val = bd_rate(h265['BPP'].values, h265['PSNR'].values, 
                             test['BPP'].values, test['PSNR'].values)
                methods.append(m)
                bdr_vals.append(val)
        
        # 只有在有数据时才绘图
        if len(methods) > 0 and len(bdr_vals) > 0:
            plt.bar(methods, bdr_vals, color='skyblue')
            plt.axhline(0, color='black', linewidth=0.8)
            plt.title('BD-Rate Comparison (vs H.265)')
            plt.ylabel('Bitrate Saving (%)')
            plt.savefig(os.path.join(OUTPUT_DIR, "BD_Rate_Bar.png"))
            plt.close()
        else:
            print("警告: 没有足够的数据计算 BD-Rate，跳过绘图")
            plt.close()

        # --- 图 3 & 图 4: 使用 temporal_samples 中的一段视频绘制 ---
        if self.temporal_samples:
            # --- 图 3: 时间稳定性图 ---
            if 'p_list' in self.temporal_samples:
                plt.figure(figsize=(10, 4))
                label = self.temporal_samples.get('label', 'Ours_VQVAE')
                plt.plot(self.temporal_samples['p_list'], label=label, color='red')
                plt.title('Temporal Consistency (PSNR per Frame)')
                plt.xlabel('Frame Index'); plt.ylabel('PSNR (dB)'); plt.legend(); plt.grid(True)
                plt.savefig(os.path.join(OUTPUT_DIR, "Temporal_Consistency.png"))

            # --- 图 4: 视觉误差图 (Error Map) ---
            if 'orig' in self.temporal_samples and 'recon' in self.temporal_samples:
                cap_o = cv2.VideoCapture(self.temporal_samples['orig'])
                cap_r = cv2.VideoCapture(self.temporal_samples['recon'])
                cap_o.set(1, 15); cap_r.set(1, 15) # 取第15帧作为样本
                _, f_o = cap_o.read(); _, f_r = cap_r.read()
                diff = cv2.absdiff(f_o, f_r)
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                # 放大误差并转为热力图
                heatmap = cv2.applyColorMap(cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(OUTPUT_DIR, "Visual_Error_Map.png"), heatmap)
                cv2.imwrite(os.path.join(OUTPUT_DIR, "Visual_Original_Frame.png"), f_o)
                cv2.imwrite(os.path.join(OUTPUT_DIR, "Visual_Recon_Frame.png"), f_r)

if __name__ == "__main__":
    VideoPaperVisualizer().run()