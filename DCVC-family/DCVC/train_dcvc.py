"""
DCVC (Deep Contextual Video Compression) 训练脚本

本脚本实现了 DCVC 模型的四阶段训练流程：
- Stage 1: 预热运动估计（Motion Estimation）和运动编码（Motion Coding）模块
- Stage 2: 冻结运动模块，训练残差编码（Residual Coding）模块
- Stage 3: 继续训练残差编码，加入比特率损失
- Stage 4: 端到端联合优化所有模块

主要特性：
1. 支持视频数据集（MP4文件）训练和验证
2. 使用 CompressAI 预训练的 I-frame 模型进行帧内编码
3. 支持分布式训练（通过 Hugging Face Accelerate）
4. 支持混合精度训练（FP16/BF16）
5. 支持指数移动平均（EMA）模型
6. 支持学习率调度器
7. 自动保存最佳模型检查点

训练流程：
1. 每个 GOP（Group of Pictures）的第一帧使用 I-frame 模型压缩
2. 后续 P-frames 使用 DCVC 模型，以前一帧的重建结果作为参考
3. 损失函数 = MSE + lambda * BPP（比特率-失真权衡）
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import math
from PIL import Image
import torchvision.transforms as transforms
import time
import logging
import datetime
from timm.utils import unwrap_model
from torch_ema import ExponentialMovingAverage

# Hugging Face Accelerate 用于分布式训练和混合精度
from accelerate import Accelerator
from accelerate.utils import set_seed

# DCVC P-frame 模型
from src.models.DCVC_net_compressai import DCVC_net

# CompressAI 提供的预训练 I-frame 模型和优化器工具
from compressai.zoo import models as compressai_models
from compressai.optimizers import net_aux_optimizer

# 视频数据集类
from dataset import VideoGOPDataset, VideoValidationDataset

# 禁用 PyTorch 动态编译的错误报告（用于兼容性）
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# 设置确定性行为（确保可复现性）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from compressai.optimizers import net_aux_optimizer

def configure_optimizers(net, args):
    """
    配置优化器，分离主网络参数和辅助参数（如熵模型的参数）
    
    DCVC 模型包含两类参数：
    1. 主网络参数（编码器、解码器等）
    2. 辅助参数（熵模型的量化参数等）
    
    Args:
        net: DCVC 模型实例
        args: 命令行参数，包含 learning_rate
    
    Returns:
        optimizer: 主网络优化器（Adam）
    """
    learning_rate = args.learning_rate
    # 配置主网络和辅助网络的优化器（都使用 Adam）
    conf = {
        "net": {"type": "Adam", "lr": learning_rate},
        "aux": {"type": "Adam", "lr": learning_rate},
    }
    # CompressAI 的工具函数会自动分离参数
    optimizer = net_aux_optimizer(net, conf)
    # 只返回主网络优化器（辅助参数由模型内部管理）
    return optimizer["net"]


def setup_device_for_mps_fallback(device, logger=None):
    """
    Setup device with MPS fallback support.
    If MPS device is detected, enable CPU fallback for unsupported operations.
    
    Args:
        device: torch.device object
        logger: Optional logger for logging messages
    
    Returns:
        device: torch.device object (may be changed to CPU if MPS has issues)
    """
    # Check if MPS is available and being used
    if device.type == 'mps':
        # Enable MPS fallback to CPU for unsupported operations
        import os
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        if logger:
            logger.warning("MPS device detected. Enabling CPU fallback for unsupported operations.")
            logger.warning("This may result in slower performance but ensures compatibility.")
        else:
            print("Warning: MPS device detected. Enabling CPU fallback for unsupported operations.")
            print("This may result in slower performance but ensures compatibility.")
    
    return device


def setup_logging(checkpoint_dir, accelerator):
    """
    设置日志配置，支持分布式训练
    
    在分布式训练中，只有主进程（rank 0）会记录详细日志到文件和控制台。
    其他进程只记录警告和错误信息，避免日志重复。
    
    Args:
        checkpoint_dir: 检查点保存目录
        accelerator: Accelerate 对象，用于判断是否为主进程
    
    Returns:
        logger: 配置好的日志记录器
    """
    log_dir = os.path.join(checkpoint_dir, 'logs')
    
    if accelerator.is_main_process:
        # 主进程：创建日志目录和文件
        os.makedirs(log_dir, exist_ok=True)
        
        # 使用时间戳命名日志文件，避免覆盖
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        # 配置日志：同时输出到文件和控制台
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),  # 文件输出
                logging.StreamHandler()          # 控制台输出
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized. Log file: {log_file}")
    else:
        # 非主进程：只记录警告和错误，减少日志量
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
    
    return logger


def compress_i_frame_with_padding(i_frame_model, frame_tensor, calculate_bpp=True):
    """
    压缩 I-frame（帧内编码），并进行必要的填充处理
    
    I-frame 模型通常要求输入尺寸是 16 的倍数（由于下采样操作）。
    因此需要对不满足条件的帧进行填充，压缩后再裁剪回原始尺寸。
    
    Args:
        i_frame_model: CompressAI 的 I-frame 压缩模型（期望 RGB 输入）
        frame_tensor: 输入 RGB 帧张量，形状 [B, C(=3), H, W]，值域 [0, 1]
        calculate_bpp: 是否计算准确的 BPP（Bits Per Pixel，每像素比特数）
    
    Returns:
        如果 calculate_bpp=True: 返回 (encoded_result, bpp) 元组
            - encoded_result['x_hat']: 重建的 RGB 帧
            - bpp: 计算得到的比特率
        如果 calculate_bpp=False: 只返回 encoded_result
    """
    B, C, H, W = frame_tensor.shape

    # 计算需要填充的像素数（使尺寸成为 16 的倍数）
    padding_r = H % 16  # 右侧填充
    padding_b = W % 16  # 底部填充

    # 如果需要填充，使用反射填充（reflect）模式
    # 反射填充可以避免边界伪影
    if padding_r > 0 or padding_b > 0:
        # pad 参数格式：(left, right, top, bottom)
        rgb_padded = torch.nn.functional.pad(frame_tensor, (0, padding_b, 0, padding_r), mode='reflect')
    else:
        rgb_padded = frame_tensor

    # 在 RGB 空间进行压缩
    encoded = i_frame_model(rgb_padded)

    # 获取重建的 RGB 帧
    x_hat_rgb = encoded['x_hat']

    # 如果之前进行了填充，现在需要裁剪回原始尺寸
    if padding_r > 0 or padding_b > 0:
        x_hat_rgb = x_hat_rgb[:, :, :H, :W]

    encoded['x_hat'] = x_hat_rgb

    # 如果需要计算 BPP
    if calculate_bpp:
        num_pixels = B * H * W
        total_bits = 0.0
        import math
        # 遍历所有似然值（latent codes 的概率分布）
        for likelihood in encoded['likelihoods'].values():
            # 使用熵编码的理论比特数：-log2(p(x))
            bits = torch.log(likelihood).sum() / (-math.log(2))
            total_bits += bits
        # BPP = 总比特数 / 像素数
        bpp = total_bits.item() / num_pixels
        return encoded, bpp
    else:
        return encoded


def compress_p_frame_with_padding(model, ref_frame, current_frame, stage):
    """
    压缩 P-frame（帧间编码），并进行必要的填充处理
    
    P-frame 使用 DCVC 模型进行压缩，需要参考帧（reference frame）和当前帧。
    同样需要处理尺寸填充问题。
    
    Args:
        model: DCVC P-frame 压缩模型
        ref_frame: 参考帧张量 [B, C(=3), H, W]，值域 [0, 1]
        current_frame: 当前帧张量 [B, C(=3), H, W]，值域 [0, 1]
        stage: 训练阶段（1-4），影响模型的前向传播行为
    
    Returns:
        dict: 包含压缩结果的字典，主要字段：
            - "recon_image": 重建的当前帧（已裁剪回原始尺寸）
            - "bpp_train": 训练时的比特率估计
            - "loss": 总损失（MSE + lambda * BPP）
            - 其他中间结果
    """
    B, C, H, W = current_frame.shape

    # 计算填充（与 I-frame 逻辑相同）
    padding_r, padding_b = H % 16, W % 16

    # 对参考帧和当前帧都进行填充（保持尺寸一致）
    if padding_r > 0 or padding_b > 0:
        ref_padded = torch.nn.functional.pad(ref_frame, (0, padding_b, 0, padding_r), mode='reflect')
        current_padded = torch.nn.functional.pad(current_frame, (0, padding_b, 0, padding_r), mode='reflect')
    else:
        ref_padded = ref_frame
        current_padded = current_frame

    # 使用填充后的帧进行 P-frame 压缩
    result = model(ref_padded, current_padded, stage=stage)

    # 从重建帧中移除填充（裁剪回原始尺寸）
    if padding_r > 0 or padding_b > 0:
        result["recon_image"] = result["recon_image"][:, :, :H, :W]

    return result


def calculate_rgb_metrics(rgb_ref, rgb_target):
    """
    计算两个 RGB 张量之间的 MSE 和 PSNR 指标
    
    MSE (Mean Squared Error): 均方误差，衡量重建质量
    PSNR (Peak Signal-to-Noise Ratio): 峰值信噪比，单位 dB，值越大表示质量越好
    
    Args:
        rgb_ref: 参考 RGB 张量，形状 (B, 3, H, W)
        rgb_target: 目标 RGB 张量，形状 (B, 3, H, W)
    
    Returns:
        dict: 包含 'mse' 和 'psnr' 的字典
    """
    # 计算均方误差（对所有像素、所有通道、所有批次求平均）
    mse = F.mse_loss(rgb_ref, rgb_target, reduction='mean').item()

    # 计算 PSNR（假设输入值域为 [0, 1]，峰值信号为 1）
    # PSNR = 10 * log10(MAX^2 / MSE) = 10 * log10(1 / MSE)
    # 如果 MSE 为 0，PSNR 为无穷大（完美重建）
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else float('inf')

    return {
        'mse': mse,
        'psnr': psnr
    }


def evaluate_video(model, i_frame_model, video_dir, device, stage, max_frames=96, accelerator=None):
    """
    在视频数据集上评估 DCVC 模型
    
    评估流程：
    1. 每个视频序列的第一帧使用 I-frame 模型压缩
    2. 后续帧使用 DCVC P-frame 模型，以前一帧的重建结果作为参考
    3. 计算每帧的 BPP、PSNR、MSE 等指标
    4. 统计整体和分类（I-frame、P-frame）的平均指标
    
    Args:
        model: DCVC P-frame 模型
        i_frame_model: CompressAI I-frame 模型
        video_dir: 验证视频目录路径
        device: 计算设备（CPU/GPU）
        stage: 训练阶段（用于模型前向传播）
        max_frames: 每个序列最多评估的帧数
        accelerator: Accelerate 对象（用于分布式训练）
    
    Returns:
        dict: 包含评估结果的字典，包括：
            - avg_psnr, avg_bpp, avg_mse: 所有帧的平均指标
            - avg_i_frame_psnr/bpp/mse: I-frame 的平均指标
            - avg_p_frame_psnr/bpp/mse: P-frame 的平均指标
            - detailed_results: 每个序列的详细结果
        如果数据集为空，返回 None
    """
    # 设置为评估模式（禁用 dropout、batch norm 的更新等）
    model.eval()
    i_frame_model.eval()

    # 创建验证数据集（从 MP4 视频文件读取）
    eval_dataset = VideoValidationDataset(
        video_dir=video_dir,
        transform=transforms.ToTensor(),  # 将 PIL Image 转换为 Tensor [0, 1]
        max_frames=max_frames
    )

    if len(eval_dataset) == 0:
        if accelerator is None or accelerator.is_main_process:
            print(f"Warning: No video sequences found in {video_dir}")
        return None

    results = {}  # 存储每个序列的结果

    if accelerator is None or accelerator.is_main_process:
        print(f"  Evaluating on {len(eval_dataset)} video sequences")

    # 禁用梯度计算（评估时不需要反向传播）
    with torch.no_grad():
        # 遍历所有视频序列
        for seq_idx in range(len(eval_dataset)):
            try:
                sequence_data = eval_dataset[seq_idx]
            except Exception as e:
                # 如果加载序列失败，记录错误并中断评估
                if accelerator is None or accelerator.is_main_process:
                    print(f"Error loading sequence {seq_idx} from evaluation dataset: {e}")
                raise  # 直接抛出异常，中断评估
            
            if sequence_data is None:
                if accelerator is None or accelerator.is_main_process:
                    print(f"Warning: Sequence {seq_idx} returned None, skipping...")
                continue

            frames = sequence_data['frames'].unsqueeze(0).to(device)
            seq_name = sequence_data['name']
            num_frames = sequence_data['num_frames']

            if accelerator is None or accelerator.is_main_process:
                print(f"\n=== Sequence: {seq_name} ({num_frames} frames) ===")

            total_bpp = 0
            total_mse = 0
            total_psnr = 0

            # Track P-frame statistics separately
            p_frame_bpp_sum = 0
            p_frame_mse_sum = 0
            p_frame_psnr_sum = 0
            p_frame_count = 0

            # Store first 3 P-frame details
            first_three_p_frames = []

            # ========== 处理 I-frame（第一帧）==========
            # 提取第一帧并添加批次维度
            i_frame = frames[0, 0, ...]  # 形状: (C, H, W)
            i_frame_batch = i_frame.unsqueeze(0)  # 形状: (1, C, H, W)

            # 使用 I-frame 模型压缩第一帧，并计算 BPP
            i_frame_result, i_frame_bpp = compress_i_frame_with_padding(
                i_frame_model, i_frame_batch, calculate_bpp=True
            )
            # 获取重建的 I-frame 作为后续 P-frame 的参考帧
            ref_frame = i_frame_result['x_hat']  # RGB，形状: (1, C, H, W)

            # 计算 I-frame 的重建质量指标
            i_frame_metric = calculate_rgb_metrics(ref_frame, i_frame_batch)
            i_frame_mse = i_frame_metric['mse']
            i_frame_psnr = i_frame_metric['psnr']

            if accelerator is None or accelerator.is_main_process:
                print(f"Frame 0 (I-frame): BPP={i_frame_bpp:.6f}, PSNR={i_frame_psnr:.2f} dB, MSE={i_frame_mse:.6f}")

            # 将 I-frame 的指标累加到总计中
            total_bpp += i_frame_bpp
            total_mse += i_frame_mse
            total_psnr += i_frame_psnr

            # ========== 处理 P-frames（后续帧）==========
            for frame_idx in range(1, num_frames):
                # 提取当前帧
                current_frame = frames[0, frame_idx, ...].unsqueeze(0)

                # 使用 DCVC 模型压缩 P-frame（需要参考帧）
                result = compress_p_frame_with_padding(model, ref_frame, current_frame, stage)

                # 提取 P-frame 的指标
                p_frame_bpp = result["bpp_train"].item()  # 训练时的 BPP 估计
                p_frame_metric = calculate_rgb_metrics(result["recon_image"], current_frame)
                p_frame_psnr = p_frame_metric['psnr']
                p_frame_mse = p_frame_metric['mse']

                if accelerator is None or accelerator.is_main_process:
                    print(f"Frame {frame_idx} (P-frame): BPP={p_frame_bpp:.6f}, "
                          f"PSNR={p_frame_psnr:.2f} dB, MSE={p_frame_mse:.6f}")

                # 累加到总计
                total_bpp += p_frame_bpp
                total_mse += p_frame_mse
                total_psnr += p_frame_psnr

                # 单独统计 P-frame 的指标（用于计算 P-frame 平均值）
                p_frame_bpp_sum += p_frame_bpp
                p_frame_mse_sum += p_frame_mse
                p_frame_psnr_sum += p_frame_psnr
                p_frame_count += 1

                # 保存前 3 个 P-frame 的详细信息（用于详细分析）
                if frame_idx <= 3:
                    first_three_p_frames.append({
                        'frame_idx': frame_idx,
                        'bpp': p_frame_bpp,
                        'psnr': p_frame_psnr,
                        'mse': p_frame_mse
                    })

                # 更新参考帧为当前帧的重建结果（用于下一帧）
                ref_frame = result["recon_image"]

            # Calculate averages (including I-frame in all metrics)
            avg_mse = total_mse / num_frames
            avg_psnr = total_psnr / num_frames
            avg_bpp = total_bpp / num_frames

            # Calculate P-frame averages
            avg_p_frame_bpp = p_frame_bpp_sum / p_frame_count if p_frame_count > 0 else 0
            avg_p_frame_mse = p_frame_mse_sum / p_frame_count if p_frame_count > 0 else 0
            avg_p_frame_psnr = p_frame_psnr_sum / p_frame_count if p_frame_count > 0 else 0

            results[seq_name] = {
                'avg_psnr': avg_psnr,
                'avg_bpp': avg_bpp,
                'avg_mse': avg_mse,
                'num_frames': num_frames,
                'i_frame_bpp': i_frame_bpp,
                'i_frame_psnr': i_frame_psnr,
                'i_frame_mse': i_frame_mse,
                'avg_p_frame_bpp': avg_p_frame_bpp,
                'avg_p_frame_psnr': avg_p_frame_psnr,
                'avg_p_frame_mse': avg_p_frame_mse,
                'p_frame_count': p_frame_count,
                'first_three_p_frames': first_three_p_frames
            }

    # Calculate overall averages
    if results:
        psnr_list = [results[seq]['avg_psnr'] for seq in results]
        bpp_list = [results[seq]['avg_bpp'] for seq in results]
        mse_list = [results[seq]['avg_mse'] for seq in results]

        # I-frame specific metrics
        i_frame_psnr_list = [results[seq]['i_frame_psnr'] for seq in results]
        i_frame_bpp_list = [results[seq]['i_frame_bpp'] for seq in results]
        i_frame_mse_list = [results[seq]['i_frame_mse'] for seq in results]

        # P-frame specific metrics
        p_frame_psnr_list = [results[seq]['avg_p_frame_psnr'] for seq in results if results[seq]['p_frame_count'] > 0]
        p_frame_bpp_list = [results[seq]['avg_p_frame_bpp'] for seq in results if results[seq]['p_frame_count'] > 0]
        p_frame_mse_list = [results[seq]['avg_p_frame_mse'] for seq in results if results[seq]['p_frame_count'] > 0]

        overall_results = {
            'avg_psnr': sum(psnr_list) / len(psnr_list),
            'avg_bpp': sum(bpp_list) / len(bpp_list),
            'avg_mse': sum(mse_list) / len(mse_list),
            'num_sequences': len(psnr_list),
            'avg_i_frame_psnr': sum(i_frame_psnr_list) / len(i_frame_psnr_list),
            'avg_i_frame_bpp': sum(i_frame_bpp_list) / len(i_frame_bpp_list),
            'avg_i_frame_mse': sum(i_frame_mse_list) / len(i_frame_mse_list),
            'avg_p_frame_psnr': sum(p_frame_psnr_list) / len(p_frame_psnr_list) if p_frame_psnr_list else 0,
            'avg_p_frame_bpp': sum(p_frame_bpp_list) / len(p_frame_bpp_list) if p_frame_bpp_list else 0,
            'avg_p_frame_mse': sum(p_frame_mse_list) / len(p_frame_mse_list) if p_frame_mse_list else 0,
            'detailed_results': results  # Store per-sequence detailed results
        }

        if accelerator is None or accelerator.is_main_process:
            print(f"\n=== OVERALL RESULTS ===")
            print(f"All Frames - Avg BPP={overall_results['avg_bpp']:.6f}, "
                  f"Avg PSNR={overall_results['avg_psnr']:.2f} dB, "
                  f"Avg MSE={overall_results['avg_mse']:.6f} "
                  f"({overall_results['num_sequences']} sequences)")
            print(f"\nI-frame avg: BPP={overall_results['avg_i_frame_bpp']:.6f}, "
                  f"PSNR={overall_results['avg_i_frame_psnr']:.2f} dB, "
                  f"MSE={overall_results['avg_i_frame_mse']:.6f}")
            print(f"P-frame avg: BPP={overall_results['avg_p_frame_bpp']:.6f}, "
                  f"PSNR={overall_results['avg_p_frame_psnr']:.2f} dB, "
                  f"MSE={overall_results['avg_p_frame_mse']:.6f}")

            # Print first 3 P-frame statistics (averaged across all sequences)
            print(f"\n=== FIRST 3 P-FRAMES DETAILED STATISTICS ===")
            for p_idx in range(1, 4):
                bpp_values = []
                psnr_values = []
                mse_values = []
                for seq_name, seq_results in results.items():
                    for p_frame_info in seq_results['first_three_p_frames']:
                        if p_frame_info['frame_idx'] == p_idx:
                            bpp_values.append(p_frame_info['bpp'])
                            psnr_values.append(p_frame_info['psnr'])
                            mse_values.append(p_frame_info['mse'])

                if bpp_values:
                    avg_bpp = sum(bpp_values) / len(bpp_values)
                    avg_psnr = sum(psnr_values) / len(psnr_values)
                    avg_mse = sum(mse_values) / len(mse_values)
                    print(f"P-frame {p_idx}: BPP={avg_bpp:.6f}, PSNR={avg_psnr:.2f} dB, MSE={avg_mse:.6f}")

        return overall_results

    return None


def train_one_epoch(model, i_frame_model, train_loader, optimizer, device, stage, epoch,
                   gradient_accumulation_steps=1, use_gop_optimization=False, 
                   grad_clip_max_norm=None, ema=None, accelerator=None, 
                   phase_name="Training"):
    """
    训练一个 epoch 的 DCVC 模型
    
    训练流程：
    1. 每个 GOP 的第一帧使用 I-frame 模型压缩（不参与训练）
    2. 后续 P-frames 使用 DCVC 模型训练
    3. 根据训练阶段冻结/解冻相应的模块参数
    4. 支持两种优化模式：
       - 逐帧优化：每个 P-frame 单独计算梯度并更新
       - GOP 级优化：累积整个 GOP 的梯度后统一更新
    
    Args:
        model: DCVC P-frame 模型
        i_frame_model: CompressAI I-frame 模型（固定，不训练）
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 计算设备
        stage: 训练阶段（1-4）
        epoch: 当前 epoch 编号
        gradient_accumulation_steps: 梯度累积步数（用于模拟更大的 batch size）
        use_gop_optimization: 是否使用 GOP 级优化（True：累积整个 GOP 的梯度）
        grad_clip_max_norm: 梯度裁剪的最大范数（防止梯度爆炸）
        ema: 指数移动平均对象（用于维护模型参数的平滑版本）
        accelerator: Accelerate 对象（用于分布式训练）
        phase_name: 阶段名称（用于日志显示）
    
    Returns:
        dict: 包含训练统计信息的字典
            - "loss": 平均损失
            - "mse": 平均 MSE
            - "psnr": 平均 PSNR
            - "bpp": 平均 BPP
    """
    # 设置模型为训练模式（启用 dropout、batch norm 更新等）
    model.train()
    # I-frame 模型始终为评估模式（不参与训练）
    i_frame_model.eval()
    
    # 获取未包装的模型（用于访问模型内部方法）
    # 在分布式训练中，模型被 Accelerate 包装，需要 unwrap 才能访问原始方法
    unwrapped_model = accelerator.unwrap_model(model) if accelerator else model

    # 更新熵模型的量化参数（quantiles）
    # 这些参数用于估计比特率，需要在训练过程中定期更新
    unwrapped_model.bitEstimator_z._update_quantiles()      # 残差编码的熵模型
    unwrapped_model.bitEstimator_z_mv._update_quantiles()  # 运动编码的熵模型

    # ========== 根据训练阶段控制参数冻结 ==========
    # Stage 2 和 3：冻结运动相关模块，只训练残差编码模块
    if stage in [2, 3]:
        # 冻结所有运动估计和运动编码相关的模块
        for param in unwrapped_model.opticFlow.parameters():           # 光流估计网络
            param.requires_grad = False
        for param in unwrapped_model.mvEncoder.parameters():          # 运动向量编码器
            param.requires_grad = False
        for param in unwrapped_model.mvDecoder_part1.parameters():    # 运动向量解码器（部分1）
            param.requires_grad = False
        for param in unwrapped_model.mvDecoder_part2.parameters():    # 运动向量解码器（部分2）
            param.requires_grad = False
        for param in unwrapped_model.mvpriorEncoder.parameters():     # 运动先验编码器
            param.requires_grad = False
        for param in unwrapped_model.mvpriorDecoder.parameters():    # 运动先验解码器
            param.requires_grad = False
        for param in unwrapped_model.auto_regressive_mv.parameters(): # 运动自回归模型
            param.requires_grad = False
        for param in unwrapped_model.entropy_parameters_mv.parameters(): # 运动熵参数
            param.requires_grad = False
        for param in unwrapped_model.bitEstimator_z_mv.parameters():  # 运动比特率估计器
            param.requires_grad = False
    else:
        # Stage 1 和 4：解冻所有参数（端到端训练）
        for param in unwrapped_model.opticFlow.parameters():
            param.requires_grad = True
        for param in unwrapped_model.mvEncoder.parameters():
            param.requires_grad = True
        for param in unwrapped_model.mvDecoder_part1.parameters():
            param.requires_grad = True
        for param in unwrapped_model.mvDecoder_part2.parameters():
            param.requires_grad = True
        for param in unwrapped_model.mvpriorEncoder.parameters():
            param.requires_grad = True
        for param in unwrapped_model.mvpriorDecoder.parameters():
            param.requires_grad = True
        for param in unwrapped_model.auto_regressive_mv.parameters():
            param.requires_grad = True
        for param in unwrapped_model.entropy_parameters_mv.parameters():
            param.requires_grad = True
        for param in unwrapped_model.bitEstimator_z_mv.parameters():
            param.requires_grad = True
    
    total_loss = 0
    total_mse = 0
    total_bpp = 0
    total_psnr = 0
    n_frames = 0
    
    progress_desc = f"{phase_name} Stage {stage} Epoch {epoch}"
    progress_bar = tqdm(train_loader, desc=progress_desc, 
                       disable=(accelerator and not accelerator.is_main_process))
    
    for batch_idx, gop_batch in enumerate(progress_bar):
        gop_batch = gop_batch.to(device)
        batch_size, gop_size, _, _, _ = gop_batch.shape
        
        # ========== 两种训练模式 ==========
        if use_gop_optimization:
            # ===== 模式 1: GOP 级优化 =====
            # 在整个 GOP 上累积梯度，然后统一更新
            # 优点：考虑帧间依赖关系，训练更稳定
            # 缺点：内存占用更大
            
            optimizer.zero_grad()  # 清空梯度
            batch_loss = 0
            batch_mse = 0
            batch_bpp = 0
            batch_psnr = 0
            
            reference_frames = None
            
            # 遍历 GOP 中的每一帧
            for frame_pos in range(gop_size):
                current_frames = gop_batch[:, frame_pos, :, :, :].to(device)
                
                if frame_pos == 0:
                    # 第一帧：使用 I-frame 模型压缩（不参与训练）
                    with torch.no_grad():
                        i_frame_result, _ = compress_i_frame_with_padding(i_frame_model, current_frames)
                        reference_frames = i_frame_result['x_hat'].detach()  # 分离梯度，作为参考
                else:
                    # P-frames：使用 DCVC 模型，累积损失（不立即反向传播）
                    result = model(reference_frames, current_frames, stage=stage)
                    
                    # 累加损失和统计信息
                    batch_loss += result["loss"]
                    batch_mse += result["mse_loss"].item() * batch_size
                    batch_bpp += result["bpp_train"].item() * batch_size
                    batch_psnr += -10 * math.log10(result["mse_loss"].item()) * batch_size
                    
                    # 使用重建帧作为下一帧的参考（保持梯度连接）
                    reference_frames = result["recon_image"]
            
            # 如果有 P-frames，进行反向传播和参数更新
            if gop_size > 1:
                # 平均化损失（除以 P-frame 数量）
                batch_loss /= (gop_size - 1)
                # 考虑梯度累积步数
                batch_loss /= gradient_accumulation_steps

                # 检查损失是否异常
                if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                    raise RuntimeError("Loss is NaN or Inf, stopping training")

                # 反向传播（累积梯度）
                if accelerator:
                    accelerator.backward(batch_loss)
                else:
                    batch_loss.backward()
                
                # 梯度裁剪（防止梯度爆炸）
                if grad_clip_max_norm is not None:
                    if accelerator:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
                
                # 更新参数
                optimizer.step()
                
                # 更新 EMA（如果启用）
                if ema is not None:
                    ema.update(unwrapped_model.parameters())
                
                # 累加统计信息
                total_loss += batch_loss.item() * (gop_size - 1)
                total_mse += batch_mse
                total_bpp += batch_bpp
                total_psnr += batch_psnr
                n_frames += batch_size * (gop_size - 1)

        else:
            # ===== 模式 2: 逐帧优化 =====
            # 每个 P-frame 单独计算梯度并更新参数
            # 优点：内存占用小，训练速度快
            # 缺点：可能忽略帧间依赖关系
            
            reference_frames = None
            
            for frame_pos in range(gop_size):
                current_frames = gop_batch[:, frame_pos, :, :, :].to(device)
                
                if frame_pos == 0:
                    # 第一帧：I-frame 压缩
                    with torch.no_grad():
                        i_frame_result, _ = compress_i_frame_with_padding(i_frame_model, current_frames)
                        reference_frames = i_frame_result['x_hat'].detach()
                else:
                    # P-frames：逐帧优化
                    optimizer.zero_grad()  # 每个 P-frame 前清空梯度

                    # 前向传播
                    result = model(reference_frames, current_frames, stage=stage)
                    loss = result["loss"]

                    # 检查损失异常
                    if torch.isnan(loss) or torch.isinf(loss):
                        raise RuntimeError("Loss is NaN or Inf, stopping training")

                    # 反向传播
                    if accelerator:
                        accelerator.backward(loss)
                    else:
                        loss.backward()
                    
                    # 梯度裁剪
                    if grad_clip_max_norm is not None:
                        if accelerator:
                            accelerator.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
                    
                    # 更新参数
                    optimizer.step()
                    
                    # 更新 EMA
                    if ema is not None:
                        ema.update(unwrapped_model.parameters())
                    
                    # 收集统计信息
                    total_loss += result["loss"].item()
                    total_mse += result["mse_loss"].item() * batch_size
                    total_bpp += result["bpp_train"].item() * batch_size
                    total_psnr += -10 * math.log10(result["mse_loss"].item()) * batch_size
                    n_frames += batch_size
                    
                    # 更新参考帧（分离梯度，避免影响下一帧的梯度计算）
                    with torch.no_grad():
                        reference_frames = result["recon_image"].detach()
        
        # Update progress bar
        if n_frames > 0 and (accelerator is None or accelerator.is_main_process):
            postfix_dict = {
                'loss': f"{total_loss / n_frames:.4f}",
                'mse': f"{total_mse / n_frames:.6f}",
                'bpp': f"{total_bpp / n_frames:.4f}",
                'psnr': f"{total_psnr / n_frames:.2f}"
            }
            progress_bar.set_postfix(postfix_dict)
    
    # Calculate epoch statistics
    if n_frames > 0:
        avg_loss = total_loss / n_frames
        avg_mse = total_mse / n_frames
        avg_psnr = total_psnr / n_frames
        avg_bpp = total_bpp / n_frames
    else:
        avg_loss = 0
        avg_mse = 0
        avg_psnr = 0
        avg_bpp = 0
    
    # 在 epoch 结束时再次更新量化参数
    unwrapped_model.bitEstimator_z._update_quantiles()
    unwrapped_model.bitEstimator_z_mv._update_quantiles()

    return {
        "loss": avg_loss,
        "mse": avg_mse,
        "psnr": avg_psnr,
        "bpp": avg_bpp,
    }


def load_checkpoint_for_stage(checkpoint_dir, stage, quality_index, lambda_value, device, logger, accelerator=None):
    """
    加载检查点：优先加载当前阶段的检查点，如果没有则加载上一阶段的
    
    检查点命名格式：
    model_dcvc_lambda_{lambda_value}_quality_{quality_index}_stage_{stage}_{suffix}.pth
    
    Args:
        checkpoint_dir: 检查点目录
        stage: 当前训练阶段（1-4）
        quality_index: 质量索引（用于命名）
        lambda_value: Lambda 值（用于命名）
        device: 加载设备
        logger: 日志记录器
        accelerator: Accelerate 对象
    
    Returns:
        tuple: (checkpoint_dict, is_resuming_current_stage)
            - checkpoint_dict: 检查点字典，如果未找到则为 None
            - is_resuming_current_stage: 是否为当前阶段的检查点（True）还是上一阶段的（False）
    """
    stage_checkpoint_patterns = [
        f'model_dcvc_lambda_{lambda_value}_quality_{quality_index}_stage_{stage}_best.pth',
        f'model_dcvc_lambda_{lambda_value}_quality_{quality_index}_stage_{stage}_latest.pth',
        f'model_dcvc_lambda_{lambda_value}_quality_{quality_index}_stage_{stage}.pth'
    ]
    
    # Try to load current stage checkpoint first
    for pattern in stage_checkpoint_patterns:
        checkpoint_path = os.path.join(checkpoint_dir, pattern)
        if os.path.exists(checkpoint_path):
            if accelerator is None or accelerator.is_main_process:
                logger.info(f"Loading checkpoint from current stage: {checkpoint_path}")
            return torch.load(checkpoint_path, map_location=device), True
    
    # Try to load from previous stage
    if stage > 1:
        prev_stage_patterns = [
            f'model_dcvc_lambda_{lambda_value}_quality_{quality_index}_stage_{stage-1}_best.pth',
            f'model_dcvc_lambda_{lambda_value}_quality_{quality_index}_stage_{stage-1}.pth'
        ]
        
        for pattern in prev_stage_patterns:
            checkpoint_path = os.path.join(checkpoint_dir, pattern)
            if os.path.exists(checkpoint_path):
                if accelerator is None or accelerator.is_main_process:
                    logger.info(f"Loading checkpoint from previous stage: {checkpoint_path}")
                return torch.load(checkpoint_path, map_location=device), False
    
    if accelerator is None or accelerator.is_main_process:
        logger.info("No checkpoint found, starting from scratch")
    return None, False


def setup_training_config(args):
    """
    根据训练阶段设置训练配置
    
    Args:
        args: 命令行参数对象
    
    Returns:
        dict: 包含训练配置的字典
            - epochs: 训练轮数
            - batch_size: 批次大小
            - gop_size: GOP 大小
            - phase_name: 阶段名称
            - use_gop_optimization: 是否使用 GOP 级优化（仅在 Stage 4 且启用 stage4_gop_opt 时）
            - eval_stage: 评估时使用的阶段（通常与训练阶段相同）
    """
    epochs = args.epochs
    batch_size = args.batch_size
    gop_size = args.gop_size
    phase_name = "Training"
    # 只有在 Stage 4 且明确启用时，才使用 GOP 级优化
    use_gop_optimization = (args.stage == 4 and args.stage4_gop_opt)
    eval_stage = args.stage
    
    return {
        'epochs': epochs,
        'batch_size': batch_size,
        'gop_size': gop_size,
        'phase_name': phase_name,
        'use_gop_optimization': use_gop_optimization,
        'eval_stage': eval_stage
    }


def setup_dataset(args, config):
    """
    设置训练数据集和数据加载器
    
    从 MP4 视频文件中读取 GOP 大小的帧序列用于训练。
    
    Args:
        args: 命令行参数对象
        config: 训练配置字典（包含 gop_size, batch_size 等）
    
    Returns:
        tuple: (dataset, data_loader)
            - dataset: VideoGOPDataset 实例
            - data_loader: DataLoader 实例
    """
    # 使用 ToTensor 转换（将 PIL Image 转换为 Tensor，并归一化到 [0, 1]）
    train_transform = transforms.ToTensor()
    
    # 检查训练视频目录
    if not args.train_video_dir:
        raise ValueError("--train_video_dir must be provided")
    
    # 创建视频 GOP 数据集
    dataset = VideoGOPDataset(
        video_dir=args.train_video_dir,      # 视频文件目录
        gop_size=config['gop_size'],         # GOP 大小（每个序列的帧数）
        transform=train_transform,            # 图像转换
        crop_size=args.crop_size             # 随机裁剪大小（用于数据增强）
    )
    
    if len(dataset) == 0:
        raise RuntimeError(f"No video files found in {args.train_video_dir}")
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],     # 批次大小
        shuffle=True,                         # 每个 epoch 随机打乱
        num_workers=args.num_workers,        # 数据加载的并行进程数
        pin_memory=True                      # 将数据固定到内存（加速 GPU 传输）
    )
    
    return dataset, data_loader


def main():
    parser = argparse.ArgumentParser(description='DCVC Training with Four Stages')
    
    # Video dataset arguments
    parser.add_argument('--train_video_dir', type=str, required=True,
                        help='Path to training video directory (MP4 files)')
    parser.add_argument('--val_video_dir', type=str, default=None,
                        help='Path to validation video directory (MP4 files)')
    parser.add_argument('--i_frame_model_name', type=str, default='cheng2020-anchor', 
                        help='CompressAI I-frame model name (e.g., cheng2020-anchor, bmshj2018-factorized, etc.)')
    parser.add_argument('--i_frame_quality', type=int, default=6, 
                        help='Quality level for CompressAI pretrained model (1-8, higher = better quality)')
    parser.add_argument('--i_frame_pretrained', action='store_true', default=True,
                        help='Use CompressAI pretrained weights for I-frame model')
    
    
    # Training arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_dcvc', help='Directory to save checkpoints')
    parser.add_argument('--lambda_value', type=float, required=True, help='Lambda value for rate-distortion trade-off')
    parser.add_argument('--quality_index', type=int, required=True, help='Quality index (1-4) for the model name')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3, 4], 
                        help='Training stage (1-4)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for this stage')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--force_learning_rate', type=float, default=None, help='Force Learning rate')
    parser.add_argument('--crop_size', type=int, default=256, help='Random crop size for training patches')
    parser.add_argument('--gop_size', type=int, default=7, help='Number of frames in a sequence for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--stage4_gop_opt', action='store_true', help='Whether to use GOP optimization in stage 4')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Resume and loading arguments
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--load_pretrained', type=str, default=None, help='Path to load pretrained weights only')
    
    # Evaluation arguments
    parser.add_argument('--eval_freq', type=int, default=1, help='Video evaluation frequency')
    parser.add_argument('--eval_max_frames', type=int, default=96, help='Maximum frames per video sequence')
    parser.add_argument('--skip_eval', action='store_true', help='Skip evaluation during training')
    
    # Learning rate scheduler
    parser.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='LR reduction factor')
    parser.add_argument('--scheduler_patience', type=int, default=3, help='Scheduler patience')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-6, help='Minimum learning rate')
    
    # EMA arguments
    parser.add_argument('--use_ema', action='store_true',default=True, help='Use Exponential Moving Average')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')
    parser.add_argument('--evaluate_both', action='store_true', default=True, help='Evaluate both normal and EMA weights')

    # Training options
    parser.add_argument('--grad_clip_max_norm', type=float, default=1.0, help='Gradient clipping max norm')
    parser.add_argument('--compile', action='store_true', help='Compile the model')
    
    # Accelerate arguments
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'],
                       help='Whether to use mixed precision training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of gradient accumulation steps')
    
    args = parser.parse_args()

    # ========== 初始化 Accelerate（分布式训练和混合精度）==========
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,              # 混合精度：'no', 'fp16', 'bf16'
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 梯度累积步数
    )
    
    device = accelerator.device  # 获取计算设备（CPU/GPU）
    
    # ========== 设置随机种子（确保可复现性）==========
    if args.seed is not None:
        set_seed(args.seed)           # Accelerate 的全局种子设置
        random.seed(args.seed)        # Python random 模块
        torch.manual_seed(args.seed)  # PyTorch 随机数生成器
    
    # ========== 创建检查点目录（仅主进程）==========
    if accelerator.is_main_process:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ========== 设置日志系统 ==========
    logger = setup_logging(args.checkpoint_dir, accelerator)
    
    # ========== 设置 MPS 回退（Mac GPU 兼容性）==========
    # 在日志初始化后、模型操作前设置
    device = setup_device_for_mps_fallback(device, logger if accelerator.is_main_process else None)
    
    # ========== 设置训练配置 ==========
    config = setup_training_config(args)
    
    # Stage descriptions
    stage_descriptions = {
        1: "Warm up MV generation part",
        2: "Train other modules",
        3: "Train with bit cost", 
        4: "End-to-end training"
    }
    
    if accelerator.is_main_process:
        logger.info("=" * 80)
        logger.info(f"DCVC TRAINING - STAGE {args.stage}: {stage_descriptions[args.stage]}")
        logger.info("=" * 80)
        logger.info(f"Lambda value: {args.lambda_value}")
        logger.info(f"Quality index: {args.quality_index}")
        logger.info(f"Device: {device}")
        logger.info(f"Num processes: {accelerator.num_processes}")
        logger.info(f"Mixed precision: {accelerator.mixed_precision}")
        logger.info(f"Epochs: {config['epochs']}")
        logger.info(f"Batch size: {config['batch_size']}")
        logger.info(f"GOP size: {config['gop_size']}")
        logger.info(f"GOP optimization: {config['use_gop_optimization']}")
        logger.info("=" * 80)

    # ========== 加载 CompressAI I-frame 模型 ==========
    # I-frame 模型用于压缩每个 GOP 的第一帧，使用预训练权重，不参与训练
    if accelerator.is_main_process:
        logger.info(f"Loading CompressAI I-frame model: {args.i_frame_model_name}")
        logger.info(f"Quality level: {args.i_frame_quality}")
        logger.info(f"Using pretrained weights: {args.i_frame_pretrained}")
    
    try:
        # 从 CompressAI 模型库加载预训练模型
        i_frame_model = compressai_models[args.i_frame_model_name](
            quality=args.i_frame_quality,        # 质量等级（1-8，越高质量越好）
            pretrained=args.i_frame_pretrained  # 是否使用预训练权重
        ).to(device)
        
        # 设置为评估模式（不训练，只推理）
        i_frame_model.eval()
        
        if accelerator.is_main_process:
            logger.info(f"Successfully loaded CompressAI model: {args.i_frame_model_name}")
            
    except KeyError:
        # 模型名称不存在
        available_models = list(compressai_models.keys())
        error_msg = f"Model '{args.i_frame_model_name}' not found in CompressAI zoo. Available models: {available_models}"
        if accelerator.is_main_process:
            logger.error(error_msg)
        raise ValueError(error_msg)
        
    except Exception as e:
        # 其他加载错误
        error_msg = f"Error loading CompressAI model: {e}"
        if accelerator.is_main_process:
            logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # 如果启用编译，使用 torch.compile 加速推理
    if args.compile:
        i_frame_model = torch.compile(i_frame_model)
        if accelerator.is_main_process:
            logger.info("I-frame model compiled for faster inference")

    # ========== 初始化 DCVC P-frame 模型 ==========
    # DCVC 模型用于压缩 P-frames，lambda 值控制率失真权衡
    model = DCVC_net(lmbda=args.lambda_value).to(device)
    
    # 如果启用编译，使用 torch.compile 加速训练
    if args.compile:
        model = torch.compile(model)
        if accelerator.is_main_process:
            logger.info("DCVC model compiled for faster training")

    # Check validation video availability
    val_video_available = args.val_video_dir and os.path.exists(args.val_video_dir) and not args.skip_eval
    if val_video_available and accelerator.is_main_process:
        logger.info(f"Video evaluation enabled. Will evaluate every {args.eval_freq} epochs.")

    # Setup dataset and data loader
    dataset, train_loader = setup_dataset(args, config)
    
    if accelerator.is_main_process:
        logger.info(f"Video dataset size: {len(dataset)} samples")
        logger.info(f"Training batches per epoch: {len(train_loader)}")

    # ========== 初始化优化器 ==========
    optimizer = configure_optimizers(model, args)

    # ========== 使用 Accelerate 准备模型、优化器和数据加载器 ==========
    # 这一步会处理分布式训练的设备分配、混合精度包装等
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    # ========== 初始化 EMA（指数移动平均）==========
    # EMA 维护模型参数的平滑版本，通常能提高模型性能
    ema = None
    if args.use_ema:
        if accelerator.is_main_process:
            logger.info(f"Using EMA with decay rate: {args.ema_decay}")
        unwrapped_model = accelerator.unwrap_model(model)
        # 创建 EMA 对象，decay 控制平滑程度（0.999 表示保留 99.9% 的旧值）
        ema = ExponentialMovingAverage(unwrapped_model.parameters(), decay=args.ema_decay)
        ema.to(device)
    
    # ========== 设置学习率调度器 ==========
    scheduler = None
    if args.use_scheduler:
        # ReduceLROnPlateau：当验证损失不再下降时降低学习率
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',                        # 监控指标越小越好
            factor=args.scheduler_factor,     # 学习率衰减因子（每次乘以这个值）
            patience=args.scheduler_patience, # 等待多少个 epoch 没有改善后降低 LR
            min_lr=args.scheduler_min_lr,    # 最小学习率
        )
        if accelerator.is_main_process:
            logger.info("ReduceLROnPlateau scheduler initialized")

    # ========== 初始化跟踪变量 ==========
    start_epoch = 0                    # 起始 epoch（用于恢复训练）
    best_loss = float('inf')           # 常规模型的最佳验证损失
    best_loss_ema = float('inf')       # EMA 模型的最佳验证损失
    global_best_loss = float('inf')    # 全局最佳损失（常规和 EMA 中的最佳）
    
    # ========== 处理检查点恢复 ==========
    # 优先级：显式 resume > 显式 load_pretrained > 自动查找
    # 检查是否有显式的恢复路径
    if args.resume:
        if accelerator.is_main_process:
            logger.info(f"Resuming training from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        best_loss_ema = checkpoint.get('best_loss_ema', float('inf'))
        global_best_loss = checkpoint.get('global_best_loss', float('inf'))

        if args.use_ema and 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])

        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 检查是否有显式的预训练权重路径（只加载模型权重，不恢复训练状态）
    elif args.load_pretrained:
        if accelerator.is_main_process:
            logger.info(f"Loading pretrained weights: {args.load_pretrained}")
        checkpoint = torch.load(args.load_pretrained, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(state_dict, strict=True)  # strict=True 要求完全匹配

    # 既没有 resume 也没有 load_pretrained - 从头开始训练
    else:
        if accelerator.is_main_process:
            logger.info(f"Starting stage {args.stage} training from scratch")

    # ========== 强制设置学习率（如果指定）==========
    # 用于在改变 GOP 大小等情况下重置学习率
    if args.force_learning_rate is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.force_learning_rate
            if accelerator.is_main_process:
                logger.info(f"Forced learning rate to {args.force_learning_rate:.6f}")

        # 强制学习率时，重置调度器（避免调度器状态与当前 LR 不一致）
        if scheduler is not None:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=args.scheduler_factor,
                patience=args.scheduler_patience,
                min_lr=args.scheduler_min_lr,
            )
            if accelerator.is_main_process:
                logger.info("Scheduler reset due to forced learning rate")

    # ==================== 主训练循环 ====================
    # 遍历每个 epoch
    for epoch in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        if accelerator.is_main_process:
            logger.info("")
            logger.info("=" * 100)
            logger.info(f"  {config['phase_name']} Stage {args.stage} - Epoch {epoch + 1}/{config['epochs']}")
            logger.info("=" * 100)
            logger.info(f"  Learning Rate: {current_lr:.6e}")
            logger.info("-" * 100)

        # Train one epoch
        train_stats = train_one_epoch(
            model, i_frame_model, train_loader, optimizer, device, 
            args.stage, epoch + 1, args.gradient_accumulation_steps, 
            config['use_gop_optimization'], args.grad_clip_max_norm, ema, accelerator,
            config['phase_name']
        )
        
        epoch_duration = time.time() - epoch_start_time

        if accelerator.is_main_process:
            logger.info("")
            logger.info("-" * 100)
            logger.info(f"  TRAINING RESULTS - Epoch {epoch + 1} (Duration: {epoch_duration:.2f}s)")
            logger.info("-" * 100)
            logger.info(f"  Train Loss:  {train_stats['loss']:.6f}")
            logger.info(f"  Train PSNR:  {train_stats['psnr']:.2f} dB")
            logger.info(f"  Train BPP:   {train_stats['bpp']:.6f}")
            logger.info(f"  Train MSE:   {train_stats['mse']:.6f}")

        # Evaluation
        video_results = None
        video_results_ema = None

        if (epoch + 1) % args.eval_freq == 0 and val_video_available:
            if accelerator.is_main_process:
                logger.info("")
                logger.info("=" * 100)
                logger.info(f"  VIDEO EVALUATION - Epoch {epoch + 1}")
                logger.info("=" * 100)

            # Video evaluation
            video_results = evaluate_video(
                model, i_frame_model, args.val_video_dir, device,
                config['eval_stage'], args.eval_max_frames, accelerator=accelerator
            )

            # EMA evaluation on video
            if args.use_ema and args.evaluate_both:
                unwrapped_model = accelerator.unwrap_model(model)
                ema.store(unwrapped_model.parameters())
                ema.copy_to(unwrapped_model.parameters())

                if accelerator.is_main_process:
                    logger.info("")
                    logger.info("-" * 100)
                    logger.info("  VIDEO EVALUATION (EMA)")
                    logger.info("-" * 100)

                video_results_ema = evaluate_video(
                    model, i_frame_model, args.val_video_dir, device,
                    config['eval_stage'], args.eval_max_frames, accelerator=accelerator
                )

                ema.restore(unwrapped_model.parameters())

            if video_results and accelerator.is_main_process:
                rd_loss = video_results['avg_bpp'] + args.lambda_value * video_results['avg_mse']
                logger.info("")
                logger.info("-" * 100)
                logger.info("  TEST RESULTS (Regular Model)")
                logger.info("-" * 100)
                logger.info(f"  Overall:")
                logger.info(f"    PSNR:     {video_results['avg_psnr']:.4f} dB")
                logger.info(f"    BPP:      {video_results['avg_bpp']:.6f}")
                logger.info(f"    MSE:      {video_results['avg_mse']:.6f}")
                logger.info(f"    RD Loss:  {rd_loss:.6f} (lambda={args.lambda_value})")
                logger.info(f"  Sequences: {video_results['num_sequences']}")
                logger.info("")
                logger.info(f"  I-frame:")
                logger.info(f"    PSNR:     {video_results['avg_i_frame_psnr']:.4f} dB")
                logger.info(f"    BPP:      {video_results['avg_i_frame_bpp']:.6f}")
                logger.info(f"    MSE:      {video_results['avg_i_frame_mse']:.6f}")
                logger.info("")
                logger.info(f"  P-frame (avg):")
                logger.info(f"    PSNR:     {video_results['avg_p_frame_psnr']:.4f} dB")
                logger.info(f"    BPP:      {video_results['avg_p_frame_bpp']:.6f}")
                logger.info(f"    MSE:      {video_results['avg_p_frame_mse']:.6f}")

                # Log first 3 P-frame details
                logger.info("")
                logger.info(f"  First 3 P-frames (detailed):")
                detailed_results = video_results.get('detailed_results', {})
                for p_idx in range(1, 4):
                    bpp_values = []
                    psnr_values = []
                    mse_values = []
                    for seq_name, seq_results in detailed_results.items():
                        for p_frame_info in seq_results['first_three_p_frames']:
                            if p_frame_info['frame_idx'] == p_idx:
                                bpp_values.append(p_frame_info['bpp'])
                                psnr_values.append(p_frame_info['psnr'])
                                mse_values.append(p_frame_info['mse'])

                    if bpp_values:
                        avg_bpp = sum(bpp_values) / len(bpp_values)
                        avg_psnr = sum(psnr_values) / len(psnr_values)
                        avg_mse = sum(mse_values) / len(mse_values)
                        logger.info(f"    P-frame {p_idx}: BPP={avg_bpp:.6f}, PSNR={avg_psnr:.2f} dB, MSE={avg_mse:.6f}")

                if video_results_ema:
                    rd_loss_ema = video_results_ema['avg_bpp'] + args.lambda_value * video_results_ema['avg_mse']
                    logger.info("")
                    logger.info("-" * 100)
                    logger.info("  TEST RESULTS (EMA Model)")
                    logger.info("-" * 100)
                    logger.info(f"  Overall:")
                    logger.info(f"    PSNR:     {video_results_ema['avg_psnr']:.4f} dB")
                    logger.info(f"    BPP:      {video_results_ema['avg_bpp']:.6f}")
                    logger.info(f"    MSE:      {video_results_ema['avg_mse']:.6f}")
                    logger.info(f"    RD Loss:  {rd_loss_ema:.6f} (lambda={args.lambda_value})")
                    logger.info(f"  Sequences: {video_results_ema['num_sequences']}")
                    logger.info("")
                    logger.info(f"  I-frame:")
                    logger.info(f"    PSNR:     {video_results_ema['avg_i_frame_psnr']:.4f} dB")
                    logger.info(f"    BPP:      {video_results_ema['avg_i_frame_bpp']:.6f}")
                    logger.info(f"    MSE:      {video_results_ema['avg_i_frame_mse']:.6f}")
                    logger.info("")
                    logger.info(f"  P-frame (avg):")
                    logger.info(f"    PSNR:     {video_results_ema['avg_p_frame_psnr']:.4f} dB")
                    logger.info(f"    BPP:      {video_results_ema['avg_p_frame_bpp']:.6f}")
                    logger.info(f"    MSE:      {video_results_ema['avg_p_frame_mse']:.6f}")

                    # Log first 3 P-frame details for EMA
                    logger.info("")
                    logger.info(f"  First 3 P-frames (detailed):")
                    detailed_results_ema = video_results_ema.get('detailed_results', {})
                    for p_idx in range(1, 4):
                        bpp_values = []
                        psnr_values = []
                        mse_values = []
                        for seq_name, seq_results in detailed_results_ema.items():
                            for p_frame_info in seq_results['first_three_p_frames']:
                                if p_frame_info['frame_idx'] == p_idx:
                                    bpp_values.append(p_frame_info['bpp'])
                                    psnr_values.append(p_frame_info['psnr'])
                                    mse_values.append(p_frame_info['mse'])

                        if bpp_values:
                            avg_bpp = sum(bpp_values) / len(bpp_values)
                            avg_psnr = sum(psnr_values) / len(psnr_values)
                            avg_mse = sum(mse_values) / len(mse_values)
                            logger.info(f"    P-frame {p_idx}: BPP={avg_bpp:.6f}, PSNR={avg_psnr:.2f} dB, MSE={avg_mse:.6f}")

        # ========== 计算测试损失（用于模型选择和调度器）==========
        # 默认使用训练损失
        current_test_loss = train_stats['loss']
        current_test_loss_ema = None
        
        # 如果有验证结果，使用验证损失（RD Loss = BPP + lambda * MSE）
        if video_results:
            current_test_loss = video_results['avg_bpp'] + args.lambda_value * video_results['avg_mse']
        if video_results_ema:
            current_test_loss_ema = video_results_ema['avg_bpp'] + args.lambda_value * video_results_ema['avg_mse']

        # 更新学习率调度器（基于验证损失）
        if scheduler:
            scheduler.step(current_test_loss)

        # ========== 检查是否为最佳模型 ==========
        # 检查常规模型是否是最佳
        is_best = current_test_loss < best_loss
        is_best_ema = False
        if is_best:
            best_loss = current_test_loss
        
        # 检查 EMA 模型是否是最佳
        if args.use_ema and current_test_loss_ema is not None:
            if current_test_loss_ema < best_loss_ema:
                best_loss_ema = current_test_loss_ema
                is_best_ema = True

        # 检查全局最佳（常规模型和 EMA 模型中的最佳者）
        current_best_loss = min(current_test_loss, current_test_loss_ema or float('inf'))
        is_global_best = current_best_loss < global_best_loss
        # 判断全局最佳是否为 EMA 模型
        is_ema_global_best = (current_test_loss_ema is not None and 
                             current_test_loss_ema == current_best_loss and 
                             is_global_best)
        
        if is_global_best:
            global_best_loss = current_best_loss

        # ========== 保存检查点（仅在主进程）==========
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            # 构建检查点数据字典
            checkpoint_data = {
                'epoch': epoch,                                    # 当前 epoch
                'model_state_dict': unwrapped_model.state_dict(), # 模型权重
                'optimizer_state_dict': optimizer.state_dict(),    # 优化器状态
                'loss': train_stats['loss'],                       # 当前训练损失
                'best_loss': best_loss,                            # 最佳验证损失（常规模型）
                'global_best_loss': global_best_loss,             # 全局最佳损失
                'stage': args.stage,                               # 训练阶段
                'quality_index': args.quality_index,               # 质量索引
                'lambda_value': args.lambda_value                  # Lambda 值
            }
            
            # 如果使用 EMA，保存 EMA 状态
            if args.use_ema:
                checkpoint_data['ema_state_dict'] = ema.state_dict()
                checkpoint_data['best_loss_ema'] = best_loss_ema
            
            # 如果使用调度器，保存调度器状态
            if scheduler:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            
            # 保存最新检查点（每个 epoch 都保存）
            latest_path = os.path.join(
                args.checkpoint_dir,
                f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_latest.pth'
            )
            torch.save(checkpoint_data, latest_path)
            
            # 保存最佳检查点（常规模型）
            if is_best or is_global_best:
                best_path = os.path.join(
                    args.checkpoint_dir,
                    f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_best.pth'
                )
                torch.save(checkpoint_data, best_path)
                logger.info("")
                logger.info("=" * 100)
                logger.info(f"  *** NEW BEST MODEL (Regular) - Epoch {epoch + 1} ***")
                logger.info("=" * 100)
                logger.info(f"  Best Loss:  {best_loss:.6f}")
                if video_results:
                    logger.info(f"  PSNR:       {video_results['avg_psnr']:.4f} dB")
                    logger.info(f"  BPP:        {video_results['avg_bpp']:.6f}")
                logger.info(f"  Saved to:   {os.path.basename(best_path)}")
                logger.info("=" * 100)

            # 保存最佳 EMA 模型
            if args.use_ema and (is_best_ema or is_ema_global_best):
                # 临时将 EMA 权重复制到模型
                ema.store(unwrapped_model.parameters())      # 保存当前模型权重
                ema.copy_to(unwrapped_model.parameters())     # 将 EMA 权重复制到模型

                # 保存 EMA 检查点
                ema_checkpoint_data = checkpoint_data.copy()
                ema_checkpoint_data['model_state_dict'] = unwrapped_model.state_dict()

                ema_path = os.path.join(
                    args.checkpoint_dir,
                    f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_best_ema.pth'
                )
                torch.save(ema_checkpoint_data, ema_path)

                # 恢复原始模型权重
                ema.restore(unwrapped_model.parameters())
                logger.info("")
                logger.info("=" * 100)
                logger.info(f"  *** NEW BEST MODEL (EMA) - Epoch {epoch + 1} ***")
                logger.info("=" * 100)
                logger.info(f"  Best Loss:  {best_loss_ema:.6f}")
                if video_results_ema:
                    logger.info(f"  PSNR:       {video_results_ema['avg_psnr']:.4f} dB")
                    logger.info(f"  BPP:        {video_results_ema['avg_bpp']:.6f}")
                logger.info(f"  Saved to:   {os.path.basename(ema_path)}")
                logger.info("=" * 100)

            # 保存全局最佳模型（常规模型和 EMA 模型中的最佳者）
            if is_global_best:
                # 如果是 EMA 模型最佳，临时复制 EMA 权重
                if is_ema_global_best:
                    ema.store(unwrapped_model.parameters())
                    ema.copy_to(unwrapped_model.parameters())

                # 保存完整检查点
                global_checkpoint_data = checkpoint_data.copy()
                global_checkpoint_data['model_state_dict'] = unwrapped_model.state_dict()
                global_checkpoint_data['is_ema'] = is_ema_global_best  # 标记是否为 EMA 模型

                global_path = os.path.join(
                    args.checkpoint_dir,
                    f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_global_best.pth'
                )
                torch.save(global_checkpoint_data, global_path)

                # 单独保存 state_dict（便于直接加载模型权重）
                global_state_dict_path = os.path.join(
                    args.checkpoint_dir,
                    f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_global_best_state_dict.pth'
                )
                torch.save(unwrapped_model.state_dict(), global_state_dict_path)

                # 如果是 EMA 模型，恢复原始权重
                if is_ema_global_best:
                    ema.restore(unwrapped_model.parameters())

                model_type = "EMA" if is_ema_global_best else "Regular"
                logger.info("")
                logger.info("*" * 100)
                logger.info(f"  *** NEW GLOBAL BEST MODEL ({model_type}) - Epoch {epoch + 1} ***")
                logger.info("*" * 100)
                logger.info(f"  Global Best Loss:  {global_best_loss:.6f}")
                if is_ema_global_best and video_results_ema:
                    logger.info(f"  PSNR:              {video_results_ema['avg_psnr']:.4f} dB")
                    logger.info(f"  BPP:               {video_results_ema['avg_bpp']:.6f}")
                elif video_results:
                    logger.info(f"  PSNR:              {video_results['avg_psnr']:.4f} dB")
                    logger.info(f"  BPP:               {video_results['avg_bpp']:.6f}")
                logger.info(f"  Saved to:          {os.path.basename(global_path)}")
                logger.info("*" * 100)

        # End of epoch separator
        if accelerator.is_main_process:
            logger.info("")
            logger.info("=" * 100)
            logger.info(f"  End of Epoch {epoch + 1}/{config['epochs']}")
            logger.info("=" * 100)
            logger.info("")

    # ========== 保存最终模型 ==========
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        
        # 使用阶段特定的命名保存
        final_path = os.path.join(
            args.checkpoint_dir,
            f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_final.pth'
        )
        torch.save(unwrapped_model.state_dict(), final_path)
        logger.info(f"Final model for stage {args.stage} saved to {final_path}")
        
        # 如果是 Stage 4，使用标准命名保存（便于后续使用）
        if args.stage == 4:
            standard_path = os.path.join(
                args.checkpoint_dir,
                f'model_dcvc_quality_{args.quality_index}_psnr.pth'
            )
            torch.save(unwrapped_model.state_dict(), standard_path)
            logger.info(f"Final model with standard naming saved to {standard_path}")

    # 等待所有进程完成（分布式训练同步点）
    accelerator.wait_for_everyone()

    # 清理 Accelerate 资源
    accelerator.end_training()

    if accelerator.is_main_process:
        logger.info("")
        logger.info("=" * 100)
        logger.info(f"  TRAINING COMPLETED - STAGE {args.stage} TRAINING")
        logger.info("=" * 100)
        logger.info(f"  Total Epochs:  {config['epochs']}")
        logger.info(f"  Stage:         {args.stage}")
        logger.info(f"  Lambda:        {args.lambda_value}")
        logger.info(f"  Quality Index: {args.quality_index}")
        logger.info("")
        logger.info("  BEST RESULTS:")
        logger.info("  " + "-" * 96)
        logger.info(f"  Best Loss (Regular):  {best_loss:.6f}")
        if args.use_ema:
            logger.info(f"  Best Loss (EMA):      {best_loss_ema:.6f}")
        logger.info(f"  Global Best Loss:     {global_best_loss:.6f}")
        logger.info("=" * 100)
        logger.info("")


if __name__ == '__main__':
    # 设置 float32 矩阵乘法的精度（'high' 表示高精度，可能稍慢但更准确）
    torch.set_float32_matmul_precision('high')
    main()
