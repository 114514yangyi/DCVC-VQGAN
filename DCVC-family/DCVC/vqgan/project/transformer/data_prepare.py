"""
数据准备模块

提供 data_injection 函数，用于从视频文件直接生成 VQ codebook index 数据
使用 VQGAN 模型对视频进行编码，然后按照 sequence_length 切分为训练样本
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional
import logging
import torch
import cv2
from einops import rearrange
from torchvision import transforms
from tqdm import tqdm

# 添加项目根目录到 Python 路径
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.model_adapter import create_model
from train.video_utils import list_videos

logger = logging.getLogger(__name__)


def load_vqgan_model(vq_config: str, vq_ckpt: str, device: str):
    """
    加载 VQGAN 模型
    
    Args:
        vq_config: VQGAN 配置文件路径
        vq_ckpt: VQGAN checkpoint 路径
        device: 设备字符串
    
    Returns:
        model: VQGAN 模型
        config: 配置文件内容
    """
    import json
    
    with open(vq_config, 'r') as f:
        config = json.load(f)
    data_args = config['data_args']
    model_args = config['model_args']
    
    # 确保 model_args 包含 sequence_length
    if 'sequence_length' not in model_args:
        model_args['sequence_length'] = data_args['sequence_length']
    elif model_args['sequence_length'] != data_args['sequence_length']:
        model_args['sequence_length'] = data_args['sequence_length']
    
    # 创建模型
    model = create_model(model_args=model_args, config_path=vq_config)
    model = model.to(device)
    
    # 加载 checkpoint
    checkpoint = torch.load(vq_ckpt, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model, config


def encode_video_to_indices(vqgan_model, video: torch.Tensor, device: str, batch_size: int = 32) -> torch.Tensor:
    """
    使用 VQGAN 将视频编码为 codebook indices
    
    Args:
        vqgan_model: VQGAN 模型
        video: 视频张量 (num_frames, height, width, channels)
        device: 设备
        batch_size: 批处理大小（用于处理长视频）
    
    Returns:
        indices: codebook indices (num_frames, h, w)
    """
    # 转换为模型输入格式: (num_frames, channels, height, width)
    num_frames, h, w, c = video.shape
    video = video.float().to(device)
    video = rearrange(video, 'd h w c -> d c h w')
    
    # 归一化
    video = video / 255.0
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    video = normalize(video)
    
    # 分批编码（避免内存溢出）
    all_indices = []
    with torch.no_grad():
        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            batch_video = video[i:end_idx]
            batch_indices = vqgan_model.encode(batch_video)
            all_indices.append(batch_indices)
    
    # 合并所有批次的indices
    indices = torch.cat(all_indices, dim=0)
    
    return indices  # (num_frames, h, w)


def load_video_frames(video_path: str, image_size: int = 256) -> np.ndarray:
    """
    加载视频的所有帧（与 recon_judge.py 和 data_pre_process.py 保持一致）
    
    Args:
        video_path: 视频文件路径
        image_size: 目标图像尺寸
    
    Returns:
        video_np: 视频数组 (num_frames, height, width, channels), uint8
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    frames = []
    
    # 读取所有帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换 BGR 到 RGB（OpenCV 默认读取为 BGR）
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 调整大小到目标分辨率
        if frame.shape[:2] != (image_size, image_size):
            frame = cv2.resize(frame, (image_size, image_size), 
                             interpolation=cv2.INTER_LINEAR)
        
        frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"无法读取视频或视频为空: {video_path}")
    
    # 转换为 numpy 数组: (num_frames, height, width, channels)
    # 确保是 uint8 类型
    video_np = np.stack(frames, axis=0).astype(np.uint8)
    
    return video_np


def process_videos_to_indices(
    video_dir: str,
    vq_config: str,
    vq_ckpt: str,
    sequence_length: int,
    image_size: int = 256,
    device: str = "cuda:4",
    num_samples: Optional[int] = None,
    max_size=10000
) -> np.ndarray:
    """
    从视频目录加载视频，使用 VQGAN 编码，然后按照 sequence_length 切分为训练样本
    
    Args:
        video_dir: 包含视频文件的目录路径
        vq_config: VQGAN 配置文件路径
        vq_ckpt: VQGAN checkpoint 路径
        sequence_length: 每个训练样本包含的帧数
        image_size: 图像尺寸
        device: 设备字符串
        num_samples: 可选，限制生成的样本数量
    
    Returns:
        所有训练样本数组，形状为 [num_samples, seq_len]
        其中 seq_len = sequence_length * h * w
    """
    # 加载 VQGAN 模型
    logger.info(f"加载 VQGAN 模型...")
    vqgan_model, config = load_vqgan_model(vq_config, vq_ckpt, device)
    
    # 获取模型参数
    if 'model_args' in config:
        vocab_size = config['model_args'].get('vocab_size', 1024)
    else:
        vocab_size = 1024
    
    # 列出所有视频文件
    video_files = list_videos(video_dir)
    video_files.sort()
    
    if len(video_files) == 0:
        raise ValueError(f"在 {video_dir} 中未找到视频文件")
    
    logger.info(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频，生成训练样本
    all_samples = []
    device_torch = torch.device(device)
    
    for video_idx, video_path in tqdm(enumerate(video_files)):
        try:

            if len(all_samples) >= max_size:
                logger.info(f"已达到最大样本数量: {max_size}")
                break

            # 加载视频帧
            video_np = load_video_frames(video_path, image_size)
            video_tensor = torch.from_numpy(video_np)  # (num_frames, h, w, c)
            
            # 使用 VQGAN 编码得到 indices
            indices = encode_video_to_indices(vqgan_model, video_tensor, device_torch)
            # indices: (num_frames, h, w)
            
            # 清理视频数据（释放内存）
            del video_np, video_tensor
            if device_torch.type == 'cuda':
                torch.cuda.empty_cache()
            
            num_frames, indices_h, indices_w = indices.shape
            indices_per_frame = indices_h * indices_w
            
            # 按照 sequence_length 切分，每个切分段作为一个训练样本
            # 只处理完整的段，不够的段直接舍弃
            num_segments = num_frames // sequence_length
            
            for i in range(num_segments):
                start_frame = i * sequence_length
                end_frame = start_frame + sequence_length
                
                # 获取一个完整段: (sequence_length, h, w)
                segment = indices[start_frame:end_frame]
                
                # 展平为 1D（与 data_pre_process.py 一致）
                segment_flat = segment.flatten()  # (sequence_length * h * w,)
                
                # 转换为 numpy
                segment_np = segment_flat.cpu().numpy().astype(np.int64)
                
                all_samples.append(segment_np)
                
                # 如果达到样本数量限制，提前退出
                if num_samples is not None and len(all_samples) >= num_samples:
                    logger.info(f"已达到样本数量限制: {num_samples}")
                    break
            
            # 清理 indices（释放 GPU 内存）
            del indices
            if device_torch.type == 'cuda':
                torch.cuda.empty_cache()
            
            if num_samples is not None and len(all_samples) >= num_samples:
                break
                
        except Exception as e:
            logger.warning(f"处理视频失败 {video_path}: {e}")
            # 清理可能的临时变量
            if 'video_np' in locals():
                del video_np
            if 'video_tensor' in locals():
                del video_tensor
            if 'indices' in locals():
                del indices
            if device_torch.type == 'cuda':
                torch.cuda.empty_cache()
            continue
    
    if len(all_samples) == 0:
        raise ValueError(f"未能成功生成任何训练样本")
    
    # 转换为 numpy 数组
    data = np.array(all_samples, dtype=np.int64)
    
    logger.info(f"从 {len(video_files)} 个视频生成了 {len(data)} 个训练样本")
    logger.info(f"每个样本长度: {data.shape[1]}")
    
    # ========== 清理模型和临时数据 ==========
    logger.info("清理 VQGAN 模型和临时数据...")
    
    # 删除模型
    del vqgan_model
    vqgan_model = None
    
    # 清理 GPU 缓存（如果使用 GPU）
    if device_torch.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("已清理 GPU 缓存")
    
    # 清理临时变量（保留 data，因为需要返回）
    del all_samples
    del config
    del vocab_size
    
    logger.info("模型和临时数据清理完成")
    
    return data


def data_injection(
    data_dir: str,
    vq_config: str = None,
    vq_ckpt: str = None,
    sequence_length: int = 4,
    image_size: int = 256,
    device: str = "cuda:4",
    num_samples: Optional[int] = None,
    seq_len: Optional[int] = None,
) -> np.ndarray:
    """
    从视频目录加载视频，使用 VQGAN 编码生成训练数据
    
    Args:
        data_dir: 包含视频文件的目录路径
        vq_config: VQGAN 配置文件路径（如果为 None，则尝试从环境变量或默认路径获取）
        vq_ckpt: VQGAN checkpoint 路径（如果为 None，则尝试从环境变量或默认路径获取）
        sequence_length: 每个训练样本包含的帧数（如果为 None，将从 seq_len 反推，但需要知道 h 和 w）
        image_size: 图像尺寸
        device: 设备字符串
        num_samples: 可选，限制生成的样本数量
        seq_len: 可选，统一序列长度（用于验证，实际由 sequence_length * h * w 决定）
    
    Returns:
        所有训练样本数组，形状为 [num_samples, seq_len]
    """
    # 如果没有提供 vq_config 和 vq_ckpt，尝试使用默认值
    if vq_config is None:
        vq_config = os.getenv('VQ_CONFIG', "/home/huyang/VqVaeVideo-master/VqVaeVideo-master/multitask/taming.json")
    if vq_ckpt is None:
        vq_ckpt = os.getenv('VQ_CKPT', "/data/huyang/save_data_taming/checkpoint_epoch6.pth.tar")
    
    # 如果 sequence_length 为 None，尝试从配置文件获取
    if sequence_length is None:
        try:
            import json
            with open(vq_config, 'r') as f:
                config = json.load(f)
            sequence_length = config.get('data_args', {}).get('sequence_length', 16)
            logger.info(f"从配置文件读取 sequence_length: {sequence_length}")
        except Exception as e:
            logger.warning(f"无法从配置文件读取 sequence_length: {e}，使用默认值 16")
            sequence_length = 16
    
    # 从视频生成训练数据
    data = process_videos_to_indices(
        video_dir=data_dir,
        vq_config=vq_config,
        vq_ckpt=vq_ckpt,
        sequence_length=sequence_length,
        image_size=image_size,
        device=device,
        num_samples=num_samples
    )
    
    # 如果提供了 seq_len，验证长度是否匹配
    if seq_len is not None:
        if data.shape[1] != seq_len:
            logger.warning(f"数据长度 {data.shape[1]} 与期望长度 {seq_len} 不匹配")
            # 可以选择截断或填充，但这里我们直接使用实际长度
    
    logger.info(f"从 {data_dir} 生成了 {len(data)} 个训练样本")
    
    return data


def get_data_injection(
    data_dir: str, 
    seq_len: Optional[int] = None,
    vq_config: str = None,
    vq_ckpt: str = None,
    sequence_length: int = 4,
    image_size: int = 256,
    device: str = "cuda:1"
) -> np.ndarray:
    """
    便捷函数，使用默认参数生成训练数据
    
    Args:
        data_dir: 包含视频文件的目录路径（原来是 .npy 文件目录，现在是视频文件目录）
        seq_len: 可选，统一序列长度（用于验证，实际由 sequence_length * h * w 决定）
        vq_config: VQGAN 配置文件路径（如果为 None，使用默认值）
        vq_ckpt: VQGAN checkpoint 路径（如果为 None，使用默认值）
        sequence_length: 每个训练样本包含的帧数
        image_size: 图像尺寸
        device: 设备字符串
    
    Returns:
        所有训练样本数组，形状为 [num_samples, seq_len]
    """
    # 从 seq_len 反推 sequence_length（如果提供了 seq_len 但没有提供 sequence_length）
    # 但这里我们优先使用传入的 sequence_length
    return data_injection(
        data_dir=data_dir,
        vq_config=vq_config,
        vq_ckpt=vq_ckpt,
        sequence_length=sequence_length,
        image_size=image_size,
        device=device,
        seq_len=seq_len
    )
