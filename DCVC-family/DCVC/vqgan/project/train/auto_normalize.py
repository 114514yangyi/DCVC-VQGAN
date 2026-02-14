"""
自动计算数据集的均值和方差，用于数据标准化
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    # 如果没有 tqdm，使用简单的进度显示
    def tqdm(iterable, desc=None):
        if desc:
            print(desc)
        return iterable

# 添加项目根目录到 Python 路径
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)


def compute_dataset_statistics(
    video_files: List[str],
    max_videos: Optional[int] = None,
    max_frames_per_video: int = 100,
    target_size: Tuple[int, int] = (256, 256),
    sample_frames: bool = True,
    num_samples: int = 10
) -> Tuple[List[float], List[float]]:
    """
    计算数据集的均值和方差（按通道计算）
    
    Args:
        video_files: 视频文件路径列表
        max_videos: 最大处理的视频数量，None 表示处理所有视频
        max_frames_per_video: 每个视频最多处理的帧数
        target_size: 目标图像尺寸 (height, width)
        sample_frames: 是否对视频帧进行采样（True 表示均匀采样，False 表示连续读取）
        num_samples: 如果 sample_frames=True，每个视频采样的帧数
    
    Returns:
        (mean, std): 均值和标准差，每个都是长度为 3 的列表 [R, G, B]
    """
    if not video_files:
        logger.warning("视频文件列表为空，返回 ImageNet 默认值")
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    # 限制处理的视频数量
    if max_videos is not None:
        video_files = video_files[:max_videos]
    
    logger.info(f"开始计算数据集统计信息...")
    logger.info(f"视频文件数量: {len(video_files)}")
    logger.info(f"每个视频最多处理 {max_frames_per_video} 帧")
    logger.info(f"目标尺寸: {target_size}")
    
    # 用于累积统计的变量
    pixel_sum = np.zeros(3, dtype=np.float64)  # RGB 三个通道的像素值总和
    pixel_squared_sum = np.zeros(3, dtype=np.float64)  # RGB 三个通道的像素值平方和
    total_pixels = 0  # 总像素数
    
    processed_videos = 0
    failed_videos = 0
    
    # 遍历所有视频文件
    for video_path in tqdm(video_files, desc="处理视频"):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"无法打开视频: {video_path}")
                failed_videos += 1
                continue
            
            # 获取视频总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                # 如果无法获取总帧数，尝试读取几帧来估算
                frame_count = 0
                while frame_count < 100:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                total_frames = frame_count
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开头
            
            if total_frames == 0:
                logger.warning(f"视频没有有效帧: {video_path}")
                cap.release()
                failed_videos += 1
                continue
            
            # 确定要处理的帧索引
            if sample_frames and total_frames > num_samples:
                # 均匀采样
                frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
            else:
                # 连续读取，但限制数量
                frame_indices = list(range(min(total_frames, max_frames_per_video)))
            
            # 读取并处理帧
            frames_read = 0
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # 调整尺寸
                if frame.shape[:2] != target_size:
                    frame = cv2.resize(frame, (target_size[1], target_size[0]), 
                                     interpolation=cv2.INTER_LINEAR)
                
                # 确保是 RGB 格式（OpenCV 默认是 BGR）
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 转换为 float32 并归一化到 [0, 1]
                frame = frame.astype(np.float32) / 255.0
                
                # 累积统计信息（按通道计算）
                # frame shape: (height, width, channels)
                pixel_sum += frame.sum(axis=(0, 1))  # 对 height 和 width 维度求和
                pixel_squared_sum += (frame ** 2).sum(axis=(0, 1))
                total_pixels += frame.shape[0] * frame.shape[1]  # height * width
                
                frames_read += 1
                if frames_read >= max_frames_per_video:
                    break
            
            cap.release()
            processed_videos += 1
            
        except Exception as e:
            logger.warning(f"处理视频时出错 {video_path}: {e}")
            failed_videos += 1
            continue
    
    if total_pixels == 0:
        logger.error("没有成功处理任何像素，返回 ImageNet 默认值")
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    # 计算均值和方差
    mean = pixel_sum / total_pixels
    # 方差 = E[X^2] - (E[X])^2
    variance = (pixel_squared_sum / total_pixels) - (mean ** 2)
    std = np.sqrt(variance)
    
    # 转换为列表
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    logger.info(f"统计计算完成:")
    logger.info(f"  成功处理视频: {processed_videos}")
    logger.info(f"  失败视频: {failed_videos}")
    logger.info(f"  总像素数: {total_pixels:,}")
    logger.info(f"  均值 (RGB): [{mean_list[0]:.6f}, {mean_list[1]:.6f}, {mean_list[2]:.6f}]")
    logger.info(f"  标准差 (RGB): [{std_list[0]:.6f}, {std_list[1]:.6f}, {std_list[2]:.6f}]")
    
    return mean_list, std_list


def compute_normalize_params(
    training_files: List[str],
    validation_files: Optional[List[str]] = None,
    max_videos: Optional[int] = None,
    max_frames_per_video: int = 100,
    target_size: Tuple[int, int] = (256, 256),
    sample_frames: bool = True,
    num_samples: int = 10
) -> Tuple[List[float], List[float]]:
    """
    计算训练集和验证集的均值和方差
    
    Args:
        training_files: 训练集视频文件路径列表
        validation_files: 验证集视频文件路径列表（可选）
        max_videos: 最大处理的视频数量，None 表示处理所有视频
        max_frames_per_video: 每个视频最多处理的帧数
        target_size: 目标图像尺寸 (height, width)
        sample_frames: 是否对视频帧进行采样
        num_samples: 如果 sample_frames=True，每个视频采样的帧数
    
    Returns:
        (mean, std): 均值和标准差，每个都是长度为 3 的列表 [R, G, B]
    """
    all_files = training_files.copy()
    if validation_files:
        all_files.extend(validation_files)
    
    return compute_dataset_statistics(
        all_files,
        max_videos=max_videos,
        max_frames_per_video=max_frames_per_video,
        target_size=target_size,
        sample_frames=sample_frames,
        num_samples=num_samples
    )


