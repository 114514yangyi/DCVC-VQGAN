"""
VQ-VAE2 视频重建脚本

功能：
1. 加载训练好的 VQ-VAE2 模型
2. 读取输入视频
3. 使用模型进行编码-解码重建
4. 保存重建后的视频

使用方法：
    python compression_encoder/compression_vqvae2.py \\
        --input <视频路径> \\
        --checkpoint <模型checkpoint路径> \\
        --config <配置文件路径> \\
        --output <输出视频路径> \\
        [--device cuda:0] \\
        [--batch-size 1]
"""
import os
import sys
import json
import argparse
import math
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms import Normalize
from torchvision.io import write_video

# 将项目根目录加入 sys.path，便于直接运行该脚本
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.model_adapter import create_model  # noqa: E402
from train.train_utils import NormalizeInverse  # noqa: E402


def load_video_frames(path: str, target_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, float]:
    """
    读取视频并返回帧数组和 fps
    
    Args:
        path: 视频文件路径
        target_size: 目标分辨率 (height, width)
    
    Returns:
        Tuple[frames, fps]:
        - frames: numpy数组，形状为 (T, H, W, C)，数据类型为 uint8，通道顺序为 RGB
        - fps: 视频帧率
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # 默认帧率

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换 BGR 到 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 调整分辨率到目标尺寸
        if frame.shape[:2] != target_size:
            frame = cv2.resize(
                frame,
                (target_size[1], target_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"视频中未读取到任何帧: {path}")

    # 堆叠为 (T, H, W, C) 的 numpy 数组，uint8 类型
    video_np = np.stack(frames, axis=0).astype(np.uint8)
    return video_np, float(fps)


def save_video_frames(path: str, frames: torch.Tensor, fps: float):
    """
    将重建的帧保存为视频文件
    
    Args:
        path: 输出视频路径
        frames: torch.Tensor，形状为 (T, C, H, W)，值域 [0, 1]，通道顺序为 RGB
        fps: 视频帧率
    """
    # 确保值域在 [0, 1]
    frames = frames.clamp(0.0, 1.0)
    # 转换为 uint8
    frames = (frames * 255.0).to(torch.uint8)
    # 转换为 (T, H, W, C) 格式
    frames = frames.permute(0, 2, 3, 1).cpu()  # (T, H, W, C) RGB
    write_video(path, frames, fps=fps)


def reconstruct_video(
    input_video: str,
    checkpoint_path: str,
    config_path: str,
    output_video: str,
    device: str = "cuda:0",
    batch_size: int = 1,
    sequence_length: Optional[int] = None,
    target_size: Tuple[int, int] = (256, 256)
):
    """
    使用 VQ-VAE2 模型重建视频
    
    Args:
        input_video: 输入视频路径
        checkpoint_path: 模型 checkpoint 路径
        config_path: 配置文件路径（JSON格式）
        output_video: 输出视频路径
        device: 运行设备，如 "cuda:0" 或 "cpu"
        batch_size: 批处理大小（用于处理长视频）
        sequence_length: 序列长度（如果为None，从配置文件读取）
        target_size: 目标分辨率 (height, width)
    """
    print("=" * 80)
    print("VQ-VAE2 视频重建")
    print("=" * 80)
    print(f"输入视频: {input_video}")
    print(f"模型checkpoint: {checkpoint_path}")
    print(f"配置文件: {config_path}")
    print(f"输出视频: {output_video}")
    print(f"设备: {device}")
    print("=" * 80)
    
    # 1. 加载配置文件
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 2. 获取模型参数和配置
    model_args = config.get('model_args', {})
    data_args = config.get('data_args', {})
    
    # 如果未指定 sequence_length，从配置文件读取
    if sequence_length is None:
        sequence_length = data_args.get('sequence_length', 8)
    
    # 获取归一化参数（从配置文件或使用默认值）
    normalize_compute_args = data_args.get('normalize_compute_args', {})
    # 默认使用 [0.5, 0.5, 0.5] 和 [0.5, 0.5, 0.5]（对应 [-1, 1] 归一化）
    # 或者使用 ImageNet 标准值
    mean = normalize_compute_args.get('mean', [0.5, 0.5, 0.5])
    std = normalize_compute_args.get('std', [0.5, 0.5, 0.5])
    
    # 如果配置中没有，尝试从训练参数推断
    # 根据 train_custom_videos.py，默认使用 [0.5, 0.5, 0.5]
    if 'mean' not in normalize_compute_args:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    
    print(f"归一化参数: mean={mean}, std={std}")
    print(f"序列长度: {sequence_length}")
    
    # 3. 创建归一化和反归一化变换
    normalize = Normalize(mean=mean, std=std)
    unnormalize = NormalizeInverse(mean=mean, std=std)
    
    # 4. 创建模型
    print("\n正在加载模型...")
    model = create_model(model_args=model_args, config_path=config_path)
    model = model.to(device)
    model.eval()
    
    # 5. 加载 checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
    
    print(f"正在加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重（兼容不同的checkpoint格式）
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    print("模型加载完成")
    
    # 6. 读取输入视频
    print(f"\n正在读取视频: {input_video}")
    frames_np, fps = load_video_frames(input_video, target_size=target_size)
    T, H, W, C = frames_np.shape
    print(f"视频信息: {T} 帧, {H}x{W}, {C} 通道, {fps:.2f} fps")
    
    # 7. 转换为 tensor 并归一化
    # 转换为 float32 并归一化到 [0, 1]
    frames_tensor = torch.from_numpy(frames_np).float() / 255.0
    # 转换为 (T, C, H, W)
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)
    frames_tensor = frames_tensor.to(device)
    
    # 应用归一化
    frames_normalized = normalize(frames_tensor)  # (T, C, H, W)
    
    # 8. 将视频分割成多个序列进行处理
    # 计算需要多少个序列
    n_sequences = math.ceil(T / sequence_length)
    print(f"\n将视频分割为 {n_sequences} 个序列（每个序列 {sequence_length} 帧）")
    
    # 如果视频长度不是 sequence_length 的倍数，需要padding
    total_frames = n_sequences * sequence_length
    if total_frames > T:
        # 重复最后一帧进行padding
        pad_frames = frames_normalized[-1:].repeat(total_frames - T, 1, 1, 1)
        frames_normalized = torch.cat([frames_normalized, pad_frames], dim=0)
        print(f"添加 {total_frames - T} 帧padding（重复最后一帧）")
    
    # 9. 重建视频
    print("\n开始重建视频...")
    reconstructed_frames = []
    
    with torch.no_grad():
        for i in range(n_sequences):
            start_idx = i * sequence_length
            end_idx = (i + 1) * sequence_length
            
            # 提取当前序列: (sequence_length, C, H, W)
            sequence = frames_normalized[start_idx:end_idx]
            
            # 模型期望的输入格式: (batch*sequence, C, H, W)
            # 这里 sequence 已经是 (sequence_length, C, H, W)，符合要求
            # 因为模型内部会将输入视为 (batch*sequence, C, H, W)，其中 batch=1
            
            # 前向传播
            try:
                model_out = model(sequence)
            except Exception as e:
                print(f"错误: 在处理序列 {i+1}/{n_sequences} 时发生错误: {e}")
                raise
            
            # 解析模型输出
            # VQVAEAdapter 的 forward 返回 (vq_loss, images_recon, perplexity, encoding_indices)
            if isinstance(model_out, (list, tuple)) and len(model_out) >= 2:
                images_recon = model_out[1]  # 第二个元素是重建图像
            elif isinstance(model_out, (list, tuple)) and len(model_out) == 1:
                images_recon = model_out[0]
            else:
                # 如果格式不同，尝试直接使用
                images_recon = model_out
            
            # 确保 images_recon 的形状正确: (sequence_length, C, H, W)
            if images_recon.dim() == 4:
                # 形状应该是 (sequence_length, C, H, W)
                if images_recon.shape[0] != sequence_length:
                    print(f"警告: 重建图像数量 ({images_recon.shape[0]}) 与序列长度 ({sequence_length}) 不匹配")
                    # 如果数量不对，尝试截取或填充
                    if images_recon.shape[0] > sequence_length:
                        images_recon = images_recon[:sequence_length]
                    else:
                        # 如果数量不足，重复最后一帧
                        pad_frames = images_recon[-1:].repeat(sequence_length - images_recon.shape[0], 1, 1, 1)
                        images_recon = torch.cat([images_recon, pad_frames], dim=0)
            elif images_recon.dim() == 5:
                # 如果是 (batch, sequence_length, C, H, W)，需要reshape
                b, d, c, h, w = images_recon.shape
                images_recon = images_recon.view(b * d, c, h, w)
                # 只取前 sequence_length 帧
                images_recon = images_recon[:sequence_length]
            else:
                raise ValueError(f"意外的输出形状: {images_recon.shape}, 期望4维或5维")
            
            # 反归一化
            images_recon = unnormalize(images_recon)  # (sequence_length, C, H, W)
            
            # 限制值域到 [0, 1]
            images_recon = images_recon.clamp(0.0, 1.0)
            
            reconstructed_frames.append(images_recon.cpu())
            
            # 显示进度
            if (i + 1) % max(1, n_sequences // 10) == 0 or (i + 1) == n_sequences:
                print(f"  进度: {i + 1}/{n_sequences} ({100*(i+1)/n_sequences:.1f}%)")
    
    # 10. 合并所有重建的帧
    print("\n合并重建帧...")
    all_reconstructed = torch.cat(reconstructed_frames, dim=0)  # (total_frames, C, H, W)
    
    # 只保留原始帧数（去掉padding）
    all_reconstructed = all_reconstructed[:T]  # (T, C, H, W)
    
    # 11. 保存视频
    print(f"\n正在保存重建视频: {output_video}")
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    save_video_frames(output_video, all_reconstructed, fps=fps)
    
    print("=" * 80)
    print("重建完成！")
    print(f"输出视频: {output_video}")
    print(f"原始帧数: {T}")
    print(f"重建帧数: {all_reconstructed.shape[0]}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='使用 VQ-VAE2 模型重建视频',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python compression_encoder/compression_vqvae2.py \\
      --input data/video.mp4 \\
      --checkpoint checkpoints/checkpoint100000.pth.tar \\
      --config multitask/task5.json \\
      --output output/reconstructed.mp4

  # 指定设备
  python compression_encoder/compression_vqvae2.py \\
      --input data/video.mp4 \\
      --checkpoint checkpoints/checkpoint100000.pth.tar \\
      --config multitask/task5.json \\
      --output output/reconstructed.mp4 \\
      --device cuda:0

  # 使用CPU
  python compression_encoder/compression_vqvae2.py \\
      --input data/video.mp4 \\
      --checkpoint checkpoints/checkpoint100000.pth.tar \\
      --config multitask/task5.json \\
      --output output/reconstructed.mp4 \\
      --device cpu
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入视频路径'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型checkpoint路径（.pth.tar文件）'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='配置文件路径（JSON格式，如task5.json）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出视频路径'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='运行设备: "cuda:0", "cpu", 或 "auto"（自动检测，默认: auto）'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='批处理大小（默认: 1，当前版本暂未使用）'
    )
    
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=None,
        help='序列长度（默认: 从配置文件读取）'
    )
    
    parser.add_argument(
        '--target-size',
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=('HEIGHT', 'WIDTH'),
        help='目标分辨率（默认: 256 256）'
    )
    
    args = parser.parse_args()
    
    # 验证输入文件存在
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入视频不存在: {args.input}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint文件不存在: {args.checkpoint}")
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    # 处理设备选择
    device = args.device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda:0'
            print(f"自动检测到CUDA，使用设备: {device}")
        else:
            device = 'cpu'
            print(f"未检测到CUDA，使用CPU")
    elif device.startswith('cuda') and not torch.cuda.is_available():
        print(f"警告: 指定了CUDA设备但CUDA不可用，回退到CPU")
        device = 'cpu'
    
    # 执行重建
    reconstruct_video(
        input_video=args.input,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_video=args.output,
        device=device,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        target_size=tuple(args.target_size)
    )


if __name__ == '__main__':
    main()

