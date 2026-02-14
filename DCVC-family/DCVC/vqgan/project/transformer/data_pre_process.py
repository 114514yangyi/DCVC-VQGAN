"""
数据预处理脚本

使用 VQGAN 预处理视频数据集，生成 codebook index 并保存为 .npy 文件
对于每个视频，生成的向量是 sequence_length * W * H
"""
import os.path
import sys

# 添加项目根目录到 Python 路径，以便可以直接运行脚本
# 注意：当前文件位于 <repo>/train/train_custom_videos.py，故只需回到上一级目录
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import os


import json
import argparse
from typing import List
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from models.model_adapter import create_model
from train.video_utils import VideoDataset, video_pipe, list_videos
from einops import rearrange


def process_videos(
    data_root: str,
    output_dir: str,
    vq_config: str,
    vq_ckpt: str,
    model_variant: str = "EMAVQ",
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    device: str = "cuda:0",
    sequence_length: int = 16
):
    """
    处理视频数据集，生成 VQ codebook index
    对于每个视频，生成的向量是 sequence_length * W * H
    
    Args:
        data_root: 原始视频数据集根目录（包含视频文件的目录）
        output_dir: 保存 codebook index 的输出目录
        vq_config: VQGAN 配置文件路径
        vq_ckpt: VQGAN 预训练 checkpoint 路径
        model_variant: VQGAN 模型变体
        image_size: 视频帧尺寸
        batch_size: 批次大小
        num_workers: DataLoader 线程数
        device: 设备字符串
        sequence_length: 每个视频处理的帧数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载 VQGAN 模型
    print(f"加载 VQGAN 模型: {model_variant}")
    print(f"配置文件: {vq_config}")
    print(f"Checkpoint: {vq_ckpt}")
    
    # 读取配置文件
    with open(vq_config, 'r') as f:
        config = json.load(f)
    data_args = config['data_args']
    model_args = config['model_args'] 
    # 准备模型参数
    # 如果配置文件中有 model_args，使用它；否则从配置中提取参数
    # ========== 创建模型 ==========
   
    # 确保 model_args 包含 sequence_length（从 data_args 获取）
    if 'sequence_length' not in model_args:
        model_args['sequence_length'] = data_args['sequence_length']
        
    elif model_args['sequence_length'] != data_args['sequence_length']:
        model_args['sequence_length'] = data_args['sequence_length']
    
    # 使用模型适配器创建模型（适配器会从配置文件读取模型类型）
    model = create_model(model_args=model_args, config_path=vq_config)
    model = model.to(device)
    
    # 手动加载 checkpoint（参考 trainVqVae.py）
    # 即使 TamingVQGANAdapter 在初始化时可能已经加载了，我们也手动加载以确保正确
    checkpoint = torch.load(vq_ckpt, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    
    # 收集所有视频文件
    video_files = list_videos(data_root)
    video_files.sort()
    
    print(f"找到 {len(video_files)} 个视频文件")
    print(f"序列长度: {sequence_length}")
    print(f"开始处理...")
    
    # 使用 video_pipe 创建数据加载器
    dataloader_wrapper = video_pipe(
        filenames=video_files,
        config_path=vq_config,
        batch_size=batch_size,
        num_threads=num_workers,
        device_id=0,
        sequence_length=sequence_length,
        shard_id=0,
        num_shards=1,
        initial_prefetch_size=1024,
        seed=None,
        random_shuffle=False,  # 按顺序处理
        target_size=(image_size, image_size)
    )
    dataloader = dataloader_wrapper.loader

    start_global_step = checkpoint.get('steps', 0)
    print(f"start_global_step: {start_global_step}")
    
    # 处理每个批次
    with torch.no_grad():
        for batch_idx, (videos, video_paths) in enumerate(tqdm(dataloader)):
            # videos 的形状: (batch_size, sequence_length, height, width, channels)
            # 转换为模型输入格式: (batch*sequence, channels, height, width)
            b, d, h, w, c = videos.shape
            videos = videos.float().to(device)
            
            # 重新排列维度: (batch, sequence, height, width, channels) -> (batch*sequence, channels, height, width)
            videos = rearrange(videos, 'b d h w c -> (b d) c h w')
            
            # 归一化: 将像素值从 [0, 255] 缩放到 [0, 1]，然后应用标准化
            videos = videos / 255.0
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            videos = normalize(videos)
            
            # 获取 codebook indices
            # 模型编码后返回的 indices 形状: (batch*sequence, h, w)
            indices = model.encode(videos)
            if indices is None:
                raise ValueError("无法从模型获取 codebook indices")
            
            # 确保 video_paths 是列表
            if isinstance(video_paths, torch.Tensor):
                video_paths = [str(p) for p in video_paths]
            elif not isinstance(video_paths, list):
                video_paths = list(video_paths)
            
            # 将 indices 重新排列回 (batch, sequence, h, w)
            # 假设 indices 的形状是 (batch*sequence, h, w)
            indices_reshaped = indices.view(b, d, indices.shape[1], indices.shape[2])
            
            # 对 batch 中的每个视频分别处理
            for i in range(b):
                # 获取第 i 个视频的 indices: (sequence_length, h, w)
                video_indices = indices_reshaped[i]  # (sequence_length, h, w)
                
                # 展平为 1D 数组: sequence_length * h * w
                idx_flat = video_indices.flatten()
                
                # 转换为 numpy
                idx_np = idx_flat.cpu().numpy()
                
                # 获取对应的视频路径
                video_path = video_paths[i]
                
                # 生成输出文件名（保持相对路径结构）
                try:
                    rel_path = os.path.relpath(video_path, data_root)
                except ValueError:
                    # 如果 video_path 不在 data_root 下，使用文件名
                    rel_path = os.path.basename(video_path)
                
                rel_path_no_ext = os.path.splitext(rel_path)[0]
                output_path = os.path.join(output_dir, str(batch_idx) + '.npy')
                
                # 创建输出目录（如果路径有目录部分）
                output_dir_path = os.path.dirname(output_path)
                if output_dir_path:
                    os.makedirs(output_dir_path, exist_ok=True)
                
                # 保存: 形状为 (sequence_length * h * w,)
                print(f"视频 {i+1}/{b}: {os.path.basename(video_path)} -> indices shape: {idx_np.shape}")
                np.save(output_path, idx_np)
    
    print(f"\n处理完成！")
    print(f"输出目录: {output_dir}")
    print(f"共处理 {len(video_files)} 个视频")


def main():
    parser = argparse.ArgumentParser(description="使用 VQGAN 预处理视频数据集，生成 codebook index")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/huyang/data/test",
        help="原始视频数据集根目录（包含视频文件的目录）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/huyang/data/vaild_indices",
        help="保存 codebook index 的输出目录",
    )
    parser.add_argument(
        "--vq_config",
        type=str,
        default="/home/huyang/VqVaeVideo-master/VqVaeVideo-master/multitask/taming_32.json",
        help="VQGAN 配置文件路径（如 multitask/taming.json）",
    )
    parser.add_argument(
        "--vq_ckpt",
        type=str,
        default="/data/huyang/save_data_taming_32/checkpoint_epoch1.pth.tar",
        help="VQGAN 预训练 checkpoint 路径",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="EMAVQ",
        help="VQGAN 模型变体，例如 EMAVQ 或 VQModel",
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="视频帧尺寸（resize & center crop）"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=16, help="每个视频处理的帧数"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="编码批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 线程数")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:3",
        help="设备字符串，例如 'cuda', 'cuda:0', 'cpu'",
    )
    args = parser.parse_args()
    
    process_videos(
        data_root=args.data_root,
        output_dir=args.output_dir,
        vq_config=args.vq_config,
        vq_ckpt=args.vq_ckpt,
        model_variant=args.model_variant,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        sequence_length=args.sequence_length
    )


if __name__ == "__main__":
    main()

