"""
数据预处理脚本

使用 VQGAN 预处理图片数据集，生成 codebook index 并保存为 .npy 文件
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
from train.video_utils import ImageDataset


def process_images(
    data_root: str,
    output_dir: str,
    vq_config: str,
    vq_ckpt: str,
    model_variant: str = "EMAVQ",
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    device: str = "cuda:0"
):
    """
    处理图片数据集，生成 VQ codebook index
    
    Args:
        data_root: 原始图片数据集根目录
        output_dir: 保存 codebook index 的输出目录
        vq_config: VQGAN 配置文件路径
        vq_ckpt: VQGAN 预训练 checkpoint 路径
        model_variant: VQGAN 模型变体
        image_size: 图片尺寸
        batch_size: 批次大小
        num_workers: DataLoader 线程数
        device: 设备字符串
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
    
    # 收集所有图片文件（递归遍历）
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if any(file.endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    image_files.sort()
    
    # 使用 ImageDataset（设置 sequence_length=1 和 random_shuffle=False 以处理单张图片）
    dataset = ImageDataset(
        filenames=image_files,
        sequence_length=1,  # 每张图片单独处理
        random_shuffle=False,  # 按顺序处理
        seed=None,
        target_size=(image_size, image_size)
    )
    
    # 自定义 collate_fn 来处理 ImageDataset 的输出格式
    def collate_fn(batch):
        # batch 是一个列表，每个元素是 (video, img_path) 的元组
        # video 的形状是 (sequence_length, height, width, channels)，这里是 (1, h, w, c)
        videos = [item[0] for item in batch]  # 每个是 (1, h, w, c) 的 numpy 数组
        paths = [item[1] for item in batch]
        
        # 堆叠成 (batch_size, sequence_length, height, width, channels)
        batch_array = np.stack(videos, axis=0)  # (batch_size, 1, h, w, c)
        
        # 移除 sequence_length 维度: (batch_size, 1, h, w, c) -> (batch_size, h, w, c)
        batch_array = batch_array.squeeze(1)  # (batch_size, h, w, c)
        
        # 转换为 tensor: (batch_size, h, w, c) -> (batch_size, c, h, w)
        # 先转换为 tensor，然后转换为 float 并归一化
        batch_tensor = torch.from_numpy(batch_array)  # (batch_size, h, w, c), uint8
        batch_tensor = batch_tensor.permute(0, 3, 1, 2)  # (batch_size, c, h, w)
        batch_tensor = batch_tensor.float() / 255.0  # 归一化到 [0, 1]
        
        # 应用 ImageNet 标准化（与训练时保持一致）
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        batch_tensor = normalize(batch_tensor)
        
        return batch_tensor, paths
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"找到 {len(image_files)} 张图片")
    print(f"开始处理...")

    start_global_step = checkpoint['steps']
    print(f"start_global_step: {start_global_step}")
    
    # 处理每个批次
    with torch.no_grad():
        for batch_idx, (images, img_paths) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            # print(images)
            # print(images.shape)
            
            # 获取 codebook indices
            # TamingVQGANAdapter 的 forward 返回 (vq_loss, images_recon, perplexity, encoding_indices)
            indices=model.encode(images)
            # print(indices)
            # print(indices.shape)
            if indices is None:
                raise ValueError("无法从模型获取 codebook indices")
            
            # indices 的形状可能是 [batch, h, w] 或 [batch, seq_len]
            # 确保 img_paths 是列表（DataLoader 可能返回 tensor）
            if isinstance(img_paths, torch.Tensor):
                img_paths = [str(p) for p in img_paths]
            elif not isinstance(img_paths, list):
                img_paths = list(img_paths)
            
            batch_size = images.shape[0]
            
            # 对 batch 中的每个样本分别处理
            for i in range(batch_size):
                # 获取第 i 个样本的 indices
                idx = indices[i]  # [h, w] 或 [seq_len]
                
                # 展平为 1D 数组（无论原始形状如何）
                idx = idx.flatten()
                
                # 转换为 numpy
                idx_np = idx.cpu().numpy()
                
                # 获取对应的图片路径
                img_path = img_paths[i]
                
                # 生成输出文件名（保持相对路径结构）
                try:
                    rel_path = os.path.relpath(img_path, data_root)
                except ValueError:
                    # 如果 img_path 不在 data_root 下，使用文件名
                    rel_path = os.path.basename(img_path)
                
                rel_path_no_ext = os.path.splitext(rel_path)[0]
                output_path = os.path.join(output_dir, rel_path_no_ext + '.npy')
                
                # 创建输出目录（如果路径有目录部分）
                output_dir_path = os.path.dirname(output_path)
                if output_dir_path:
                    os.makedirs(output_dir_path, exist_ok=True)
                
                # 保存
                print(idx_np.shape)
                np.save(output_path, idx_np)
    
    print(f"\n处理完成！")
    print(f"输出目录: {output_dir}")
    print(f"共处理 {len(dataset)} 张图片")


def main():
    parser = argparse.ArgumentParser(description="使用 VQGAN 预处理图片数据集，生成 codebook index")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/huyang/data/vaild",
        help="原始图片数据集根目录（递归遍历所有图片）",
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
        "--image_size", type=int, default=256, help="图片尺寸（resize & center crop）"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="编码批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 线程数")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:4",
        help="设备字符串，例如 'cuda', 'cuda:0', 'cpu'",
    )
    args = parser.parse_args()
    
    process_images(
        data_root=args.data_root,
        output_dir=args.output_dir,
        vq_config=args.vq_config,
        vq_ckpt=args.vq_ckpt,
        model_variant=args.model_variant,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device
    )


if __name__ == "__main__":
    main()

