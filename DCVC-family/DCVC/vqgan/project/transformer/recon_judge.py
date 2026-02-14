#!/usr/bin/env python3
"""
视频压缩和重建评估脚本

使用 VQGAN 压缩视频，然后使用训练好的 Transformer 和算术编码进行进一步压缩，
最后重建视频并评估质量指标（BPP, bits/index, PSNR, FID等）
"""

import os.path
import sys

# 添加项目根目录到 Python 路径
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import os
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from einops import rearrange
from torchvision import transforms
from torchvision.io import write_video
import cv2
import pickle
import tempfile
import shutil
import csv
from typing import Dict, List, Tuple

from models.model_adapter import create_model
from train.video_utils import VideoDataset, video_pipe, list_videos
from train.train_utils import NormalizeInverse
from transformer.train_transformer import TransformerEntropyModel, VQIndexDataset
from transformer.encode_methods import (
    compress_arithmetic_coding, decompress_arithmetic_coding
)
from transformer.data_prepare import get_data_injection

# 尝试导入 FID 计算库
try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("⚠ 未安装 pytorch-fid，将跳过 FID 计算")
    print("  安装方法: pip install pytorch-fid")




import random

def set_deterministic(seed=42):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # 设置CUDA确定性选项
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁用自动优化
    
    # 设置环境变量
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

set_deterministic(42)


# 添加项目根目录到 Python 路径
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from transformer.train_transformer import TransformerEntropyModel
from transformer.encode_methods import (
    compress_arithmetic_coding,
    decompress_arithmetic_coding_incremental
)

def load_vqgan_model(vq_config: str, vq_ckpt: str, device: str):
    """加载 VQGAN 模型"""
    print(f"加载 VQGAN 模型...")
    print(f"配置文件: {vq_config}")
    print(f"Checkpoint: {vq_ckpt}")
    
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


def load_transformer_model(transformer_ckpt: str, vocab_size: int, seq_len: int, device: str):
    """加载训练好的 Transformer 模型"""
    print(f"加载 Transformer 模型...")
    print(f"Checkpoint: {transformer_ckpt}")
    
    checkpoint = torch.load(transformer_ckpt, map_location=device, weights_only=False)
    
    # 从 checkpoint 获取参数（如果存在）
    args = checkpoint.get('args', {})
    vocab_size = args.get('vocab_size', vocab_size)
    seq_len = args.get('seq_len', seq_len)
    d_model = args.get('d_model', 256)
    num_layers = args.get('num_layers', 8)
    
    # 创建模型
    model = TransformerEntropyModel(
        num_codes=vocab_size,  # 注意：参数名是 num_codes，不是 vocab_size
        d_model=d_model,
        num_layers=num_layers,
        max_seq_len=seq_len
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 打印模型信息
    print(f"Transformer 模型参数:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  seq_len (max_seq_len): {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  num_layers: {num_layers}")
    
    return model, vocab_size


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


def decode_indices_to_video(vqgan_model, indices: torch.Tensor, device: str, 
                           normalize: transforms.Normalize, 
                           unnormalize: NormalizeInverse, batch_size: int = 32) -> np.ndarray:
    """
    使用 VQGAN 将 codebook indices 解码为视频
    
    Args:
        vqgan_model: VQGAN 模型
        indices: codebook indices (num_frames, h, w)
        device: 设备
        normalize: 归一化变换
        unnormalize: 反归一化变换
        batch_size: 批处理大小（用于处理长视频）
    
    Returns:
        video: 重建的视频 (num_frames, height, width, channels), uint8
    """
    num_frames = indices.shape[0]
    indices = indices.to(device)
    
    # 分批解码（避免内存溢出）
    all_frames = []
    with torch.no_grad():
        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            batch_indices = indices[i:end_idx]
            
            # 解码
            decoded = vqgan_model.decode(batch_indices)
            
            # 反归一化
            decoded = unnormalize(decoded)
            
            # 转换为 numpy 并调整范围
            decoded = decoded.clamp(0, 1)
            decoded = (decoded * 255).cpu().numpy().astype(np.uint8)
            
            # 重新排列维度: (batch, channels, height, width) -> (batch, height, width, channels)
            decoded = rearrange(decoded, 'd c h w -> d h w c')
            
            all_frames.append(decoded)
    
    # 合并所有帧
    video = np.concatenate(all_frames, axis=0)
    
    return video

def compress_indices_with_transformer_slidewindow(transformer_model, indices: torch.Tensor, 
                                     device: str, sequence_length: int, h: int, w: int,
                                     video_path: str = None, enable_comparison_test: bool = False,
                                     test: bool = False) -> Tuple[bytes, Dict]:
    """
    使用 Transformer + 算术编码的「时间滑动窗口」方式压缩 codebook indices。

    核心区别于 compress_indices_with_transformer:
    - 仍然按 sequence_length 帧一段进行压缩；
    - 第 1 段：与原实现完全一致（BOS + 本段所有帧 -> 预测概率 -> 算术编码）；
    - 第 2 段及之后：
        * 取「上一段的最后 sequence_length-1 帧」作为时间上下文；
        * 将这些上下文帧的 indices 作为前缀，仅用来提供条件，不再重新编码；
        * 仅对当前段的 sequence_length 帧进行算术编码；
        * 等价于：在时间维度上做“内缩滑动窗口”预测。
    """
    num_frames, indices_h, indices_w = indices.shape
    indices = indices.to(device)

    # 将indices展平为 (num_frames, h*w)
    indices_per_frame = indices_h * indices_w
    indices_reshaped = indices.view(num_frames, indices_per_frame)  # (num_frames, h*w)

    # 只处理完整的段（长度正好是 sequence_length 帧），不够的段直接舍弃
    num_segments = num_frames // sequence_length

    all_metadata = []

    with torch.no_grad():
        for seg_idx in range(num_segments):
            start_frame = seg_idx * sequence_length
            end_frame = start_frame + sequence_length  # 完整的 sequence_length 帧

            # 当前段的帧 (sequence_length, h*w)
            segment_frames = indices_reshaped[start_frame:end_frame]
            segment_flat = segment_frames.flatten()  # (sequence_length * h * w,)
            segment_length = sequence_length * indices_per_frame
            assert len(segment_flat) == segment_length, \
                f"段长度不匹配: {len(segment_flat)} != {segment_length}"

            # -------- 构造时间滑动窗口上下文 --------
            if seg_idx == 0:
                # 第一段：无额外时间上下文，只使用 BOS + 本段
                context_flat = None
            else:
                # 取上一段的最后 sequence_length-1 帧作为上下文
                # 例如 sequence_length=4，则取上一段的 frame[1],frame[2],frame[3]
                prev_start = (seg_idx - 1) * sequence_length
                prev_end = prev_start + sequence_length
                prev_segment_frames = indices_reshaped[prev_start:prev_end]  # (sequence_length, h*w)

                # 上一段的最后 sequence_length-1 帧
                context_frames = prev_segment_frames[-(sequence_length - 1):]  # (sequence_length-1, h*w)
                context_flat = context_frames.flatten()  # ((sequence_length-1) * h * w,)

            # -------- 拼接输入序列（仅第一段与原实现完全一致）--------
            bos_token_id = transformer_model.bos_token_id

            if context_flat is None:
                # 第一段：与 compress_indices_with_transformer 完全一致
                input_flat = segment_flat  # (L,)
                # 添加 batch 维度
                input_flat = input_flat.unsqueeze(0)  # (1, L)

                # BOS + 当前段
                segment_with_bos = torch.cat([
                    torch.full((1, 1), bos_token_id, dtype=torch.long, device=device),
                    input_flat
                ], dim=1)  # (1, L+1)

                # 获取概率分布
                rate_dummy, probs_full = transformer_model(segment_with_bos, return_probs=True)
                # probs_full: (1, L+1, vocab_size)
                probs = probs_full[:, :-1, :]  # (1, L, vocab_size) —— 对应当前段的每个 symbol

                # 与原函数保持一致的调试输出，仅对第一段打印一次
                pred_indices = probs.argmax(dim=-1)  # (1, L)
                actual_indices = input_flat.long()   # (1, L)
                pred_indices_flat = pred_indices[0]
                actual_indices_flat = actual_indices[0]

                correct = (pred_indices_flat == actual_indices_flat).sum().item()
                total = len(pred_indices_flat)
                accuracy = correct / total if total > 0 else 0.0

                # 根据 PMF 计算理论 rate
                probs_flat = probs[0]  # (L, vocab_size)
                indices_expanded = actual_indices_flat.unsqueeze(-1)
                prob_values = torch.gather(probs_flat, dim=-1, index=indices_expanded).squeeze(-1)
                eps = 1e-10
                nll_bits = -torch.log2(prob_values + eps)
                rate = nll_bits.mean().item()

                print(f"[滑动窗口压缩] 段 {seg_idx+1}/{num_segments} (首段): "
                      f"Accuracy={accuracy:.4f}, Rate={rate:.4f} bits/symbol")

                # 实际编码的 indices & probs（整段）
                target_indices = input_flat[0]                     # (L,)
                target_probs = probs                              # (1, L, vocab_size)
                context_len_tokens = 0
            else:
                # 后续段：使用「上一段最后 sequence_length-1 帧」作为时间上下文
                # 上下文不再编码，只作为条件
                context_flat = context_flat.to(device)
                context_len_tokens = context_flat.numel()

                # 拼接：context | 当前段
                concat_flat = torch.cat([context_flat, segment_flat], dim=0)  # (C + L,)
                concat_flat = concat_flat.unsqueeze(0)  # (1, C+L)

                # BOS + context + 当前段
                input_with_bos = torch.cat([
                    torch.full((1, 1), bos_token_id, dtype=torch.long, device=device),
                    concat_flat
                ], dim=1)  # (1, 1 + C + L)

                # 一次前向，获得整个序列的概率
                rate_dummy, probs_full = transformer_model(input_with_bos, return_probs=True)
                # probs_full: (1, 1 + C + L, vocab_size)
                probs_all = probs_full[:, :-1, :]  # (1, C + L, vocab_size) —— 位置对齐 indices

                # 我们只需要对「当前段」的 L 个位置做算术编码：
                # 对应 probs_all 中最后 L 个位置
                target_probs = probs_all[:, context_len_tokens:, :]  # (1, L, vocab_size)
                assert target_probs.shape[1] == segment_length, \
                    f"target_probs 长度不匹配: {target_probs.shape[1]} != {segment_length}"

                target_indices = segment_flat  # (L,)

                # 计算准确率和理论 rate（仅基于当前段）
                pred_indices = target_probs.argmax(dim=-1)  # (1, L)
                actual_indices = target_indices.unsqueeze(0).long()  # (1, L)
                pred_indices_flat = pred_indices[0]
                actual_indices_flat = actual_indices[0]

                correct = (pred_indices_flat == actual_indices_flat).sum().item()
                total = len(pred_indices_flat)
                accuracy = correct / total if total > 0 else 0.0

                probs_flat = target_probs[0]  # (L, vocab_size)
                indices_expanded = actual_indices_flat.unsqueeze(-1)
                prob_values = torch.gather(probs_flat, dim=-1, index=indices_expanded).squeeze(-1)
                eps = 1e-10
                nll_bits = -torch.log2(prob_values + eps)
                rate = nll_bits.mean().item()

                print(f"[滑动窗口压缩] 段 {seg_idx+1}/{num_segments}: "
                      f"上下文帧数={sequence_length-1}, Accuracy={accuracy:.4f}, Rate={rate:.4f} bits/symbol")

            # ------------- 算术编码 -------------
            # target_indices: (L,)
            # target_probs:   (1, L, vocab_size)
            segment_indices_batch = target_indices.unsqueeze(0)  # (1, L)

            # 这里我们仅实现一次性编码（test=False）；如需逐步编码，可扩展为自定义 incremental 版本
            if test:
                print("⚠ compress_indices_with_transformer_slidewindow 当前不支持 test=True 的逐步压缩，"
                      "将忽略 test 标志并使用一次性编码。")

            from transformer.encode_methods import compress_arithmetic_coding
            actual_vocab_size = target_probs.shape[-1]
            compressed_bytes, metadata = compress_arithmetic_coding(
                segment_indices_batch,
                target_probs,  # (1, L, vocab_size)
                vocab_size=actual_vocab_size,
                store_pmf=False
            )

            # 元数据记录
            metadata['segment_idx'] = seg_idx
            metadata['start_frame'] = start_frame
            metadata['end_frame'] = end_frame
            metadata['actual_frames'] = sequence_length
            metadata['actual_length'] = segment_length
            metadata['incremental'] = False  # 当前实现为一次性编码
            metadata['compressed_bytes'] = compressed_bytes
            metadata['original_indices'] = target_indices.cpu().numpy().tolist()
            metadata['context_token_length'] = int(context_len_tokens)
            metadata['context_type'] = 'sliding_window_last_segments'

            # 调试：打印实际 bits/index
            actual_compressed_size = len(compressed_bytes)
            actual_bits_per_index = (actual_compressed_size * 8) / segment_length
            print(f"    压缩大小: {actual_compressed_size} bytes, "
                  f"实际 bits/index: {actual_bits_per_index:.4f}, 理论 rate: {rate:.4f}")

            all_metadata.append(metadata)

    # 组合全局元数据
    processed_frames = num_segments * sequence_length
    combined_metadata = {
        'num_segments': num_segments,
        'sequence_length': sequence_length,
        'num_frames': num_frames,
        'processed_frames': processed_frames,
        'indices_h': indices_h,
        'indices_w': indices_w,
        'indices_per_frame': indices_per_frame,
        'segments': all_metadata,
        'slidewindow': True
    }

    # 与原接口保持一致：压缩数据都保存在 metadata 中，这里返回空 bytes
    return b'', combined_metadata




def compress_indices_with_transformer(transformer_model, indices: torch.Tensor, 
                                     device: str, sequence_length: int, h: int, w: int,
                                     video_path: str = None, enable_comparison_test: bool = False,
                                     test: bool = False) -> Tuple[bytes, Dict]:
    """
    使用 Transformer 和算术编码压缩 codebook indices
    
    将整个视频的indices按照每sequence_length帧分组，每组作为一个序列传给transformer压缩
    
    Args:
        transformer_model: Transformer 模型
        indices: codebook indices (num_frames, h, w)
        device: 设备
        sequence_length: 每段包含的帧数
        h: 空间高度
        w: 空间宽度
        video_path: 视频路径（用于对比测试）
        enable_comparison_test: 是否启用对比测试
    
    Returns:
        (compressed_bytes, metadata) - 只包含压缩数据，不包括元数据大小
    """
    num_frames, indices_h, indices_w = indices.shape
    indices = indices.to(device)
    
    # 将indices展平为 (num_frames, h*w)
    indices_per_frame = indices_h * indices_w
    indices_reshaped = indices.view(num_frames, indices_per_frame)  # (num_frames, h*w)
    
    # 只处理完整的段（长度正好是 sequence_length 帧），不够的段直接舍弃
    num_segments = num_frames // sequence_length
    
    # ========== 正常压缩流程 ========
    all_compressed = []
    all_metadata = []
    
    with torch.no_grad():
        for i in range(num_segments):
            start_frame = i * sequence_length
            end_frame = start_frame + sequence_length  # 完整的 sequence_length 帧
            segment_frames = indices_reshaped[start_frame:end_frame]  # (sequence_length, h*w)
            
            # 展平为1D: (sequence_length * h * w,)
            segment_flat = segment_frames.flatten()
            expected_length = sequence_length * indices_per_frame
            
            # 确保长度正确（应该是完整的）
            assert len(segment_flat) == expected_length, f"段长度不匹配: {len(segment_flat)} != {expected_length}"
            
            # 添加 batch 维度: (1, sequence_length * h * w)
            segment_flat = segment_flat.unsqueeze(0)
            
            # 添加 BOS token（与 test_transformer.py 一致）
            bos_token_id = transformer_model.bos_token_id
            segment_with_bos = torch.cat([
                torch.full((1, 1), bos_token_id, dtype=torch.long, device=device),
                segment_flat
            ], dim=1)  # (1, sequence_length * h * w + 1)
            
            # 获取概率分布（传入包含BOS的序列）
            rate, probs_full = transformer_model(segment_with_bos, return_probs=True)
            # probs_full: (1, sequence_length * h * w + 1, vocab_size)
            
            # 排除 BOS 位置的概率（与 test_transformer.py 一致）
            probs = probs_full[:, :-1, :]  # (1, sequence_length * h * w, vocab_size)

            print(f"rate: {rate}")

            # 计算 accuracy: 预测索引与实际索引的匹配率
            pred_indices = probs.argmax(dim=-1)  # (1, sequence_length * h * w)
            actual_indices = segment_flat.long()  # (1, sequence_length * h * w)
            
            # 计算准确率（使用完整段）
            pred_indices_flat = pred_indices[0]  # (sequence_length * h * w,)
            actual_indices_flat = actual_indices[0]  # (sequence_length * h * w,)
            
            # 调试信息：检查数据范围和格式
            if i == 0:
                print(f"  调试信息 - 段 {i+1}:")
                print(f"    实际索引范围: [{actual_indices_flat.min().item()}, {actual_indices_flat.max().item()}]")
                print(f"    预测索引范围: [{pred_indices_flat.min().item()}, {pred_indices_flat.max().item()}]")
                print(f"    实际索引前10个: {actual_indices_flat[:10].cpu().numpy()}")
                print(f"    预测索引前10个: {pred_indices_flat[:10].cpu().numpy()}")
                print(f"    vocab_size: {transformer_model.vocab_size}")
                print(f"    序列长度: {len(actual_indices_flat)}")
                print(f"    segment_flat shape: {segment_flat.shape}")
                print(f"    probs shape: {probs.shape}")
                
                # 检查概率分布
                probs_sample = probs[0, :5]  # 前5个位置的概率分布
                print(f"    前5个位置的概率分布最大值: {probs_sample.max(dim=-1)[0].cpu().numpy()}")
                print(f"    前5个位置的概率分布熵: {(-probs_sample * torch.log(probs_sample + 1e-10)).sum(dim=-1).cpu().numpy()}")
            
            correct = (pred_indices_flat == actual_indices_flat).sum().item()
            total = len(pred_indices_flat)
            accuracy = correct / total if total > 0 else 0.0
            
            # 计算 rate (熵率): 平均每个符号的比特数
            # 使用与 train_transformer.py 中 evaluate_compression_rate 相同的方法
            probs_flat = probs[0]  # (sequence_length * h * w, vocab_size)
            
            # 使用 gather 提取每个位置对应索引的概率（与 evaluate_compression_rate 一致）
            indices_expanded = actual_indices_flat.unsqueeze(-1)  # (sequence_length * h * w, 1)
            prob_values = torch.gather(probs_flat, dim=-1, index=indices_expanded).squeeze(-1)  # (sequence_length * h * w,)
            
            # 计算负对数似然（bits）：-log2(p)，与 evaluate_compression_rate 一致
            eps = 1e-10
            nll_bits = -torch.log2(prob_values + eps)  # (sequence_length * h * w,)
            
            # 平均每个符号的比特数
            rate = nll_bits.mean().item()
            
            # 打印统计信息
            print(f"  段 {i+1}/{num_segments}: Accuracy={accuracy:.4f}, Rate={rate:.4f} bits/symbol")
            
            # 从 probs 的实际形状获取 vocab_size（更可靠）
            actual_vocab_size = probs.shape[-1]
            segment_indices = segment_flat[0]  # (segment_length,)
            
            # 根据 test 参数选择压缩方法
            if test:
                # test=True: 使用逐步压缩（每次预测下一个index的概率，然后压缩）
                from transformer.encode_methods import compress_arithmetic_coding_incremental
                compressed_bytes, metadata = compress_arithmetic_coding_incremental(
                    segment_indices,
                    transformer_model,
                    device,
                    vocab_size=actual_vocab_size
                )
                is_incremental = True
            else:
                # test=False: 使用一次性编码压缩（已经有了完整的概率分布 probs）
                from transformer.encode_methods import compress_arithmetic_coding
                # probs: (1, sequence_length * h * w, vocab_size)
                # segment_indices: (sequence_length * h * w,) -> (1, sequence_length * h * w)
                segment_indices_batch = segment_indices.unsqueeze(0)  # (1, sequence_length * h * w)
                compressed_bytes, metadata = compress_arithmetic_coding(
                    segment_indices_batch,
                    probs,  # (1, sequence_length * h * w, vocab_size)
                    vocab_size=actual_vocab_size,
                    store_pmf=False  # 不存储PMF以减小文件大小
                )
                is_incremental = False
            
            # 添加段信息到元数据，并将压缩数据直接存储在 metadata 中
            metadata['segment_idx'] = i
            metadata['start_frame'] = start_frame
            metadata['end_frame'] = end_frame
            metadata['actual_frames'] = sequence_length
            metadata['actual_length'] = expected_length
            metadata['incremental'] = is_incremental  # 标记是否为逐步压缩格式
            metadata['compressed_bytes'] = compressed_bytes  # 将压缩数据直接存储在 metadata 中
            # 将原始 index segment 存储到 metadata 中，用于解压缩时比较差异
            metadata['original_indices'] = segment_flat[0].cpu().numpy().tolist()  # 存储为列表以便序列化
            
            # 打印压缩大小信息（用于调试）
            actual_compressed_size = len(compressed_bytes)
            segment_length_full = len(segment_flat[0])
            actual_bits_per_index = (actual_compressed_size * 8) / segment_length_full
            print(f"    压缩大小: {actual_compressed_size} bytes, 实际 bits/index: {actual_bits_per_index:.4f}, 理论 rate: {rate:.4f}")
            
            all_metadata.append(metadata)
    
    # 合并元数据（压缩数据已经存储在各自的 metadata 中）
    # 记录实际处理的帧数（只包含完整段）
    processed_frames = num_segments * sequence_length
    combined_metadata = {
        'num_segments': num_segments,
        'sequence_length': sequence_length,
        'num_frames': num_frames,  # 原始总帧数
        'processed_frames': processed_frames,  # 实际处理的帧数（完整段）
        'indices_h': indices_h,
        'indices_w': indices_w,
        'indices_per_frame': indices_per_frame,
        'segments': all_metadata
    }
    
    # 返回空的 bytes（因为压缩数据已经存储在 metadata 中）
    # 为了保持接口兼容性，返回一个空的 bytes
    return b'', combined_metadata




def decompress_indices_with_transformer(transformer_model, compressed_bytes: bytes,
                                       metadata: Dict, device: str,
                                       original_indices: torch.Tensor = None) -> torch.Tensor:
    """
    使用 Transformer 和算术编码解压 codebook indices
    
    Args:
        transformer_model: Transformer 模型
        compressed_bytes: 压缩的字节流（包含压缩数据和元数据）
        metadata: 元数据
        device: 设备
        original_indices: 可选的原始indices (num_frames, h, w)，用于验证解码结果
    
    Returns:
        indices: 解压后的 codebook indices (num_frames, h, w)
    """
    num_segments = metadata['num_segments']
    indices_h = metadata['indices_h']
    indices_w = metadata['indices_w']
    indices_per_frame = metadata['indices_per_frame']
    sequence_length = metadata['sequence_length']
    
    # 解压每个段（压缩数据现在直接从 metadata 中获取）
    all_indices = []
    
    with torch.no_grad():
        for i, seg_metadata in enumerate(metadata['segments']):
            # 从 metadata 中直接获取压缩数据
            if 'compressed_bytes' not in seg_metadata:
                raise ValueError(f"段 {i+1} 的 metadata 中没有 compressed_bytes")
            seg_bytes = seg_metadata['compressed_bytes']
            
            # 检查是否是逐步压缩格式
            is_incremental = seg_metadata.get('incremental', False)
            expected_length = seg_metadata['actual_length']
            
            # 优先使用 metadata 中存储的原始 indices，如果没有则使用传入的 original_indices
            original_segment_from_metadata = None
            if 'original_indices' in seg_metadata:
                # 从 metadata 中恢复原始 indices
                original_segment_from_metadata = torch.tensor(
                    seg_metadata['original_indices'], 
                    dtype=torch.long, 
                    device=device
                )  # (expected_length,)
            
            # 如果 metadata 中没有，则尝试从传入的 original_indices 中提取
            original_segment = original_segment_from_metadata
            if original_segment is None and original_indices is not None:
                start_frame = seg_metadata.get('start_frame', i * sequence_length)
                indices_per_frame = metadata['indices_per_frame']
                segment_start_idx = start_frame * indices_per_frame
                segment_end_idx = segment_start_idx + expected_length
                original_segment = original_indices.view(-1)[segment_start_idx:segment_end_idx].to(device)
            
            if is_incremental:
                # 使用逐步解码：每次预测下一个index的概率，然后解码
                from transformer.encode_methods import decompress_arithmetic_coding_incremental
                
                # 逐步解码整个段
                segment = decompress_arithmetic_coding_incremental(
                    seg_bytes,
                    seg_metadata,
                    transformer_model,
                    device,
                    original_indices=original_segment  # 传入原始indices用于验证
                )
            else:
                # 直接解码整个段：由于没有存储PMF，需要使用Transformer逐步解码
                # 使用逐步解码（因为需要Transformer预测概率分布，且numpyAc支持自回归解码）
                from transformer.encode_methods import decompress_arithmetic_coding_incremental
                segment = decompress_arithmetic_coding_incremental(
                    seg_bytes,
                    seg_metadata,
                    transformer_model,
                    device,
                    original_indices=original_segment
                )
            
            # 验证长度（应该是完整的段，没有填充）
            if len(segment) != expected_length:
                raise ValueError(f"解压后的段长度不匹配: {len(segment)} != {expected_length}")
            
            # 如果 metadata 中有原始 indices，进行逐段比较
            if original_segment_from_metadata is not None:
                segment_cpu = segment.cpu()
                original_cpu = original_segment_from_metadata.cpu()
                is_equal = torch.equal(segment_cpu, original_cpu)
                
                if is_equal:
                    print(f"  ✓ 段 {i+1}: 解压后的 indices 与原始 indices 完全一致")
                else:
                    # 计算差异统计
                    diff_mask = (segment_cpu != original_cpu)
                    num_diff = diff_mask.sum().item()
                    diff_ratio = num_diff / expected_length if expected_length > 0 else 0.0
                    
                    print(f"  ✗ 段 {i+1}: 解压后的 indices 与原始 indices 不一致")
                    print(f"    不一致的位置数: {num_diff}/{expected_length} ({diff_ratio*100:.2f}%)")
                    
                    # 显示前几个不一致的位置
                    diff_positions = torch.nonzero(diff_mask, as_tuple=False)
                    if len(diff_positions) > 0:
                        print(f"    前5个不一致的位置:")
                        for idx, pos in enumerate(diff_positions[:5]):
                            pos_val = pos.item()
                            orig_val = original_cpu[pos_val].item()
                            decomp_val = segment_cpu[pos_val].item()
                            print(f"      位置 {pos_val}: 原始={orig_val}, 解压={decomp_val}")
            
            all_indices.append(segment)
    
    # 合并所有段
    indices_flat = torch.cat(all_indices, dim=0)
    
    # 重新整形为 (processed_frames, h, w)
    # 只处理完整段，所以使用 processed_frames 而不是 num_frames
    processed_frames = metadata.get('processed_frames', num_segments * metadata['sequence_length'])
    expected_total_length = processed_frames * indices_per_frame
    
    # 确保长度匹配
    if len(indices_flat) != expected_total_length:
        raise ValueError(f"解压后的总长度不匹配: {len(indices_flat)} != {expected_total_length}")
    
    indices = indices_flat.view(processed_frames, indices_h, indices_w)
    
    # 如果原始帧数大于处理的帧数，需要填充到原始长度（用于后续处理）
    num_frames = metadata['num_frames']
    if processed_frames < num_frames:
        # 在末尾填充零值
        padding_frames = num_frames - processed_frames
        padding = torch.zeros(padding_frames, indices_h, indices_w, 
                             dtype=indices.dtype, device=indices.device)
        indices = torch.cat([indices, padding], dim=0)
    
    return indices


def decompress_indices_with_transformer_slidewindow(transformer_model, compressed_bytes: bytes,
                                                   metadata: Dict, device: str,
                                                   original_indices: torch.Tensor = None) -> torch.Tensor:
    """
    使用 Transformer 和算术编码解压 codebook indices（滑动窗口版本）
    
    与 decompress_indices_with_transformer 的区别：
    - 第一段：使用 BOS token 开始，正常解码
    - 后续段：使用前一段的最后 sequence_length-1 帧作为上下文，然后解码当前段
    
    Args:
        transformer_model: Transformer 模型
        compressed_bytes: 压缩的字节流（包含压缩数据和元数据）
        metadata: 元数据（必须包含 slidewindow=True）
        device: 设备
        original_indices: 可选的原始indices (num_frames, h, w)，用于验证解码结果
    
    Returns:
        indices: 解压后的 codebook indices (num_frames, h, w)
    """
    # 检查是否是滑动窗口格式
    if not metadata.get('slidewindow', False):
        raise ValueError("metadata 中没有 slidewindow=True，请使用 decompress_indices_with_transformer")
    
    num_segments = metadata['num_segments']
    indices_h = metadata['indices_h']
    indices_w = metadata['indices_w']
    indices_per_frame = metadata['indices_per_frame']
    sequence_length = metadata['sequence_length']
    bos_token_id = transformer_model.bos_token_id
    
    # 解压每个段（压缩数据现在直接从 metadata 中获取）
    all_indices = []
    
    # 用于存储前一段的indices（用于滑动窗口上下文）
    previous_segment_indices = None
    
    with torch.no_grad():
        for i, seg_metadata in enumerate(metadata['segments']):
            # 从 metadata 中直接获取压缩数据
            if 'compressed_bytes' not in seg_metadata:
                raise ValueError(f"段 {i+1} 的 metadata 中没有 compressed_bytes")
            seg_bytes = seg_metadata['compressed_bytes']
            
            # 检查是否是逐步压缩格式
            is_incremental = seg_metadata.get('incremental', False)
            expected_length = seg_metadata['actual_length']
            
            # 优先使用 metadata 中存储的原始 indices，如果没有则使用传入的 original_indices
            original_segment_from_metadata = None
            if 'original_indices' in seg_metadata:
                # 从 metadata 中恢复原始 indices
                original_segment_from_metadata = torch.tensor(
                    seg_metadata['original_indices'], 
                    dtype=torch.long, 
                    device=device
                )  # (expected_length,)
            
            # 如果 metadata 中没有，则尝试从传入的 original_indices 中提取
            original_segment = original_segment_from_metadata
            if original_segment is None and original_indices is not None:
                start_frame = seg_metadata.get('start_frame', i * sequence_length)
                indices_per_frame = metadata['indices_per_frame']
                segment_start_idx = start_frame * indices_per_frame
                segment_end_idx = segment_start_idx + expected_length
                original_segment = original_indices.view(-1)[segment_start_idx:segment_end_idx].to(device)
            
            # 滑动窗口解码：根据段索引决定上下文
            if i == 0:
                # 第一段：使用 BOS token 开始
                context_tokens = torch.tensor([bos_token_id], dtype=torch.long, device=device)
            else:
                # 后续段：使用前一段的最后 sequence_length-1 帧作为上下文
                if previous_segment_indices is None:
                    raise ValueError(f"段 {i+1} 需要前一段的indices作为上下文，但 previous_segment_indices 为空")
                
                # 前一段的最后 sequence_length-1 帧的indices
                context_token_length = seg_metadata.get('context_token_length', (sequence_length - 1) * indices_per_frame)
                context_tokens = previous_segment_indices[-context_token_length:]  # (context_token_length,)
                
                # 在前面添加 BOS token（如果需要）
                context_tokens = torch.cat([torch.tensor([bos_token_id], dtype=torch.long, device=device), 
                                           context_tokens], dim=0)
            
            # 使用逐步解码（因为需要Transformer预测概率分布，且numpyAc支持自回归解码）
            from transformer.encode_methods import decompress_arithmetic_coding_incremental
            
            # 修改 decompress_arithmetic_coding_incremental 以支持自定义上下文
            # 由于 decompress_arithmetic_coding_incremental 内部会使用 BOS token 初始化上下文，
            # 我们需要创建一个包装函数，或者直接在这里实现逐步解码逻辑
            
            # 方案：直接在这里实现滑动窗口的逐步解码
            seq_len = expected_length
            vocab_size = seg_metadata.get('vocab_size', transformer_model.vocab_size)
            
            # 初始化 numpyAc 解码器
            try:
                from numpyAc.numpyAc import arithmeticDeCoding
            except ImportError:
                raise ImportError("请安装 numpyAc 库: pip install numpyAc")
            decodec = arithmeticDeCoding(seg_bytes, seq_len, vocab_size, binfile=None)
            
            # 使用当前上下文开始解码
            current_context = context_tokens.clone()  # 包含 BOS + 前一段的上下文（如果有）
            decoded_segment = []
            
            # 逐步解码当前段
            for pos in range(seq_len):
                # 使用 get_conditional_probs 获取当前位置的概率分布
                next_probs = transformer_model.transformer.get_conditional_probs(
                    current_context,
                    temperature=1.0,
                    use_rope=True
                )  # (vocab_size,)
                
                # 归一化概率
                from transformer.encode_methods import _normalize_probs
                next_probs = _normalize_probs(next_probs.unsqueeze(0)).squeeze(0)  # (vocab_size,)
                
                # 转换为 numpy 数组（numpyAc 需要 numpy 数组）
                next_probs_np = next_probs.cpu().numpy().reshape(1, vocab_size)  # (1, vocab_size)
                
                # 使用 numpyAc 解码当前位置
                decoded_index = decodec.decode(next_probs_np)
                
                # 转换为 torch tensor
                decoded_index = torch.tensor(decoded_index, dtype=torch.long, device=device)
                
                # 验证：如果提供了原始indices，比较解码的index和原始index
                if original_segment is not None and pos < len(original_segment):
                    original_index = original_segment[pos].item() if isinstance(original_segment, torch.Tensor) else original_segment[pos]
                    decoded_index_val = decoded_index.item()
                    
                    if original_index != decoded_index_val:
                        if pos < 10 or pos % 1000 == 0:
                            print(f"  ⚠️ 段 {i+1} 位置 {pos}: 解码index={decoded_index_val}, 原始index={original_index}, 不一致!")
                    elif pos < 10 or pos % 1000 == 0:
                        print(f"  ✓ 段 {i+1} 位置 {pos}: 解码index={decoded_index_val}, 原始index={original_index}, 一致")
                
                # 将解码的index添加到上下文（用于下一步预测）
                current_context = torch.cat([current_context, decoded_index.unsqueeze(0)], dim=0)
                decoded_segment.append(decoded_index)
            
            # 合并所有解码的indices
            segment = torch.stack(decoded_segment, dim=0)  # (seq_len,)
            
            # 验证长度（应该是完整的段，没有填充）
            if len(segment) != expected_length:
                raise ValueError(f"解压后的段长度不匹配: {len(segment)} != {expected_length}")
            
            # 如果 metadata 中有原始 indices，进行逐段比较
            if original_segment_from_metadata is not None:
                segment_cpu = segment.cpu()
                original_cpu = original_segment_from_metadata.cpu()
                is_equal = torch.equal(segment_cpu, original_cpu)
                
                if is_equal:
                    print(f"  ✓ 段 {i+1}: 解压后的 indices 与原始 indices 完全一致")
                else:
                    # 计算差异统计
                    diff_mask = (segment_cpu != original_cpu)
                    num_diff = diff_mask.sum().item()
                    diff_ratio = num_diff / expected_length if expected_length > 0 else 0.0
                    
                    print(f"  ✗ 段 {i+1}: 解压后的 indices 与原始 indices 不一致")
                    print(f"    不一致的位置数: {num_diff}/{expected_length} ({diff_ratio*100:.2f}%)")
                    
                    # 显示前几个不一致的位置
                    diff_positions = torch.nonzero(diff_mask, as_tuple=False)
                    if len(diff_positions) > 0:
                        print(f"    前5个不一致的位置:")
                        for idx, pos in enumerate(diff_positions[:5]):
                            pos_val = pos.item()
                            orig_val = original_cpu[pos_val].item()
                            decomp_val = segment_cpu[pos_val].item()
                            print(f"      位置 {pos_val}: 原始={orig_val}, 解压={decomp_val}")
            
            all_indices.append(segment)
            
            # 更新 previous_segment_indices：保存当前段的indices，用于下一段的上下文
            # 注意：需要保存最后 sequence_length-1 帧的indices
            context_token_length = seg_metadata.get('context_token_length', (sequence_length - 1) * indices_per_frame)
            previous_segment_indices = segment[-context_token_length:]  # (context_token_length,)
    
    # 合并所有段
    indices_flat = torch.cat(all_indices, dim=0)
    
    # 重新整形为 (processed_frames, h, w)
    # 只处理完整段，所以使用 processed_frames 而不是 num_frames
    processed_frames = metadata.get('processed_frames', num_segments * metadata['sequence_length'])
    expected_total_length = processed_frames * indices_per_frame
    
    # 确保长度匹配
    if len(indices_flat) != expected_total_length:
        raise ValueError(f"解压后的总长度不匹配: {len(indices_flat)} != {expected_total_length}")
    
    indices = indices_flat.view(processed_frames, indices_h, indices_w)
    
    # 如果原始帧数大于处理的帧数，需要填充到原始长度（用于后续处理）
    num_frames = metadata['num_frames']
    if processed_frames < num_frames:
        # 在末尾填充零值
        padding_frames = num_frames - processed_frames
        padding = torch.zeros(padding_frames, indices_h, indices_w, 
                             dtype=indices.dtype, device=indices.device)
        indices = torch.cat([indices, padding], dim=0)
    
    return indices


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 PSNR"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def process_video(video_path: str, vqgan_model, transformer_model, 
                 normalize: transforms.Normalize, unnormalize: NormalizeInverse,
                 device: str, sequence_length: int, image_size: int,
                 output_dir: str, recon_output_dir: str, original_output_dir: str,
                 enable_comparison_test: bool = True, test: bool = False) -> Dict:
    """
    处理单个视频：压缩和重建
    
    Returns:
        包含各种指标的字典
    """
    # 加载整个视频的所有帧（与 data_pre_process.py 和 video_utils.py 保持一致）
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频的 fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # 默认帧率
    
    frames = []
    frames_read = 0
    
    # 读取所有帧（与 video_utils.py 的 VideoDataset 类似，但读取所有帧而不是 sequence_length）
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换 BGR 到 RGB（OpenCV 默认读取为 BGR）
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 调整大小到目标分辨率（与 video_utils.py 一致）
        if frame.shape[:2] != (image_size, image_size):
            frame = cv2.resize(frame, (image_size, image_size), 
                             interpolation=cv2.INTER_LINEAR)
        
        frames.append(frame)
        frames_read += 1
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"无法读取视频或视频为空: {video_path}")
    
    # 转换为 numpy 数组: (num_frames, height, width, channels)
    # 确保是 uint8 类型（与 video_utils.py 和 data_pre_process.py 一致）
    video_np = np.stack(frames, axis=0).astype(np.uint8)  # (num_frames, h, w, c)
    num_frames = len(frames)
    
    # 清理 frames 列表释放内存
    del frames
    
    # 转换为 torch tensor（保持 uint8 类型，在 encode_video_to_indices 中再转换为 float）
    video_tensor = torch.from_numpy(video_np)

    print(f"video_tensor shape: {video_tensor.shape}")
    
    # ========== 步骤1: VQGAN 编码整个视频 ==========
    indices = encode_video_to_indices(vqgan_model, video_tensor, device)
    # indices: (num_frames, h, w)
    
    # 获取indices的空间尺寸
    _, indices_h, indices_w = indices.shape
    
    # ========== 步骤2: Transformer + 算术编码压缩（使用滑动窗口）==========
    # 传递 video_path 用于对比测试
    compressed_bytes, compression_metadata = compress_indices_with_transformer_slidewindow(
        transformer_model, indices, device, sequence_length, indices_h, indices_w,
        video_path=video_path, enable_comparison_test=enable_comparison_test,
        test=test  # 传递 test 参数，控制是否在压缩时进行立即解压验证
    )
    
    # 将 fps 添加到 metadata 中，用于后续保存视频
    compression_metadata['fps'] = fps
    
    # ========== 步骤3: 保存压缩文件（合并compressed_bytes和metadata）==========
    video_name = Path(video_path).stem
    compressed_file_path = os.path.join(output_dir, f"{video_name}_compressed.pkl")
    # compressed_file_path_without_meta = os.path.join(output_dir+"/without_meta/", f"{video_name}_compressed.pkl")
    
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(output_dir+"/without_meta/", exist_ok=True)
    # 将compressed_bytes和metadata合并保存到一个文件
    file_data = {
        'compressed_bytes': compressed_bytes,
        'metadata': compression_metadata
    }
    with open(compressed_file_path, 'wb') as f:
        pickle.dump(file_data, f)

    # with open(compressed_file_path_without_meta) as f:
    #     pickle.dump(compressed_bytes, f)
    
    # ========== 步骤4: 读取并解压（仅在 test=True 时执行）==========
    if test:
        print(f"\n{'='*60}")
        print("步骤4: 读取并解压")
        print('='*60)
        
        with open(compressed_file_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        loaded_bytes = loaded_data['compressed_bytes']
        loaded_metadata = loaded_data['metadata']
        
        # 根据 metadata 中的 slidewindow 标记选择解压函数
        if loaded_metadata.get('slidewindow', False):
            # 使用滑动窗口解压函数
            decompressed_indices = decompress_indices_with_transformer_slidewindow(
                transformer_model, loaded_bytes, loaded_metadata, device,
                original_indices=indices  # 传入原始indices用于验证
            )
        else:
            # 使用普通解压函数
            decompressed_indices = decompress_indices_with_transformer(
                transformer_model, loaded_bytes, loaded_metadata, device,
                original_indices=indices  # 传入原始indices用于验证
            )
        # decompressed_indices: (num_frames, h, w)
        
        # ========== 步骤4.5: 比较压缩前后的indices是否一致 ==========
        print(f"\n{'='*60}")
        print("比较压缩前后的indices")
        print('='*60)
        
        # 只比较实际处理的帧（不包括填充的零值帧）
        processed_frames = compression_metadata['processed_frames']
        original_indices_processed = indices[:processed_frames]  # 只取实际处理的帧
        decompressed_indices_processed = decompressed_indices[:processed_frames]  # 只取实际处理的帧
        
        # 确保形状一致
        if original_indices_processed.shape != decompressed_indices_processed.shape:
            print(f"⚠ 警告: 形状不匹配!")
            print(f"  原始indices形状: {original_indices_processed.shape}")
            print(f"  解压indices形状: {decompressed_indices_processed.shape}")
        else:
            # 转换为相同设备进行比较
            original_indices_cpu = original_indices_processed.cpu()
            decompressed_indices_cpu = decompressed_indices_processed.cpu()
            
            # 比较是否完全一致
            indices_equal = torch.equal(original_indices_cpu, decompressed_indices_cpu)
            
            if indices_equal:
                print(f"✓ 压缩前后的indices完全一致!")
                print(f"  处理的帧数: {processed_frames}")
                print(f"  每帧indices数: {indices_h * indices_w}")
                print(f"  总indices数: {processed_frames * indices_h * indices_w}")
            else:
                # 计算差异统计
                diff_mask = (original_indices_cpu != decompressed_indices_cpu)
                num_diff = diff_mask.sum().item()
                total_indices = original_indices_cpu.numel()
                diff_ratio = num_diff / total_indices if total_indices > 0 else 0.0
                
                print(f"✗ 压缩前后的indices不一致!")
                print(f"  处理的帧数: {processed_frames}")
                print(f"  总indices数: {total_indices}")
                print(f"  不一致的indices数: {num_diff}")
                print(f"  不一致比例: {diff_ratio * 100:.4f}%")
                
                # 显示前几个不一致的位置
                diff_positions = torch.nonzero(diff_mask, as_tuple=False)
                if len(diff_positions) > 0:
                    print(f"\n  前10个不一致的位置:")
                    for idx, pos in enumerate(diff_positions[:10]):
                        frame_idx, h_idx, w_idx = pos[0].item(), pos[1].item(), pos[2].item()
                        orig_val = original_indices_cpu[frame_idx, h_idx, w_idx].item()
                        decomp_val = decompressed_indices_cpu[frame_idx, h_idx, w_idx].item()
                        print(f"    位置 [{frame_idx}, {h_idx}, {w_idx}]: 原始={orig_val}, 解压={decomp_val}")
                    
                    if len(diff_positions) > 10:
                        print(f"    ... 还有 {len(diff_positions) - 10} 个不一致的位置")
                
                # 按帧统计差异
                print(f"\n  按帧统计差异:")
                for frame_idx in range(min(processed_frames, 10)):  # 只显示前10帧
                    frame_orig = original_indices_cpu[frame_idx]
                    frame_decomp = decompressed_indices_cpu[frame_idx]
                    frame_diff = (frame_orig != frame_decomp).sum().item()
                    if frame_diff > 0:
                        print(f"    帧 {frame_idx}: {frame_diff} 个不一致的indices")
                
                # 统计差异值的分布
                diff_values_orig = original_indices_cpu[diff_mask]
                diff_values_decomp = decompressed_indices_cpu[diff_mask]
                if len(diff_values_orig) > 0:
                    print(f"\n  差异值统计:")
                    print(f"    原始值范围: [{diff_values_orig.min().item()}, {diff_values_orig.max().item()}]")
                    print(f"    解压值范围: [{diff_values_decomp.min().item()}, {diff_values_decomp.max().item()}]")
                    print(f"    平均差异: {(diff_values_orig.float() - diff_values_decomp.float()).abs().mean().item():.4f}")
        
        print(f"{'='*60}\n")
        
        # 使用解压后的indices进行重建
        indices_for_recon = decompressed_indices
    else:
        # 如果 test=False，直接使用原始indices进行重建
        print(f"\n跳过解压验证步骤，直接使用原始indices进行重建")
        indices_for_recon = indices
    
    # ========== 步骤5: VQGAN 解码重建视频 ==========
    recon_video = decode_indices_to_video(
        vqgan_model, indices_for_recon, device, normalize, unnormalize, batch_size=32
    )
    
    # ========== 步骤6: 保存重建视频 ==========
    # 参考 compression.py 的方式保存视频，使用 torchvision.io.write_video
    os.makedirs(recon_output_dir, exist_ok=True)
    recon_video_path = os.path.join(recon_output_dir, f"{video_name}_recon.mp4")
    
    # 只保存实际处理的帧（不包括填充的零值帧）
    processed_frames = compression_metadata['processed_frames']
    recon_video_actual = recon_video[:processed_frames]  # 只取实际处理的帧
    
    # 检查视频帧格式
    if len(recon_video_actual) == 0:
        raise ValueError("重建的视频没有有效帧")
    
    frame_height, frame_width = recon_video_actual[0].shape[:2]
    print(f"保存重建视频: {len(recon_video_actual)} 帧, 尺寸: {frame_width}x{frame_height}")
    
    # 从 metadata 中获取 fps，如果没有则使用默认值
    fps = compression_metadata.get('fps', 30.0)
    if fps <= 0:
        fps = 30.0
    
    # 将 numpy 数组转换为 torch.Tensor，格式为 (T, C, H, W)，值在 [0, 1] 范围内
    # recon_video_actual 当前是 (T, H, W, C) RGB uint8
    recon_tensor = torch.from_numpy(recon_video_actual).float() / 255.0  # (T, H, W, C) float32 [0, 1]
    recon_tensor = recon_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
    
    # 确保值在 [0, 1] 范围内
    recon_tensor = recon_tensor.clamp(0.0, 1.0)
    
    # 转换为 uint8 并 permute 为 (T, H, W, C) 格式（torchvision.write_video 需要）
    recon_tensor_uint8 = (recon_tensor * 255.0).to(torch.uint8)
    recon_tensor_uint8 = recon_tensor_uint8.permute(0, 2, 3, 1).cpu()  # (T, H, W, C) RGB uint8
    
    # 使用 torchvision.io.write_video 保存视频（与 compression.py 一致）
    try:
        write_video(recon_video_path, recon_tensor_uint8, fps=fps)
        print(f"视频已保存: {recon_video_path} (使用 torchvision.io.write_video)")
    except Exception as e:
        print(f"使用 torchvision.io.write_video 保存失败: {e}")
        # 如果失败，尝试使用 OpenCV 作为备选方案
        print("尝试使用 OpenCV 作为备选方案...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(recon_video_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            raise ValueError(f"无法创建视频文件")
        
        for frame in recon_video_actual:
            # 确保是 uint8 类型
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            # 转换 RGB 到 BGR（OpenCV 需要 BGR 格式）
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        print(f"视频已保存: {recon_video_path} (使用 OpenCV)")
    
    # 验证视频文件是否创建成功
    if not os.path.exists(recon_video_path):
        raise ValueError(f"视频文件创建失败: {recon_video_path}")
    
    file_size = os.path.getsize(recon_video_path)
    if file_size == 0:
        raise ValueError(f"视频文件为空: {recon_video_path}")
    
    print(f"视频文件大小: {file_size / 1024:.2f} KB")
    
    # ========== 步骤6.5: 保存裁剪后的原始视频 ==========
    # 只保存实际处理的帧（与重建视频帧数一致）
    os.makedirs(original_output_dir, exist_ok=True)
    original_video_path = os.path.join(original_output_dir, f"{video_name}_original.mp4")
    
    # 裁剪原始视频，只保留实际处理的帧
    video_original_actual = video_np[:processed_frames]  # 只取实际处理的帧
    
    # 检查视频帧格式
    if len(video_original_actual) == 0:
        raise ValueError("裁剪后的原始视频没有有效帧")
    
    print(f"保存裁剪后的原始视频: {len(video_original_actual)} 帧, 尺寸: {frame_width}x{frame_height}")
    
    # 将 numpy 数组转换为 torch.Tensor，格式为 (T, C, H, W)，值在 [0, 1] 范围内
    # video_original_actual 当前是 (T, H, W, C) RGB uint8
    original_tensor = torch.from_numpy(video_original_actual).float() / 255.0  # (T, H, W, C) float32 [0, 1]
    original_tensor = original_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
    
    # 确保值在 [0, 1] 范围内
    original_tensor = original_tensor.clamp(0.0, 1.0)
    
    # 转换为 uint8 并 permute 为 (T, H, W, C) 格式（torchvision.write_video 需要）
    original_tensor_uint8 = (original_tensor * 255.0).to(torch.uint8)
    original_tensor_uint8 = original_tensor_uint8.permute(0, 2, 3, 1).cpu()  # (T, H, W, C) RGB uint8
    
    # 使用 torchvision.io.write_video 保存视频
    try:
        write_video(original_video_path, original_tensor_uint8, fps=fps)
        print(f"裁剪后的原始视频已保存: {original_video_path} (使用 torchvision.io.write_video)")
    except Exception as e:
        print(f"使用 torchvision.io.write_video 保存失败: {e}")
        # 如果失败，尝试使用 OpenCV 作为备选方案
        print("尝试使用 OpenCV 作为备选方案...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(original_video_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            raise ValueError(f"无法创建视频文件")
        
        for frame in video_original_actual:
            # 确保是 uint8 类型
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            # 转换 RGB 到 BGR（OpenCV 需要 BGR 格式）
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        print(f"裁剪后的原始视频已保存: {original_video_path} (使用 OpenCV)")
    
    # 验证视频文件是否创建成功
    if not os.path.exists(original_video_path):
        raise ValueError(f"裁剪后的原始视频文件创建失败: {original_video_path}")
    
    original_file_size = os.path.getsize(original_video_path)
    if original_file_size == 0:
        raise ValueError(f"裁剪后的原始视频文件为空: {original_video_path}")
    
    print(f"裁剪后的原始视频文件大小: {original_file_size / 1024:.2f} KB")
    
    # ========== 步骤7: 计算指标 ==========
    # 原始视频大小（字节）
    original_size = video_np.nbytes
    
    # 压缩后大小（压缩数据现在存储在 metadata 中）
    # 计算所有段的压缩数据总大小
    num_segments = compression_metadata['num_segments']
    processed_frames = compression_metadata['processed_frames']
    
    # 从每个段的 metadata 中获取压缩数据大小
    compressed_size = 0
    for seg_metadata in compression_metadata['segments']:
        if 'compressed_bytes' in seg_metadata:
            compressed_size += len(seg_metadata['compressed_bytes'])
    
    # 计算 BPP (bits per pixel) - 只基于实际压缩数据
    num_pixels = video_np.shape[0] * video_np.shape[1] * video_np.shape[2]
    bpp = (compressed_size * 8) / num_pixels
    
    # 计算 bits/index - 只基于实际压缩数据和实际处理的 indices
    # 只计算实际压缩的 indices 数量（processed_frames，不包括未处理的帧）
    num_indices = processed_frames * indices_h * indices_w
    bits_per_index = (compressed_size * 8) / num_indices if num_indices > 0 else 0.0
    
    # 计算 PSNR（只计算实际处理的帧）
    psnr_values = []
    processed_frames = compression_metadata['processed_frames']
    # 使用实际处理的帧数，而不是所有帧（包括填充的零值帧）
    num_frames_to_compare = min(len(video_np), processed_frames, len(recon_video_actual))
    for i in range(num_frames_to_compare):
        psnr = calculate_psnr(video_np[i], recon_video_actual[i])
        psnr_values.append(psnr)
    avg_psnr = np.mean(psnr_values) if psnr_values else 0.0
    
    # 计算压缩比
    compression_ratio = original_size / compressed_size
    
    metrics = {
        'video_path': video_path,
        'original_size_bytes': original_size,
        'compressed_size_bytes': compressed_size,
        'compression_ratio': compression_ratio,
        'bpp': bpp,
        'bits_per_index': bits_per_index,
        'psnr': avg_psnr,
        'psnr_per_frame': psnr_values,
        'num_indices': num_indices,
        'num_pixels': num_pixels,
        'original_video_path': original_video_path,  # 裁剪后的原始视频路径
        'recon_video_path': recon_video_path,  # 重建视频路径
        'compressed_data_size': compressed_size  # 压缩的纯数据大小（字节）
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='视频压缩和重建评估')
    parser.add_argument('--video_dir', type=str, 
    default='/data/huyang/data/test',
                       help='输入视频目录')
    parser.add_argument('--vq_config', type=str, 
    default="/home/huyang/VqVaeVideo-master/VqVaeVideo-master/multitask/taming.json",
                       help='VQGAN 配置文件路径')
    parser.add_argument('--vq_ckpt', type=str, 
    default="/data/huyang/save_data_taming/checkpoint_epoch6.pth.tar",
                       help='VQGAN checkpoint 路径')
    parser.add_argument('--transformer_ckpt', type=str, 
    default="/data/huyang/vqgan_result/16/transformer/best_model.pth",
                       help='Transformer checkpoint 路径')
    parser.add_argument('--output_dir', type=str, 
    default="/data/huyang/vqgan_result/16/output",
                       help='压缩文件输出目录')
    parser.add_argument('--recon_output_dir', type=str, 
    default="/data/huyang/vqgan_result/16/output_video",
                       help='重建视频输出目录')
    parser.add_argument('--output_video_original', type=str, 
    default="/data/huyang/vqgan_result/16/output_video_original",
                       help='裁剪后的原始视频输出目录')
    parser.add_argument('--sequence_length', type=int, default=4,
                       help='序列长度')
    parser.add_argument('--image_size', type=int, default=256,
                       help='图像尺寸')
    parser.add_argument('--device', type=str, default='cuda:4',
                       help='设备')
    parser.add_argument('--vocab_size', type=int, default=2048,
                       help='词汇表大小')
    parser.add_argument('--test', action='store_true',
                       help='是否进行解压验证（测试压缩和解压的一致性）')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    vqgan_model, vq_config = load_vqgan_model(args.vq_config, args.vq_ckpt, device)
    
    # 从配置文件获取 vocab_size
    if 'model_args' in vq_config:
        vocab_size = vq_config['model_args'].get('vocab_size', args.vocab_size)
    else:
        vocab_size = args.vocab_size
    
    # 计算实际的序列长度（sequence_length * h * w）
    # 需要先获取 indices 的空间尺寸
    # 临时编码一帧来获取尺寸
    test_video = torch.zeros(1, args.image_size, args.image_size, 3, dtype=torch.uint8)
    test_indices = encode_video_to_indices(vqgan_model, test_video, device, batch_size=1)
    _, indices_h, indices_w = test_indices.shape
    indices_per_frame = indices_h * indices_w
    seq_len = args.sequence_length * indices_per_frame
    
    print(f"Indices 空间尺寸: {indices_h} x {indices_w}")
    print(f"每帧 indices 数: {indices_per_frame}")
    print(f"序列长度 (sequence_length * h * w): {seq_len}")
    
    transformer_model, vocab_size = load_transformer_model(
        args.transformer_ckpt, vocab_size, seq_len, device
    )
    
    # 创建归一化和反归一化
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=mean, std=std)
    unnormalize = NormalizeInverse(mean=mean, std=std)
    
    # 获取视频文件列表
    video_files = list_videos(args.video_dir)
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频
    all_metrics = []
    for video_path in tqdm(video_files, desc="处理视频"):
        try:
            metrics = process_video(
                video_path, vqgan_model, transformer_model,
                normalize, unnormalize, device, args.sequence_length,
                args.image_size, args.output_dir, args.recon_output_dir,
                args.output_video_original,
                enable_comparison_test=False,  # 不再使用这个参数
                test=args.test  # 使用新的 test 参数控制是否进行解压验证
            )
            all_metrics.append(metrics)
            
            print(f"\n视频: {Path(video_path).name}")
            print(f"  压缩比: {metrics['compression_ratio']:.2f}:1")
            print(f"  BPP: {metrics['bpp']:.4f}")
            print(f"  Bits/index: {metrics['bits_per_index']:.4f}")
            print(f"  PSNR: {metrics['psnr']:.2f} dB")
        except Exception as e:
            print(f"处理视频 {video_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    # 计算总体统计
    if all_metrics:
        print(f"\n{'='*60}")
        print("总体统计")
        print('='*60)
        
        avg_bpp = np.mean([m['bpp'] for m in all_metrics])
        avg_bits_per_index = np.mean([m['bits_per_index'] for m in all_metrics])
        avg_psnr = np.mean([m['psnr'] for m in all_metrics])
        avg_compression_ratio = np.mean([m['compression_ratio'] for m in all_metrics])
        
        total_original = sum([m['original_size_bytes'] for m in all_metrics])
        total_compressed = sum([m['compressed_size_bytes'] for m in all_metrics])
        
        print(f"处理视频数: {len(all_metrics)}")
        print(f"总原始大小: {total_original / 1024 / 1024:.2f} MB")
        print(f"总压缩大小: {total_compressed / 1024 / 1024:.2f} MB")
        print(f"平均压缩比: {avg_compression_ratio:.2f}:1")
        print(f"平均 BPP: {avg_bpp:.4f}")
        print(f"平均 Bits/index: {avg_bits_per_index:.4f}")
        print(f"平均 PSNR: {avg_psnr:.2f} dB")
        
        # 保存统计结果
        stats_path = os.path.join(args.output_dir, 'statistics.json')
        stats = {
            'num_videos': len(all_metrics),
            'avg_bpp': float(avg_bpp),
            'avg_bits_per_index': float(avg_bits_per_index),
            'avg_psnr': float(avg_psnr),
            'avg_compression_ratio': float(avg_compression_ratio),
            'total_original_bytes': int(total_original),
            'total_compressed_bytes': int(total_compressed),
            'metrics': all_metrics
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n统计结果已保存到: {stats_path}")
        
        # ========== 生成 CSV 文件 ==========
        csv_path = os.path.join(args.output_dir, 'video_comparison.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['orig', 'recon', 'size'])
            # 写入数据（使用绝对路径）
            for m in all_metrics:
                orig_path = os.path.abspath(m.get('original_video_path', '')) if m.get('original_video_path') else ''
                recon_path = os.path.abspath(m.get('recon_video_path', '')) if m.get('recon_video_path') else ''
                compressed_size = m.get('compressed_data_size', 0)
                writer.writerow([orig_path, recon_path, compressed_size])
        print(f"CSV 文件已保存到: {csv_path}")


if __name__ == '__main__':
    main()

