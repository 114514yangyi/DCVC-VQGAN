#!/usr/bin/env python3
"""
编码方法模块：提供算术编码函数（使用 numpyAc 库）
"""

import torch
import numpy as np
from typing import Dict, Tuple
import struct
import tempfile
import os

# 本模块**强依赖** numpyAc 提供的算术编码。
# 如果库不可用，直接抛出 ImportError，而不是使用任何简化或回退实现。
try:
    import numpyAc
    from numpyAc.numpyAc import arithmeticCoding, arithmeticDeCoding
    HAS_NUMPYAC = True
except ImportError:
    HAS_NUMPYAC = False
    raise ImportError("请安装 numpyAc 库")

from tqdm import tqdm


def _normalize_probs(probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """归一化概率分布"""
    probs = probs.clamp(min=eps)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def _pmf_to_cdf(pmf: torch.Tensor) -> torch.Tensor:
    """
    将 PMF（概率质量函数）转换为 CDF（累积分布函数）
    
    Args:
        pmf: 概率分布，shape (..., vocab_size)，应该已经归一化（sum=1）
    
    Returns:
        cdf: 累积分布函数，shape (..., vocab_size + 1)，最后一个值为 1.0
    """
    # 计算累积和（CDF）
    # 注意：如果 PMF 已经归一化，cumsum 的最后一个值应该接近 1.0
    cdf = torch.cumsum(pmf, dim=-1)
    
    # 确保 CDF 是单调递增的，并且最后一个值不超过 1.0
    # 由于浮点误差，最后一个值可能略小于 1.0，我们需要确保它是 1.0
    # 但不要 clamp 中间的值，因为这会破坏 CDF 的性质
    cdf = torch.clamp(cdf, max=1.0)
    
    # 添加 1.0 作为最后一个值（torchac 需要）
    # CDF 形状: (..., vocab_size + 1)
    # cdf = torch.cat([cdf, torch.ones_like(cdf[..., :1])], dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    return cdf


def compress_arithmetic_coding(
    indices: torch.Tensor,
    probabilities: torch.Tensor,
    vocab_size: int,
    store_pmf: bool = False  # 默认不存储PMF以减小文件大小
) -> Tuple[bytes, Dict]:
    """
    使用 numpyAc 库的算术编码压缩索引序列
    
    Args:
        indices: 索引张量，shape (batch_size, seq_len) 或 (1, seq_len)
        probabilities: 概率分布，shape (batch_size, seq_len, vocab_size)
        vocab_size: 词汇表大小
    
    Returns:
        (compressed_bytes, metadata): 压缩后的字节流和元数据
    """
    # 确保输入格式正确
    if indices.dim() == 1:
        indices = indices.unsqueeze(0)
    if probabilities.dim() == 2:
        probabilities = probabilities.unsqueeze(0)
    
    batch_size, seq_len = indices.shape
    device = indices.device
    
    # 归一化概率
    probs = _normalize_probs(probabilities)
    
    # 从 probabilities 的实际形状获取 vocab_size（更可靠）
    actual_vocab_size = probs.shape[-1]
    
    # 确保 indices 是整数类型且在有效范围内
    indices_int = indices.long()
    # 检查索引范围（使用实际的 vocab_size）
    if indices_int.min() < 0 or indices_int.max() >= actual_vocab_size:
        raise ValueError(f"索引值超出范围 [0, {actual_vocab_size-1}]: min={indices_int.min()}, max={indices_int.max()}")
    
    # 转换为 numpy 数组（numpyAc 需要 numpy 数组）
    # probs: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
    probs_np = probs.cpu().numpy().reshape(-1, actual_vocab_size)  # (batch_size * seq_len, vocab_size)
    indices_np = indices_int.cpu().numpy().reshape(-1).astype(np.int16)  # (batch_size * seq_len,)
    
    # 使用 numpyAc 进行编码
    codec = arithmeticCoding()
    compressed_bytes, real_bits = codec.encode(probs_np, indices_np, binfile=None)
    
    # 存储PMF用于解码（存储原始概率，不存储CDF，因为可以从PMF重建）
    # 注意：为了减小文件大小，可以选择不存储PMF，在解压时使用Transformer模型重新计算
    metadata = {
        'seq_len': seq_len,
        'vocab_size': actual_vocab_size,  # 使用实际的 vocab_size
        'device': str(device),
        'method': 'numpyAc',
        'store_pmf': store_pmf
    }
    
    # 只有在需要时才存储PMF（会显著增加文件大小）
    if store_pmf:
        metadata['pmf'] = probs.cpu().numpy().tolist()  # 存储为列表以便序列化
    
    return compressed_bytes, metadata


def decompress_arithmetic_coding(
    compressed_bytes: bytes,
    metadata: Dict,
    transformer_model=None,
    partial_indices=None
) -> torch.Tensor:
    """
    使用算术编码解压索引序列（兼容旧格式，使用 numpyAc）
    
    Args:
        compressed_bytes: 压缩后的字节流
        metadata: 元数据字典
        transformer_model: 可选的Transformer模型，用于在PMF不存在时重新计算概率分布
        partial_indices: 可选的部分已解码索引，用于逐步解码
    
    Returns:
        indices: 解压后的索引张量
    """
    # 检查是否是多个批次（从 recon_judge.py 的格式）
    if 'batch_size' in metadata:
        batch_size = metadata['batch_size']
        offset = 4
        all_indices = []
        
        for i in range(batch_size):
            seg_len = struct.unpack('<I', compressed_bytes[offset:offset+4])[0]
            offset += 4
            seg_bytes = compressed_bytes[offset:offset+seg_len]
            offset += seg_len
            
            seg_metadata = metadata['segments'][i]
            indices = decompress_arithmetic_coding(seg_bytes, seg_metadata, transformer_model, partial_indices)
            all_indices.append(indices)
        
        return torch.cat(all_indices, dim=0)
    
    # 单个批次
    method = metadata.get('method', 'numpyAc')
    if method not in ['numpyAc', 'torchac']:
        raise ValueError(f"不支持的算术编码解码方法: {method}")
    
    seq_len = metadata['seq_len']
    vocab_size = metadata['vocab_size']
    device_str = metadata.get('device', 'cpu')
    device = torch.device(device_str)
    
    # 检查是否有PMF，如果没有则使用Transformer模型重新计算
    store_pmf = metadata.get('store_pmf', True)  # 默认True以保持兼容性
    if 'pmf' in metadata and metadata['pmf'] is not None:
        # 使用存储的PMF
        pmf_list = metadata['pmf']
    pmf = torch.tensor(pmf_list, dtype=torch.float32, device=device)  # (seq_len, vocab_size)
    
    # 如果原始是 batch 格式，需要添加 batch 维度
    if pmf.dim() == 2:
        pmf = pmf.unsqueeze(0)  # (1, seq_len, vocab_size)
    
        # 转换为 numpy 数组
        pmf_np = pmf.cpu().numpy().reshape(-1, vocab_size)  # (seq_len, vocab_size)
    
        # 使用 numpyAc 进行解码
        decodec = arithmeticDeCoding(compressed_bytes, seq_len, vocab_size, binfile=None)
        decoded_list = []
        for i in range(seq_len):
            decoded_symbol = decodec.decode(pmf_np[i:i+1, :])  # 传入单个位置的 PDF
            decoded_list.append(decoded_symbol)
        
        decoded = torch.tensor(decoded_list, dtype=torch.long, device=device)
    elif transformer_model is not None:
        # 使用Transformer模型重新计算概率分布（需要逐步解码）
        raise ValueError(
            "metadata中没有PMF，请使用 decompress_arithmetic_coding_incremental 进行逐步解码。"
        )
    else:
        raise ValueError("metadata中没有PMF，且没有提供transformer_model。无法解压。")
    
    return decoded


def compress_arithmetic_coding_incremental(
    indices: torch.Tensor,
    transformer_model,
    device: str,
    vocab_size: int
) -> Tuple[bytes, Dict]:
    """
    逐步压缩算术编码（每次预测下一个index的概率，然后压缩）
    
    Args:
        indices: 索引张量 (seq_len,)
        transformer_model: Transformer模型，用于预测概率分布
        device: 设备
        vocab_size: 词汇表大小
    
    Returns:
        (compressed_bytes, metadata): 压缩后的字节流和元数据
    """
    # 确保indices在正确的设备上
    if isinstance(indices, torch.Tensor):
        indices = indices.to(device)
    else:
        indices = torch.tensor(indices, dtype=torch.long, device=device)
    
    seq_len = len(indices)
    bos_token_id = transformer_model.transformer.bos_token_id
    
    # 初始化上下文：BOS token
    current_context = torch.tensor([bos_token_id], dtype=torch.long, device=device)
    all_probs = []
    all_indices = []
    
    # 逐步压缩：每次预测下一个index的概率，然后压缩
    for pos in tqdm(range(seq_len), desc="逐步压缩"):
        # 使用 get_conditional_probs 获取下一个index的概率分布
        next_probs = transformer_model.transformer.get_conditional_probs(
            current_context, 
            temperature=1.0, 
            use_rope=True
        )  # (vocab_size,)
        
        # 归一化概率
        next_probs = _normalize_probs(next_probs.unsqueeze(0)).squeeze(0)  # (vocab_size,)
        
        # 获取当前要压缩的index
        current_index = indices[pos].item()
        
        # 存储概率和索引
        all_probs.append(next_probs.unsqueeze(0))  # (1, vocab_size)
        all_indices.append(torch.tensor([[current_index]], dtype=torch.long, device=device))  # (1, 1)
        
        # 将当前index添加到上下文（用于下一步预测）
        current_context = torch.cat([current_context, torch.tensor([current_index], dtype=torch.long, device=device)], dim=0)
    
    # 合并所有概率和索引
    all_probs_tensor = torch.cat(all_probs, dim=0).unsqueeze(0)  # (1, seq_len, vocab_size)
    all_indices_tensor = torch.cat(all_indices, dim=1)  # (1, seq_len)
    
    # 使用算术编码压缩
    compressed_bytes, metadata = compress_arithmetic_coding(
        all_indices_tensor,
        all_probs_tensor,
        vocab_size=vocab_size,
        store_pmf=False
    )
    
    # 更新metadata
    metadata['seq_len'] = seq_len
    metadata['incremental'] = True
    
    return compressed_bytes, metadata


def decompress_arithmetic_coding_incremental(
    compressed_bytes: bytes,
    metadata: Dict,
    transformer_model,
    device: str,
    original_indices: torch.Tensor = None  # 可选的原始indices，用于验证
) -> torch.Tensor:
    """
    逐步解码算术编码（每次预测下一个index的概率，然后解码）
    
    使用 numpyAc 的自回归解码功能，每次只传入当前位置的 PDF
    
    Args:
        compressed_bytes: 压缩后的字节流（完整的压缩数据）
        metadata: 元数据字典
        transformer_model: Transformer模型，用于预测概率分布
        device: 设备
        original_indices: 可选的原始indices，用于验证解码结果
    
    Returns:
        indices: 解压后的索引张量 (seq_len,)
    """
    seq_len = metadata['seq_len']
    vocab_size = metadata['vocab_size']
    bos_token_id = transformer_model.transformer.bos_token_id
    
    # 初始化 numpyAc 解码器
    decodec = arithmeticDeCoding(compressed_bytes, seq_len, vocab_size, binfile=None)
    
    # 初始化上下文：BOS token
    current_context = torch.tensor([bos_token_id], dtype=torch.long, device=device)
    all_decoded = []
    
    # 逐步解码：每次预测下一个index的概率，然后解码
    for pos in tqdm(range(seq_len), desc="逐步解码"):
        # 使用 get_conditional_probs 获取当前位置的概率分布
        next_probs = transformer_model.transformer.get_conditional_probs(
            current_context,
            temperature=1.0,
            use_rope=True
        )  # (vocab_size,)
        
        # 归一化概率
        next_probs = _normalize_probs(next_probs.unsqueeze(0)).squeeze(0)  # (vocab_size,)
        
        # 转换为 numpy 数组（numpyAc 需要 numpy 数组）
        next_probs_np = next_probs.cpu().numpy().reshape(1, vocab_size)  # (1, vocab_size)
        
        # 使用 numpyAc 解码当前位置（只传入当前位置的 PDF）
        decoded_index = decodec.decode(next_probs_np)
        
        # 转换为 torch tensor
        decoded_index = torch.tensor(decoded_index, dtype=torch.long, device=device)
        
        # 验证：如果提供了原始indices，比较解码的index和原始index
        if original_indices is not None and pos < len(original_indices):
            original_index = original_indices[pos].item() if isinstance(original_indices, torch.Tensor) else original_indices[pos]
            decoded_index_val = decoded_index.item()
            
            if original_index != decoded_index_val:
                print(f"  ⚠️ 位置 {pos}: 解码index={decoded_index_val}, 原始index={original_index}, 不一致!")
            elif pos < 10 or pos % 1000 == 0:
                print(f"  ✓ 位置 {pos}: 解码index={decoded_index_val}, 原始index={original_index}, 一致")
        
        # 将解码的index添加到上下文（用于下一步预测）
        current_context = torch.cat([current_context, decoded_index.unsqueeze(0)], dim=0)
        all_decoded.append(decoded_index)
    
    # 合并所有解码的indices
    decoded_indices = torch.stack(all_decoded, dim=0)  # (seq_len,)
    
    return decoded_indices


# 为了向后兼容，提供range_coding的别名（指向算术编码）
compress_range_coding = compress_arithmetic_coding
decompress_range_coding = decompress_arithmetic_coding

