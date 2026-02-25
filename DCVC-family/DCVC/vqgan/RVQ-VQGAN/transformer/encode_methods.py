"""
VQGAN Transformer 压缩 - 编码相关

本模块不使用熵模型实际压缩后看大小，而是用 Transformer 输出的概率估算压缩所需 bytes：
  bits = -log2(p) 对每个 index 求和，bytes = ceil(bits / 8)。

实际算术编码（ANS/range_coding）可在此扩展；评估阶段 CSV 中的 size 仅使用概率估算。
"""

import numpy as np
import torch


def estimate_compressed_bytes_from_probs(
    probs: torch.Tensor,
    indices: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    由 Transformer 输出的概率估算压缩后字节数（不进行实际熵编码）。
    probs: [B, L, num_codes]，indices: [B, L]（或 [B, L+1] 含 BOS 时自动取 [:, 1:]）。
    返回每个样本的估算 bytes（float，可再取 ceil 得到整数）。
    """
    if indices.shape[1] == probs.shape[1] + 1:
        indices = indices[:, 1:]
    assert probs.shape[:2] == indices.shape[:2]
    indices_flat = indices.long().clamp(0, probs.size(-1) - 1)
    p = torch.gather(probs, dim=-1, index=indices_flat.unsqueeze(-1)).squeeze(-1)
    nll_bits = -torch.log2(p.clamp(min=eps))
    bits_per_sample = nll_bits.sum(dim=1)
    return bits_per_sample / 8.0
