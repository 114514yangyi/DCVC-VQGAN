"""
VQVAE2 + Transformer 验证评估

- 使用 Transformer 输出的概率估算压缩后字节数（不调用熵编码器）
- 生成 CSV：orig, recon, size（size = 估算的 bytes）
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np

_VQVAE2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _VQVAE2_ROOT not in sys.path:
    sys.path.insert(0, _VQVAE2_ROOT)


def estimate_compressed_bytes_from_probs(
    model,
    indices: torch.Tensor,
    device: torch.device,
    bos_token_id: int,
) -> float:
    """
    用 Transformer 对整段 index 做一次前向，取每个位置真实 index 的概率 p，
    估算比特数 = sum(-log2(p))，字节数 = bits / 8。

    indices: (seq_len,) 或 (1, seq_len)，不含 BOS
    """
    if indices.dim() == 1:
        indices = indices.unsqueeze(0)
    indices = indices.to(device)
    seq_len = indices.size(1)
    bos = torch.full((indices.size(0), 1), bos_token_id, dtype=torch.long, device=device)
    full = torch.cat([bos, indices], dim=1)  # (1, seq_len+1)
    input_seq = full[:, :-1]
    mask = torch.triu(torch.ones(seq_len + 1, seq_len + 1, device=device), diagonal=1)
    with torch.no_grad():
        logits, probs = model.transformer(input_seq, mask=mask, use_rope=True)
    # probs: (1, seq_len+1, num_codes)；预测的是下一个位置，所以 probs[:, i, :] 对应 target full[:, i+1]
    # 即 probs[:, :-1, :] 对应 target indices
    probs = probs[:, :-1, :]  # (1, seq_len, num_codes)
    indices_1 = indices.unsqueeze(-1)  # (1, seq_len, 1)
    p = torch.gather(probs, 2, indices_1).squeeze(-1)  # (1, seq_len)
    eps = 1e-10
    bits = (-torch.log2(p + eps)).sum().item()
    return bits / 8.0


def run_validation_csv(
    model,
    val_meta_list: List[Dict[str, Any]],
    device: torch.device,
    output_dir: str,
    csv_name: str = "video_comparison.csv",
) -> str:
    """
    对 val_meta_list 中每个视频的 indices_segments 用 Transformer 估算字节数，
    写 CSV：orig, recon, size（估算 bytes）。

    val_meta_list 每项: original_path, recon_path, indices_segments (list of (seq_len,) array)
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, csv_name)
    bos_token_id = model.transformer.bos_token_id

    rows = []
    for meta in val_meta_list:
        orig_path = meta.get("original_path", "")
        recon_path = meta.get("recon_path", "")
        segments = meta.get("indices_segments", [])
        total_bytes = 0.0
        for seg in segments:
            seg_t = torch.from_numpy(seg).long()
            total_bytes += estimate_compressed_bytes_from_probs(
                model, seg_t, device, bos_token_id
            )
        size_int = int(np.ceil(total_bytes))
        rows.append((os.path.abspath(orig_path), os.path.abspath(recon_path), size_int))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["orig", "recon", "size"])
        for r in rows:
            writer.writerow(list(r))
    return csv_path
