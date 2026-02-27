#!/usr/bin/env python3
"""
VQGAN Codebook Index 压缩 - Transformer 训练脚本

- 使用训练集视频经 VQGAN 编码得到 index，训练 Transformer 学习 index 分布以压缩存储。
- 验证集：VQGAN 生成 index 的同时生成重构视频并保存；验证阶段（从第 2 个 epoch 起）生成 CSV，
  表中压缩大小由 Transformer 输出概率估算（bits = -log2(p)，bytes = bits/8），不使用熵编码实际压缩。
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_VQGAN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _VQGAN_ROOT not in sys.path:
    sys.path.insert(0, _VQGAN_ROOT)

from transformer import data_prepare

LN2 = 0.6931471805599453


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------- 1. Dataset ---------------


class VQIndexDataset(Dataset):
    """VQGAN codebook index 数据集，data 形状 [N, seq_len] int64。"""
    def __init__(self, data: np.ndarray):
        self.data = np.asarray(data, dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).long()


class ValIndexDataset(Dataset):
    """验证集：返回 (indices, meta)，便于评估时按样本写 CSV（概率估算 bytes）。"""
    def __init__(self, data: np.ndarray, meta: list):
        self.data = np.asarray(data, dtype=np.int64)
        self.meta = list(meta)
        assert len(self.data) == len(self.meta)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).long(), self.meta[idx]


# --------------- 2. Transformer 概率模型（RoPE + Decoder-only）-------------------


class MultiheadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size
        self.values = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        self.keys = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        self.queries = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def sinusoidal_position_embedding(self, batch_size, nums_head, max_len, output_dim, device):
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(-1)
        ids = torch.arange(0, output_dim // 2, dtype=torch.float, device=device)
        theta = torch.pow(10000, -2 * ids / output_dim)
        embeddings = position * theta
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))
        return embeddings.reshape(batch_size, nums_head, max_len, output_dim)

    def RoPE(self, q, k):
        B, H, L, D = q.shape
        pos_emb = self.sinusoidal_position_embedding(B, H, L, D, q.device)
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape)
        q = q * cos_pos + q2 * sin_pos
        k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape)
        k = k * cos_pos + k2 * sin_pos
        return q, k

    def forward(self, x, mask, use_rope=True):
        B, L, _ = x.shape
        values = self.values(x).view(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)
        keys = self.keys(x).view(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)
        queries = self.queries(x).view(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)
        if use_rope:
            queries, keys = self.RoPE(queries, keys)
        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        if mask is not None:
            mask_bool = mask if mask.dtype == torch.bool else (mask == 1)
            energy = energy.masked_fill(mask_bool, torch.finfo(energy.dtype).min)
        attention = F.softmax(energy / (self.head_dim ** 0.5), dim=-1)
        out = torch.matmul(attention, values).permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        return self.fc_out(out)


class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc2 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc3 = nn.Linear(ff_hidden_dim, embed_size)

    def forward(self, x):
        gate = torch.sigmoid(self.fc1(x))
        transformed = torch.relu(self.fc2(x))
        return self.fc3(gate * transformed)


class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_dim, dropout=0):
        super().__init__()
        self.attention = MultiheadAttention(embed_size, heads)
        self.feed_forward = FeedForward(embed_size, ff_hidden_dim)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, use_rope=True):
        x = x + self.dropout(self.attention(self.norm1(x), mask, use_rope=use_rope))
        return x + self.dropout(self.feed_forward(self.norm2(x)))


class TransformerProbabilityModel(nn.Module):
    """Decoder-only Transformer + RoPE，BOS=num_codes，输出 num_codes 维概率。"""
    def __init__(self, num_codes=1024, max_seq_len=16384, d_model=256, nhead=8, num_layers=6,
                 dropout=0, ff_hidden_dim=None):
        super().__init__()
        self.bos_token_id = num_codes
        self.num_codes = num_codes
        self.vocab_size = num_codes + 1
        self.max_seq_len = max_seq_len + 1
        self.d_model = d_model
        if ff_hidden_dim is None:
            ff_hidden_dim = d_model * 3
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.decoder_blocks = nn.ModuleList(
            [DecoderLayer(d_model, nhead, ff_hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, self.num_codes)
        self.dropout = nn.Dropout(dropout)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None, temperature=1.0, use_rope=True):
        batch_size, seq_len = x.size()
        x = torch.clamp(x, 0, self.vocab_size - 1)
        x_emb = self.embedding(x)
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        for block in self.decoder_blocks:
            x_emb = block(x_emb, mask, use_rope=use_rope)
        logits = self.fc_out(x_emb) / temperature
        probabilities = F.softmax(logits, dim=-1)
        return logits, probabilities

    def get_conditional_probs(self, context_indices, temperature=1.0, use_rope=True):
        if context_indices.dim() == 1:
            context_indices = context_indices.unsqueeze(0)
        _, seq_len = context_indices.shape
        mask = torch.triu(torch.ones(seq_len, seq_len, device=context_indices.device), diagonal=1)
        logits, probs = self.forward(context_indices, mask=mask, temperature=temperature, use_rope=use_rope)
        probs = probs[:, -1, :]
        if probs.shape[0] == 1:
            probs = probs.squeeze(0)
        return probs


class TransformerEntropyModel(nn.Module):
    """包装 Transformer 概率模型，forward 返回 (rate, probs)，率由 NLL 转 bits。"""
    def __init__(self, num_codes=1024, max_seq_len=16384, d_model=256, num_layers=6, nhead=8):
        super().__init__()
        self.transformer = TransformerProbabilityModel(
            num_codes=num_codes, max_seq_len=max_seq_len, d_model=d_model,
            nhead=nhead, num_layers=num_layers
        )
        self.num_codes = num_codes
        self.vocab_size = num_codes + 1
        self.bos_token_id = num_codes

    def forward(self, indices, return_probs=True):
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)
        batch_size, seq_len = indices.shape
        mask = torch.triu(torch.ones(seq_len, seq_len, device=indices.device), diagonal=1)
        logits, probs = self.transformer(indices, mask=mask, use_rope=True)
        input_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        target_indices = indices[:, 1:].contiguous().view(-1)
        nll = F.cross_entropy(input_logits, target_indices, reduction="mean")
        rate = nll / LN2
        if return_probs:
            return rate, probs
        return rate


# --------------- 3. 基于概率估算压缩大小（不用熵编码）---------------


def estimate_bits_from_probs(probs: torch.Tensor, indices: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    probs: [B, seq_len, num_codes], indices: [B, seq_len] 或 [B, seq_len+1]（含 BOS）。
    若 probs 对应预测位置 0..seq_len-1，则 indices 应为目标 [d0..d_{L-1}]，即 indices 与 probs 对齐：
    probs[b,t] 预测的是 indices[b,t]。
    返回每个样本的总 bits（标量或 per-sample）。
    """
    if indices.shape[1] == probs.shape[1] + 1:
        indices = indices[:, 1:]
    assert probs.shape[:2] == indices.shape[:2]
    indices_flat = indices.long().clamp(0, probs.size(-1) - 1)
    p = torch.gather(probs, dim=-1, index=indices_flat.unsqueeze(-1)).squeeze(-1)
    nll_bits = -torch.log2(p.clamp(min=eps))
    return nll_bits.sum(dim=1)


def evaluate_compression_rate_by_probs(model, dataloader, device, vocab_size: int):
    """
    用 Transformer 输出概率估算压缩率，不做实际熵编码。
    返回: (avg_bits_per_index, compression_ratio, fixed_bits_per_index)
    """
    model.eval()
    fixed_bits = np.log2(vocab_size)
    total_bits = 0.0
    total_indices = 0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                indices = batch[0].to(device)
            else:
                indices = batch.to(device)
            B, L = indices.shape
            bos = torch.full((B, 1), model.bos_token_id, dtype=torch.long, device=device)
            full = torch.cat([bos, indices], dim=1)
            mask = torch.triu(torch.ones(L + 1, L + 1, device=device), diagonal=1)
            _, probs = model.transformer(full, mask=mask, use_rope=True)
            probs = probs[:, :-1, :]
            bits_per_sample = estimate_bits_from_probs(probs, indices)
            total_bits += bits_per_sample.sum().item()
            total_indices += B * L
    if total_indices == 0:
        return fixed_bits, 1.0, fixed_bits
    avg_bits = total_bits / total_indices
    ratio = fixed_bits / avg_bits if avg_bits > 0 else 1.0
    return avg_bits, ratio, fixed_bits


def evaluate_and_write_csv(
    model,
    val_loader: DataLoader,
    device,
    csv_path: str,
    vocab_size: int,
):
    """
    按验证集逐样本用 Transformer 概率估算压缩 bytes，按 (original_path, recon_path) 聚合后写 CSV。
    表头: orig, recon, size（size 为估算的 bytes，不经过熵编码实际压缩）。
    """
    model.eval()
    agg = defaultdict(lambda: 0.0)
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="估算压缩大小并写 CSV"):
            indices, meta_batch = batch
            indices = indices.to(device)
            B, L = indices.shape
            bos = torch.full((B, 1), model.bos_token_id, dtype=torch.long, device=device)
            full = torch.cat([bos, indices], dim=1)
            mask = torch.triu(torch.ones(L + 1, L + 1, device=device), diagonal=1)
            _, probs = model.transformer(full, mask=mask, use_rope=True)
            probs = probs[:, :-1, :]
            bits_per_sample = estimate_bits_from_probs(probs, indices)
            for i in range(B):
                if isinstance(meta_batch, dict):
                    m = {k: (v[i] if hasattr(v, "__getitem__") else v) for k, v in meta_batch.items()}
                else:
                    m = meta_batch[i] if isinstance(meta_batch, (list, tuple)) else meta_batch
                orig = m.get("original_path") or m.get("orig", "")
                recon = m.get("recon_path") or m.get("recon", "")
                bits = bits_per_sample[i].item()
                bytes_est = max(0, int(np.ceil(bits / 8.0)))
                key = (orig, recon)
                agg[key] += bytes_est
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["orig", "recon", "size"])
        for (orig, recon), size in sorted(agg.items()):
            w.writerow([orig, recon, size])
    print(f"CSV 已写入: {csv_path}（size 为 Transformer 概率估算 bytes）")


# --------------- 4. 训练与验证 ---------------


def train_epoch(model, train_loader, optimizer, device, epoch, grad_accum_steps=1, use_amp=False):
    model.train()
    total_loss = total_rate = total_acc = 0.0
    num_batches = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [训练]")
    for batch_idx, indices in enumerate(pbar):
        indices = indices.to(device, non_blocking=True)
        B, L = indices.shape
        bos = torch.full((B, 1), model.bos_token_id, dtype=torch.long, device=device)
        full = torch.cat([bos, indices], dim=1)
        input_seq = full[:, :-1]
        target_seq = full[:, 1:]
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits, probs = model.transformer(input_seq, mask=mask, use_rope=True)
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_seq.reshape(-1)
            nll = F.cross_entropy(logits_flat, target_flat, reduction="mean")
            rate = nll / LN2
            preds = logits.argmax(dim=-1)
            acc = (preds == target_seq).float().mean()
            loss = rate / grad_accum_steps
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * grad_accum_steps
        total_rate += rate.item()
        total_acc += acc.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}", rate=f"{total_rate/num_batches:.4f}", acc=f"{total_acc/num_batches:.4f}")
    return total_loss / num_batches, total_rate / num_batches, total_acc / num_batches


def validate(model, val_loader, device, vocab_size: int):
    """验证：返回 (avg_loss, avg_rate, avg_acc)。"""
    model.eval()
    total_loss = total_rate = total_acc = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[验证]"):
            if isinstance(batch, (list, tuple)):
                indices = batch[0].to(device)
            else:
                indices = batch.to(device)
            B, L = indices.shape
            bos = torch.full((B, 1), model.bos_token_id, dtype=torch.long, device=device)
            full = torch.cat([bos, indices], dim=1)
            input_seq = full[:, :-1]
            target_seq = full[:, 1:]
            mask = torch.triu(torch.ones(L, L, device=device), diagonal=1)
            logits, probs = model.transformer(input_seq, mask=mask, use_rope=True)
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_seq.reshape(-1)
            nll = F.cross_entropy(logits_flat, target_flat, reduction="mean")
            rate = nll / LN2
            preds = logits.argmax(dim=-1)
            acc = (preds == target_seq).float().mean()
            total_loss += rate.item()
            total_rate += rate.item()
            total_acc += acc.item()
            num_batches += 1
    n = max(num_batches, 1)
    return total_loss / n, total_rate / n, total_acc / n


# --------------- 5. 主流程 ---------------


def load_config(config_path: str) -> dict:
    """从JSON配置文件加载配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def merge_config_with_args(config: dict, args: argparse.Namespace) -> argparse.Namespace:
    """将配置文件中的参数合并到args中，命令行参数优先（如果命令行参数不是None）"""
    # 模型参数
    if "model" in config:
        model_cfg = config["model"]
        if args.vocab_size is None:
            args.vocab_size = model_cfg.get("vocab_size", 1024)
        if args.d_model is None:
            args.d_model = model_cfg.get("d_model", 512)
        if args.num_layers is None:
            args.num_layers = model_cfg.get("num_layers", 4)
    
    # 数据参数
    if "data" in config:
        data_cfg = config["data"]
        if args.seq_len is None:
            args.seq_len = data_cfg.get("seq_len", 8192)
        if args.sequence_length is None:
            args.sequence_length = data_cfg.get("sequence_length", 8)
        if args.image_size is None:
            args.image_size = data_cfg.get("image_size", 256)
        if args.train_data_dir is None or args.train_data_dir == "":
            args.train_data_dir = data_cfg.get("train_data_dir", "")
        if args.val_data_dir is None or args.val_data_dir == "":
            args.val_data_dir = data_cfg.get("val_data_dir", "")
        if args.train_num_samples is None:
            args.train_num_samples = data_cfg.get("train_num_samples", None)
        if args.val_num_samples is None:
            args.val_num_samples = data_cfg.get("val_num_samples", None)
    
    # VQGAN参数
    if "vqgan" in config:
        vq_cfg = config["vqgan"]
        if args.vq_config is None or args.vq_config == "":
            args.vq_config = vq_cfg.get("vq_config", "")
        if args.vq_ckpt is None or args.vq_ckpt == "":
            args.vq_ckpt = vq_cfg.get("vq_ckpt", "")
    
    # 训练参数
    if "training" in config:
        train_cfg = config["training"]
        if args.batch_size is None:
            args.batch_size = train_cfg.get("batch_size", 1)
        if args.grad_accum_steps is None:
            args.grad_accum_steps = train_cfg.get("grad_accum_steps", 4)
        # use_amp 处理：如果命令行未指定（None），则使用配置文件的值
        if args.use_amp is None:
            args.use_amp = train_cfg.get("use_amp", False)
        if args.epochs is None:
            args.epochs = train_cfg.get("epochs", 100)
        if args.lr is None:
            args.lr = train_cfg.get("lr", 1e-4)
        if args.seed is None:
            args.seed = train_cfg.get("seed", 42)
        if args.resume is None:
            resume_val = train_cfg.get("resume", None)
            args.resume = resume_val if resume_val else None
        if args.val_start_epoch is None:
            args.val_start_epoch = train_cfg.get("val_start_epoch", 2)
        args.weight_decay = train_cfg.get("weight_decay", 1e-4)
    
    # 输出参数
    if "output" in config:
        output_cfg = config["output"]
        if args.output_dir is None:
            args.output_dir = output_cfg.get("output_dir", "./runs/transformer")
        if args.recon_output_dir is None or args.recon_output_dir == "":
            args.recon_output_dir = output_cfg.get("recon_output_dir", "")
        if args.original_output_dir is None or args.original_output_dir == "":
            args.original_output_dir = output_cfg.get("original_output_dir", "")
    
    # 设备参数
    if "device" in config:
        if args.device is None:
            args.device = config.get("device", "cuda:0")
    
    return args


def main():
    parser = argparse.ArgumentParser(description="VQGAN index Transformer 压缩训练")
    parser.add_argument("--config", type=str, default="", help="配置文件路径（JSON格式）")
    parser.add_argument("--vocab_size", type=int, default=None, help="VQGAN codebook 大小")
    parser.add_argument("--seq_len", type=int, default=None, help="序列长度")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--use_amp", action="store_true", help="是否使用混合精度训练")
    parser.add_argument("--no_use_amp", dest="use_amp", action="store_false", help="不使用混合精度训练")
    parser.set_defaults(use_amp=None)  # 默认None，表示未指定
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--train_num_samples", type=int, default=None)
    parser.add_argument("--val_num_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vq_config", type=str, default=None, help="VQGAN 配置路径")
    parser.add_argument("--vq_ckpt", type=str, default=None, help="VQGAN checkpoint 路径")
    parser.add_argument("--train_data_dir", type=str, default=None, help="训练视频目录")
    parser.add_argument("--val_data_dir", type=str, default=None, help="验证视频目录")
    parser.add_argument("--sequence_length", type=int, default=None, help="每样本帧数")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--recon_output_dir", type=str, default=None, help="验证重构视频输出目录")
    parser.add_argument("--original_output_dir", type=str, default=None, help="验证裁剪原视频输出目录")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--val_start_epoch", type=int, default=None, help="从该 epoch 开始验证并写 CSV")
    args = parser.parse_args()
    
    # 如果指定了配置文件，则加载并合并配置
    if args.config:
        config = load_config(args.config)
        # 合并配置（命令行参数优先）
        args = merge_config_with_args(config, args)
        print(f"已从配置文件加载参数: {args.config}")
    else:
        # 如果没有指定配置文件，使用默认值
        defaults = {
            "vocab_size": 1024,
            "seq_len": 8192,
            "batch_size": 1,
            "grad_accum_steps": 4,
            "use_amp": False,
            "epochs": 100,
            "lr": 1e-4,
            "d_model": 512,
            "num_layers": 4,
            "train_num_samples": None,
            "val_num_samples": None,
            "output_dir": "./runs/transformer",
            "device": "cuda:0",
            "seed": 42,
            "vq_config": "",
            "vq_ckpt": "",
            "train_data_dir": "",
            "val_data_dir": "",
            "sequence_length": 8,
            "image_size": 256,
            "recon_output_dir": "",
            "original_output_dir": "",
            "resume": None,
            "val_start_epoch": 2,
            "weight_decay": 1e-4,
        }
        for key, default_val in defaults.items():
            if key == "use_amp":
                # use_amp 特殊处理：如果命令行未指定（None），使用默认值
                if args.use_amp is None:
                    args.use_amp = default_val
            else:
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, default_val)

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("设备:", device)

    vq_config = args.vq_config or os.getenv("VQ_CONFIG", "")
    vq_ckpt = args.vq_ckpt or os.getenv("VQ_CKPT", "")
    if not vq_config or not vq_ckpt:
        raise ValueError("请提供 --vq_config / --vq_ckpt 或设置 VQ_CONFIG / VQ_CKPT")
    if not args.train_data_dir:
        raise ValueError("请提供 --train_data_dir")

    # 训练集 index 数据
    print("生成训练集 index 数据...")
    train_data = data_prepare.get_data_injection(
        data_dir=args.train_data_dir,
        seq_len=args.seq_len,
        vq_config=vq_config,
        vq_ckpt=vq_ckpt,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        device=args.device,
        num_samples=args.train_num_samples,
    )
    if train_data.size == 0:
        raise ValueError("训练集为空")
    actual_seq_len = train_data.shape[1]
    if args.seq_len != actual_seq_len:
        print(f"使用实际 seq_len={actual_seq_len}（与 --seq_len {args.seq_len} 不同则已忽略）")
    seq_len = actual_seq_len

    train_dataset = VQIndexDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # 验证集：带重构视频与 meta
    val_dataset = None
    val_loader = None
    val_meta = []
    if args.val_data_dir:
        recon_dir = args.recon_output_dir or str(output_dir / "recon")
        orig_dir = args.original_output_dir or str(output_dir / "original")
        print("生成验证集 index 与重构视频...")
        val_data, val_meta = data_prepare.process_val_videos_to_indices_and_recon(
            video_dir=args.val_data_dir,
            vq_config=vq_config,
            vq_ckpt=vq_ckpt,
            sequence_length=args.sequence_length,
            image_size=args.image_size,
            device=args.device,
            recon_output_dir=recon_dir,
            original_output_dir=orig_dir,
            num_samples=args.val_num_samples,
        )
        def _collate_val(batch):
            indices = torch.stack([b[0] for b in batch])
            metas = [b[1] for b in batch]
            return indices, metas

        val_dataset = ValIndexDataset(val_data, val_meta)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=_collate_val,
        )
        print(f"验证集样本数: {len(val_dataset)}")
    else:
        print("未提供 --val_data_dir，仅训练不验证/不写 CSV")

    model = TransformerEntropyModel(
        num_codes=args.vocab_size,
        max_seq_len=seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    weight_decay = getattr(args, "weight_decay", 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"从 epoch {start_epoch} 恢复")

    train_history = {"loss": [], "rate": [], "acc": [], "val_loss": [], "val_rate": [], "val_acc": []}
    best_val_rate = float("inf")
    best_model_path = output_dir / "best_model.pth"
    fixed_bits = np.log2(args.vocab_size)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_rate, train_acc = train_epoch(
            model, train_loader, optimizer, device, epoch,
            grad_accum_steps=args.grad_accum_steps, use_amp=args.use_amp,
        )
        train_history["loss"].append(train_loss)
        train_history["rate"].append(train_rate)
        train_history["acc"].append(train_acc)

        if val_loader is not None:
            val_loss, val_rate, val_acc = validate(model, val_loader, device, args.vocab_size)
            train_history["val_loss"].append(val_loss)
            train_history["val_rate"].append(val_rate)
            train_history["val_acc"].append(val_acc)
            scheduler.step(val_loss)
            print(f"Epoch {epoch} 训练 loss={train_loss:.4f} rate={train_rate:.4f} acc={train_acc:.4f}")
            print(f"        验证 loss={val_loss:.4f} rate={val_rate:.4f} acc={val_acc:.4f}")

            if val_rate <= best_val_rate:
                best_val_rate = val_rate
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_rate": val_rate,
                    "train_history": train_history,
                    "args": vars(args),
                }, best_model_path)
                print(f"  保存最佳模型 (val_rate={val_rate:.4f})")

            if epoch >= args.val_start_epoch:
                csv_path = output_dir / f"video_comparison_epoch{epoch}.csv"
                evaluate_and_write_csv(model, val_loader, device, str(csv_path), args.vocab_size)
        else:
            scheduler.step(train_loss)
            print(f"Epoch {epoch} 训练 loss={train_loss:.4f} rate={train_rate:.4f} acc={train_acc:.4f}")

    if val_loader is not None and best_model_path.exists():
        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        avg_bits, ratio, _ = evaluate_compression_rate_by_probs(model, val_loader, device, args.vocab_size)
        print("最终评估（概率估算）:")
        print(f"  平均 bits/index: {avg_bits:.4f}, 固定: {fixed_bits:.4f}, 压缩比: {ratio:.2f}:1")
        if val_meta:
            csv_final = output_dir / "video_comparison.csv"
            evaluate_and_write_csv(model, val_loader, device, str(csv_final), args.vocab_size)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_history["loss"], label="训练损失")
    if train_history["val_loss"]:
        plt.plot(train_history["val_loss"], label="验证损失")
    plt.legend()
    plt.xlabel("Epoch")
    plt.subplot(1, 3, 2)
    plt.plot(train_history["rate"], label="训练率")
    if train_history["val_rate"]:
        plt.plot(train_history["val_rate"], label="验证率")
    plt.axhline(y=fixed_bits, color="r", linestyle="--", label=f"固定 {fixed_bits:.2f} bits")
    plt.legend()
    plt.xlabel("Epoch")
    plt.subplot(1, 3, 3)
    plt.plot(train_history["acc"], label="训练准确率")
    if train_history["val_acc"]:
        plt.plot(train_history["val_acc"], label="验证准确率")
    plt.legend()
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()
    print("训练曲线已保存:", output_dir / "training_curves.png")

    config = {
        "vocab_size": args.vocab_size,
        "seq_len": seq_len,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "model_params": total_params,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("训练完成. 输出目录:", output_dir)


if __name__ == "__main__":
    main()
