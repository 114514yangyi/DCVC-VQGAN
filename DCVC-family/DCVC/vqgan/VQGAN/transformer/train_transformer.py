#!/usr/bin/env python3
"""
VQVAE2 Codebook Index 压缩 - Transformer 训练脚本

- 使用 VQVAE2 将训练/验证视频编码为 index（每帧 top+bottom 等层级拼接）
- Transformer 学习 index 序列的概率分布，用于估算压缩比特率
- 验证阶段（从第 2 个 epoch 起）生成 CSV：orig, recon, size（size 由 Transformer 概率估算 bytes，不用熵编码）
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_VQVAE2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _VQVAE2_ROOT not in sys.path:
    sys.path.insert(0, _VQVAE2_ROOT)

# 本地 data_prepare / recon_judge
from transformer import data_prepare
from transformer import recon_judge


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============== Transformer 模型（与 project 一致，不依赖 CompressAI）==============

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
        embeddings = embeddings.unsqueeze(0).unsqueeze(0).expand(batch_size, nums_head, -1, -1)
        embeddings = embeddings.reshape(batch_size, nums_head, max_len, output_dim)
        return embeddings

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
    def __init__(self, num_codes=512, max_seq_len=8192, d_model=256, nhead=8, num_layers=6, dropout=0, ff_hidden_dim=None):
        super().__init__()
        if ff_hidden_dim is None:
            ff_hidden_dim = d_model * 3
        self.bos_token_id = num_codes
        self.num_codes = num_codes
        self.vocab_size = num_codes + 1
        self.max_seq_len = max_seq_len + 1
        self.d_model = d_model
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
        for dec in self.decoder_blocks:
            x_emb = dec(x_emb, mask, use_rope=use_rope)
        logits = self.fc_out(x_emb) / temperature
        probabilities = F.softmax(logits, dim=-1)
        return logits, probabilities

    def get_conditional_probs(self, context_indices, temperature=1.0, use_rope=True):
        if context_indices.dim() == 1:
            context_indices = context_indices.unsqueeze(0)
        seq_len = context_indices.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=context_indices.device), diagonal=1)
        logits, probs = self.forward(context_indices, mask=mask, temperature=temperature, use_rope=use_rope)
        probs = probs[:, -1, :]
        if probs.size(0) == 1:
            probs = probs.squeeze(0)
        return probs


class TransformerEntropyModel(nn.Module):
    """包装 Transformer，forward 返回 (rate, probs)，rate 由 NLL 转 bits."""
    def __init__(self, num_codes=512, max_seq_len=8192, d_model=256, num_layers=6, nhead=8):
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
        input_logits = logits[:, :-1, :].contiguous()
        target_indices = indices[:, 1:].contiguous()
        logits_flat = input_logits.view(-1, logits.size(-1))
        indices_flat = target_indices.view(-1)
        nll = F.cross_entropy(logits_flat, indices_flat, reduction="mean")
        rate = nll / 0.6931471805599453
        if return_probs:
            return rate, probs
        return rate


# ============== 数据集 ==============

class VQIndexDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).long()


# ============== 训练 / 验证 / 压缩率估算 ==============

def train_epoch(model, loader, optimizer, device, epoch, grad_accum_steps=1, use_amp=False):
    model.train()
    total_loss = total_rate = total_acc = 0.0
    num_batches = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]")
    for batch_idx, indices in enumerate(pbar):
        indices = indices.to(device, non_blocking=True)
        bos = model.transformer.bos_token_id
        bos_col = torch.full((indices.size(0), 1), bos, dtype=torch.long, device=device)
        full = torch.cat([bos_col, indices], dim=1)
        input_seq = full[:, :-1]
        target_seq = full[:, 1:]
        seq_len = input_seq.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits, probs = model.transformer(input_seq, mask=mask, use_rope=True)
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_seq.reshape(-1)
            nll = F.cross_entropy(logits_flat, target_flat, reduction="mean")
            rate = nll / 0.6931471805599453
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == target_seq).float().mean()
            loss = rate / grad_accum_steps
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * grad_accum_steps
        total_rate += rate.item()
        total_acc += acc.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}", rate=f"{total_rate/num_batches:.4f}", acc=f"{total_acc/num_batches:.4f}")
    return total_loss / num_batches, total_rate / num_batches, total_acc / num_batches


def validate(model, loader, device):
    model.eval()
    total_loss = total_rate = total_acc = 0.0
    num_batches = 0
    with torch.no_grad():
        for indices in tqdm(loader, desc="[val]"):
            indices = indices.to(device)
            bos = model.transformer.bos_token_id
            bos_col = torch.full((indices.size(0), 1), bos, dtype=torch.long, device=device)
            full = torch.cat([bos_col, indices], dim=1)
            input_seq = full[:, :-1]
            target_seq = full[:, 1:]
            seq_len = input_seq.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            logits, probs = model.transformer(input_seq, mask=mask, use_rope=True)
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_seq.reshape(-1)
            nll = F.cross_entropy(logits_flat, target_flat, reduction="mean")
            rate = nll / 0.6931471805599453
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == target_seq).float().mean()
            total_loss += rate.item()
            total_rate += rate.item()
            total_acc += acc.item()
            num_batches += 1
    return total_loss / num_batches, total_rate / num_batches, total_acc / num_batches


def evaluate_compression_rate(model, loader, device, vocab_size):
    """
    用 Transformer 概率估算平均 bits/index（不调用熵编码器）。
    返回 (avg_bits, compression_ratio, fixed_bits)。
    """
    model.eval()
    total_bits = 0.0
    total_indices = 0
    bos_id = model.transformer.bos_token_id
    with torch.no_grad():
        for indices in loader:
            indices = indices.to(device)
            bos_col = torch.full((indices.size(0), 1), bos_id, dtype=torch.long, device=device)
            full = torch.cat([bos_col, indices], dim=1)
            input_seq = full[:, :-1]
            seq_len = input_seq.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            _, probs = model.transformer(input_seq, mask=mask, use_rope=True)
            # probs (B, seq_len, num_codes) 对应预测 indices
            target = indices
            p = torch.gather(probs, 2, target.unsqueeze(-1)).squeeze(-1)
            eps = 1e-10
            bits = (-torch.log2(p + eps)).sum().item()
            total_bits += bits
            total_indices += indices.numel()
    avg_bits = total_bits / total_indices if total_indices else 0.0
    fixed_bits = np.log2(vocab_size)
    compression_ratio = fixed_bits / avg_bits if avg_bits > 0 else 0.0
    return avg_bits, compression_ratio, fixed_bits


def main():
    parser = argparse.ArgumentParser(description="VQVAE2 Index Transformer 训练")
    parser.add_argument("--train_data_dir", type=str, required=True, help="训练视频目录")
    parser.add_argument("--val_data_dir", type=str, required=True, help="验证视频目录")
    parser.add_argument("--vq_config", type=str, default=None, help="VQVAE2 配置，默认 VQVAE2/config.json")
    parser.add_argument("--vq_ckpt", type=str, required=True, help="VQVAE2 checkpoint")
    parser.add_argument("--sequence_length", type=int, default=8, help="每段帧数")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=512, help="codebook 大小，需与 VQVAE2 nb_entries 一致")
    parser.add_argument("--seq_len", type=int, default=None, help="由数据自动得到时可省略")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--train_num_samples", type=int, default=None)
    parser.add_argument("--val_num_samples", type=int, default=None)
    parser.add_argument("--max_train_size", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--recon_output_dir", type=str, default=None, help="验证重构视频目录，默认 output_dir/recon")
    parser.add_argument("--original_output_dir", type=str, default=None, help="验证原视频裁剪目录，默认 output_dir/original")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = args.recon_output_dir or str(output_dir / "recon")
    orig_dir = args.original_output_dir or str(output_dir / "original")

    if args.vq_config is None:
        args.vq_config = os.path.join(_VQVAE2_ROOT, "config.json")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # 训练集 index 数据
    print("加载训练集 index...")
    train_data = data_prepare.get_data_injection(
        data_dir=args.train_data_dir,
        seq_len=args.seq_len,
        vq_config=args.vq_config,
        vq_ckpt=args.vq_ckpt,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        device=args.device,
        num_samples=args.train_num_samples,
        max_size=args.max_train_size,
    )
    seq_len = train_data.shape[1]
    if args.seq_len is not None and args.seq_len != seq_len:
        print(f"警告: 数据 seq_len={seq_len} 与 --seq_len={args.seq_len} 不一致，以数据为准")
    args.seq_len = seq_len

    # 验证集：带重构与 meta（用于 CSV）
    print("准备验证集（编码 + 重构 + meta）...")
    val_data, val_meta_list = data_prepare.prepare_validation_with_recon(
        val_video_dir=args.val_data_dir,
        vq_config=args.vq_config,
        vq_ckpt=args.vq_ckpt,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        device=args.device,
        recon_output_dir=recon_dir,
        original_output_dir=orig_dir,
        max_val_videos=args.val_num_samples,
    )

    train_dataset = VQIndexDataset(train_data)
    val_dataset = VQIndexDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    print(f"训练样本: {len(train_dataset)}, 验证样本(段): {len(val_dataset)}, seq_len={seq_len}, vocab_size={args.vocab_size}")

    model = TransformerEntropyModel(
        num_codes=args.vocab_size,
        max_seq_len=seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"从 {args.resume} 恢复，epoch {start_epoch}")

    best_val_rate = float("inf")
    best_model_path = output_dir / "best_model.pth"
    train_history = {"loss": [], "rate": [], "acc": [], "val_loss": [], "val_rate": [], "val_acc": []}

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_rate, train_acc = train_epoch(
            model, train_loader, optimizer, device, epoch,
            grad_accum_steps=args.grad_accum_steps, use_amp=args.use_amp,
        )
        val_loss, val_rate, val_acc = validate(model, val_loader, device)
        scheduler.step(val_loss)

        train_history["loss"].append(train_loss)
        train_history["rate"].append(train_rate)
        train_history["acc"].append(train_acc)
        train_history["val_loss"].append(val_loss)
        train_history["val_rate"].append(val_rate)
        train_history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}  train loss={train_loss:.4f} rate={train_rate:.4f} acc={train_acc:.4f}  val rate={val_rate:.4f} acc={val_acc:.4f}")

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

        # 从第 2 个 epoch 开始，验证时生成 CSV（orig, recon, size 用 Transformer 概率估算）
        if epoch >= 2 and val_meta_list:
            csv_path = recon_judge.run_validation_csv(
                model, val_meta_list, device, str(output_dir),
                csv_name=f"video_comparison_epoch{epoch}.csv",
            )
            print(f"  生成 CSV: {csv_path}")

    # 最终评估：用概率估算压缩率
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False)["model_state_dict"])
    avg_bits, compression_ratio, fixed_bits = evaluate_compression_rate(model, val_loader, device, args.vocab_size)
    print("最终评估（概率估算）:")
    print(f"  平均 bits/index: {avg_bits:.4f}, 固定长度: {fixed_bits:.4f}, 压缩比: {compression_ratio:.2f}:1")

    # 训练曲线
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_history["loss"], label="train")
    plt.plot(train_history["val_loss"], label="val")
    plt.legend()
    plt.title("Loss")
    plt.subplot(1, 3, 2)
    plt.plot(train_history["rate"], label="train")
    plt.plot(train_history["val_rate"], label="val")
    plt.axhline(y=fixed_bits, color="r", linestyle="--", label=f"fixed {fixed_bits:.2f}")
    plt.legend()
    plt.title("Rate (bits/index)")
    plt.subplot(1, 3, 3)
    plt.plot(train_history["acc"], label="train")
    plt.plot(train_history["val_acc"], label="val")
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()

    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "train_history": train_history,
        "avg_bits_per_index": avg_bits,
        "compression_ratio": compression_ratio,
        "fixed_bits_per_index": fixed_bits,
    }, output_dir / "final_model.pth")
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "vocab_size": args.vocab_size,
            "seq_len": seq_len,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "avg_bits_per_index": avg_bits,
            "compression_ratio": compression_ratio,
        }, f, indent=2)
    print("训练完成. 输出目录:", output_dir)


if __name__ == "__main__":
    main()
