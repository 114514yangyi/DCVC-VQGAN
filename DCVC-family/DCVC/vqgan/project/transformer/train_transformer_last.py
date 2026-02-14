#!/usr/bin/env python3

"""

VQ-VAE Codebook Index 压缩 - Transformer 训练脚本

使用 CompressAI 框架进行快速原型开发



安装依赖:

pip install torch torchvision compressai numpy tqdm matplotlib



用法:

python train_transformer_compression.py --vocab_size 512 --seq_len 256 --epochs 50

"""

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import numpy as np

import random

import argparse

import time

import os

from pathlib import Path

from tqdm import tqdm

import matplotlib.pyplot as plt


# CompressAI 相关导入

try:
    import compressai

    from compressai.entropy_models import EntropyModel

    from compressai.layers import GDN

    print(f"✓ CompressAI 版本: {compressai.__version__}")

except ImportError:
    print("请先安装 CompressAI: pip install compressai")
    exit(1)
# 导入自定义编码方法

try:
    from transformer.encode_methods import create_compress_injection, create_decompress_injection

    print("✓ 成功导入 encode_methods")

except ImportError:
    try:
        from encode_methods import create_compress_injection, create_decompress_injection

        print("✓ 成功导入 encode_methods")
    except ImportError as e:
        print(f"⚠ 无法导入 encode_methods: {e}")
        print("  将使用默认的 NotImplementedError 实现")
# 尝试导入 data_prepare

try:
    from transformer.data_prepare import get_data_injection

except ImportError:
    try:
        from data_prepare import get_data_injection

    except ImportError:
        get_data_injection = None

        print("警告: 无法导入 data_prepare，将使用模拟数据")
# 设置随机种子

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False



set_seed(42)
# ============================================================================
# 1. 自定义数据集：模拟 VQ-VAE 的 codebook index

# ============================================================================

class VQIndexDataset(Dataset):
    """
    模拟 VQ-VAE codebook index 的数据集

    在实际应用中，这里应该加载真实的 VQ-VAE 输出索引

    """
    def __init__(self, num_samples=10000, vocab_size=512, seq_len=256, 
                 spatial_correlation=True, data_injection=None,max_size=100000):
        """
        参数:
            num_samples: 样本数量

            vocab_size: codebook 大小 (默认 512)
            seq_len: 序列长度 (默认 256 = 16x16)
            spatial_correlation: 是否模拟空间相关性

            data_injection: 数据注入函数，如果提供则使用真实数据

        """
        self.num_samples = num_samples

        self.vocab_size = vocab_size

        self.seq_len = seq_len

        self.spatial_correlation = spatial_correlation

        
        # 如果提供了 data_injection，使用真实数据

        if data_injection is not None:
            # data_injection 现在直接返回所有数据数组

            self.data = data_injection
            print(len(self.data))
            print(self.data.shape)

            # # 确保数据是 2D 数组 [num_samples, seq_len]
            # if self.data.ndim == 1:
            #     # 如果是 1D，假设只有一个样本

            #     self.data = self.data.reshape(1, -1)
            # elif self.data.ndim > 2:
            #     # 如果是更高维，展平除第一维外的所有维度

            #     self.data = self.data.reshape(len(self.data), -1)
            
            # # 限制样本数量

            # if num_samples is not None and num_samples < len(self.data):
            #     self.data = self.data[:num_samples]
            
            self.num_samples = len(self.data)
            print(f"使用真实数据，加载了 {self.num_samples} 个样本")
        else:
            # 生成模拟数据
            exit(0)
            self.data = self._generate_data()
    
    def _generate_data(self):
        """生成具有空间相关性的模拟索引数据"""
        data = []
        
        for _ in range(self.num_samples):
            if self.spatial_correlation:
                # 方法1: 模拟具有空间相关性的索引（更真实）
                # 创建一个基础模式，然后添加噪声

                base_pattern = np.random.randint(0, self.vocab_size // 8, (16, 16))
                
                # 使用双线性插值创建平滑变化

                try:
                    from scipy import ndimage

                    smoothed = ndimage.gaussian_filter(base_pattern.astype(float), sigma=1.0)
                except ImportError:
                    # 如果没有 scipy，使用简单的平滑

                    smoothed = base_pattern.astype(float)
                
                # 添加一些高频细节

                noise = np.random.randint(-5, 6, (16, 16))
                indices_2d = np.clip(smoothed + noise, 0, self.vocab_size - 1).astype(int)
                
                # 展平为序列（光栅顺序）

                indices_seq = indices_2d.flatten()
            else:
                # 方法2: 简单随机索引

                indices_seq = np.random.randint(0, self.vocab_size, self.seq_len)
            
            data.append(indices_seq)
        
        return np.array(data, dtype=np.int64)
    
    def __len__(self):
        return self.num_samples

    
    def __getitem__(self, idx):
        indices = self.data[idx]
        return torch.from_numpy(indices).long()
# ============================================================================
# 2. Transformer 概率模型

# ============================================================================

# ============================================================================
# 从参考代码导入 RoPE 相关的注意力机制
# ============================================================================

class MultiheadAttention(nn.Module):
    """多头注意力机制，支持 RoPE (Rotary Positional Encoding)"""
    def __init__(self, embed_size, heads):
        super(MultiheadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        self.keys = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        self.queries = nn.Linear(embed_size, self.head_dim * heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def sinusoidal_position_embedding(self, batch_size, nums_head, max_len, output_dim, device):
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
        ids = torch.arange(0, output_dim // 2, dtype=torch.float)
        theta = torch.pow(10000, -2 * ids / output_dim)

        embeddings = position * theta

        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))

        embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
        embeddings = embeddings.to(device)
        return embeddings

    def RoPE(self, q, k):
        # q,k: (B, H, L, D)
        batch_size = q.shape[0]
        nums_head = q.shape[1]
        max_len = q.shape[2]
        # max_len=4096
        output_dim = q.shape[-1]

        pos_emb = self.sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)

        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

        # q,k: (B, H, L, D)
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
        q2 = q2.reshape(q.shape)
        q = q * cos_pos + q2 * sin_pos

        k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
        k2 = k2.reshape(k.shape)
        k = k * cos_pos + k2 * sin_pos

        return q, k

    def forward(self, x, mask, use_rope=True):
        B = x.shape[0]
        len_seq = x.shape[1]

        values = self.values(x).view(B, len_seq, self.heads, self.head_dim)
        keys = self.keys(x).view(B, len_seq, self.heads, self.head_dim)
        queries = self.queries(x).view(B, len_seq, self.heads, self.head_dim)

        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        # [B, H, L, D]

        if use_rope:
            queries, keys = self.RoPE(queries, keys)

        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        if mask is not None:
            # NOTE: 在 mixed precision (fp16) 下，-1e20 无法转换为 half，会触发 overflow。
            # 用 energy.dtype 可表示的最小值进行 mask，等价于在 softmax 前把概率压到 0。
            mask_bool = mask if mask.dtype == torch.bool else (mask == 1)
            energy = energy.masked_fill(mask_bool, torch.finfo(energy.dtype).min)

        attention = F.softmax(energy / (self.head_dim ** (1 / 2)), dim=-1)

        out = torch.matmul(attention, values)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, len_seq, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out


class FeedForward(nn.Module):
    """SwiGLU 前馈网络"""
    def __init__(self, embed_size, ff_hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc2 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc3 = nn.Linear(ff_hidden_dim, embed_size)

    def forward(self, x):
        gate = torch.sigmoid(self.fc1(x))
        transformed = torch.relu(self.fc2(x))
        output = gate * transformed
        return self.fc3(output)


class DecoderLayer(nn.Module):
    """Decoder 层，包含多头注意力和前馈网络"""
    def __init__(self, embed_size, heads, ff_hidden_dim, dropout=0):
        super(DecoderLayer, self).__init__()
        self.attention = MultiheadAttention(embed_size, heads)
        self.feed_forward = FeedForward(embed_size, ff_hidden_dim)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, use_rope=True):
        attention = self.attention(self.norm1(x), mask, use_rope=use_rope)
        x = self.dropout(attention) + x
        forward = self.feed_forward(self.norm2(x))
        out = self.dropout(forward) + x
        return out

class TransformerProbabilityModel(nn.Module):
    """
    Transformer 自回归概率模型（基于 Decoder-only Transformer with RoPE）
    
    关键特性：
    - 使用BOS token（值为num_codes）作为序列开头
    - vocab_size = num_codes + 1（包含BOS）
    - 使用 RoPE (Rotary Positional Encoding) 替代传统位置编码
    - 使用 SwiGLU 前馈网络
    """
    def __init__(self, num_codes=1024, max_seq_len=16384, d_model=256, 
                 nhead=8, num_layers=6, dropout=0, ff_hidden_dim=None):
        super().__init__()
        self.bos_token_id = num_codes  # BOS token的值等于num_codes
        self.num_codes = num_codes
        self.vocab_size = num_codes + 1  # 包含BOS的词汇表大小
        self.max_seq_len = max_seq_len + 1  # 包含BOS
        self.d_model = d_model
        
        # 如果没有指定 ff_hidden_dim，使用默认值（通常是 d_model 的 1.5-4 倍）
        if ff_hidden_dim is None:
            ff_hidden_dim = d_model * 3  # 参考代码中使用 384 对于 128 的 embed_size
        
        # Token嵌入层（用于离散token）
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        
        # Decoder 层
        self.decoder_blocks = nn.ModuleList(
            [DecoderLayer(d_model, nhead, ff_hidden_dim, dropout) for _ in range(num_layers)]
        )
        
        # 输出层：预测每个位置的概率分布（只输出 num_codes，不包含BOS）
        self.fc_out = nn.Linear(d_model, self.num_codes)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None, temperature=1.0, use_rope=True):
        """
        前向传播：计算每个位置的条件概率分布
        
        参数:
            x: 输入索引序列 [batch, seq_len]（应该包含BOS token）
            mask: 因果掩码 [seq_len, seq_len]（如果为None，会自动生成）
            temperature: 温度参数，控制分布平滑度
            use_rope: 是否使用 RoPE
            
        返回:
            logits: [batch, seq_len, num_codes]
            probabilities: [batch, seq_len, num_codes]
        """
        batch_size, seq_len = x.size()
        
        # 0. 输入验证和修复：确保索引在有效范围内（避免 CUDA 错误）
        x = torch.clamp(x, 0, self.vocab_size - 1)
        
        # 1. Token嵌入
        x_emb = self.embedding(x)  # [batch, seq_len, d_model]
        
        # 2. 如果没有提供mask，生成因果掩码（参考 train_trans.py）
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(x.device)
        
        # 3. 通过 Decoder 层
        for decoder in self.decoder_blocks:
            x_emb = decoder(x_emb, mask, use_rope=use_rope)
        
        # 4. 输出投影（只输出 num_codes，不包含BOS）
        logits = self.fc_out(x_emb) / temperature  # [batch, seq_len, num_codes]
        
        # 5. 计算概率（softmax）
        probabilities = F.softmax(logits, dim=-1)  # [batch, seq_len, num_codes]
        
        return logits, probabilities

    
    def get_conditional_probs(self, context_indices, temperature=1.0, sequence_length=4096, use_rope=True):
        """
        获取给定上下文时下一个符号的条件概率
        
        参数:
            context_indices: 上下文索引 [batch, context_len] 或 [context_len]
                            （必须以BOS开头）
            temperature: 温度参数
            sequence_length: 目标序列长度（不包含BOS），用于填充
            use_rope: 是否使用 RoPE
            
        返回:
            probs: 条件概率分布 [batch, num_codes] 或 [num_codes]
                   （只包含0到num_codes-1的概率，不包含BOS）
        """
        if len(context_indices.shape) == 1:
            context_indices = context_indices.unsqueeze(0)  # 添加 batch 维度

        batch_size, context_len = context_indices.shape
        
        # 生成因果掩码
        mask = torch.triu(torch.ones(context_len, context_len), diagonal=1).to(context_indices.device)
        
        # 前向传播获取logits
        logits, probs = self.forward(context_indices, mask=mask, temperature=temperature, use_rope=use_rope)
        

        probs = probs[:, -1, :]
        # 取最后一个位置的logits（即要预测的位置）
        # last_logits = logits[:, -1, :]  # [batch, num_codes]
        
        # # 计算概率
        # probs = F.softmax(last_logits / temperature, dim=-1)  # [batch, num_codes]
        
        if probs.shape[0] == 1:
            probs = probs.squeeze(0)  # 移除 batch 维度 -> [num_codes]
        
        return probs
    def get_conditional_probs_by_index(self, context_indices, index,temperature=1.0, sequence_length=4096, use_rope=True):
        if len(context_indices.shape) == 1:
            context_indices = context_indices.unsqueeze(0)  # 添加 batch 维度

        batch_size, context_len = context_indices.shape
        
        # 生成因果掩码
        mask = torch.triu(torch.ones(context_len, context_len), diagonal=1).to(context_indices.device)
        
        # 前向传播获取logits
        logits, probs = self.forward(context_indices, mask=mask, temperature=temperature, use_rope=use_rope)
        
        # 取第index位置的logits（即要预测的位置）
        # 我要获得第index位置的数据,那么我就要从BOS开始算起应该是index位置的预测值
        # 而因为我logits输出的是对应id的下一个id的预测值,所以这里取的是index
        probs = probs[:, index, :]  # [batch, num_codes]
        
        # 计算概率
        # probs = F.softmax(last_logits / temperature, dim=-1)  # [batch, num_codes]
        
        if probs.shape[0] == 1:
            probs = probs.squeeze(0)  # 移除 batch 维度 -> [num_codes]
        
        return probs



# ============================================================================
# 3. 基于 CompressAI 的熵模型包装器

# ============================================================================

class TransformerEntropyModel(EntropyModel):
    """
    将 Transformer 概率模型包装为 CompressAI 熵模型（支持BOS token）

    继承自 compressai.entropy_models.EntropyModel

    """
    def __init__(self, num_codes=1024, max_seq_len=16384, d_model=256, num_layers=4,
                 likelihood_type="gaussian", quant_mode="noise",
                 compress_injection=None, decompress_injection=None):
        """
        参数:
            num_codes: VQGAN codebook 大小（0到num_codes-1）
            max_seq_len: 原始序列最大长度（不包含BOS）
            d_model: Transformer 隐藏维度
            num_layers: Transformer 层数
            likelihood_type: 似然类型（实际使用分类分布，保留用于兼容性）
            quant_mode: 量化模式
            compress_injection: 压缩注入函数
            decompress_injection: 解压注入函数

        """
        # EntropyModel 的初始化参数：likelihood_bound, entropy_coder, entropy_coder_precision
        super().__init__()
        
        # Transformer 概率模型（内部会处理BOS token）
        self.transformer = TransformerProbabilityModel(
            num_codes=num_codes,
            max_seq_len=max_seq_len,
            d_model=d_model,
            num_layers=num_layers
        )
        
        self.num_codes = num_codes
        self.vocab_size = num_codes + 1  # 包含BOS
        self.bos_token_id = num_codes
        self.quant_mode = quant_mode
        
        # 用于概率量化的参数
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.compress_injection = compress_injection
        self.decompress_injection = decompress_injection

        
    def forward(self, indices, return_probs=True):
        """
        训练阶段的前向传播（支持BOS token）
        
        参数:
            indices: 输入索引 [batch, seq_len]（应该包含BOS token在开头）
            return_probs: 是否返回概率分布
            
        返回:
            如果 return_probs=True: 返回 (rate, probs)
            否则: 返回 rate（作为率损失）
        """
        # 允许既传入 [seq_len] 也传入 [batch, seq_len]，统一转为二维
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)
        batch_size, seq_len = indices.shape

        # 生成因果掩码
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(indices.device)
        
        # 获取所有位置的概率分布
        # Transformer 使用因果掩码，所以每个位置的 logits 已经是条件概率
        logits, probs = self.transformer(indices, mask=mask, use_rope=True)

        # 计算损失：输入是 [BOS, d0, ..., d254]，目标是 [d0, d1, ..., d255]
        # 所以需要shift：logits[:, :-1] 对应 targets[:, 1:]
        input_logits = logits[:, :-1, :].contiguous()  # [batch, seq_len-1, num_codes]
        target_indices = indices[:, 1:].contiguous()  # [batch, seq_len-1]
        
        # 将 logits 和 indices 展平以计算交叉熵
        logits_flat = input_logits.view(-1, logits.size(-1))  # [batch * (seq_len-1), num_codes]
        indices_flat = target_indices.view(-1)  # [batch * (seq_len-1)]
        
        # 计算交叉熵损失（负对数似然）
        nll_loss = F.cross_entropy(logits_flat, indices_flat, reduction='mean')
        
        # 率损失就是平均负对数似然（单位：nats，转换为 bits 需要除以 ln(2)）
        # ln(2) ≈ 0.693
        rate = nll_loss / 0.6931471805599453  # 转换为 bits
        
        # 如果需要概率，则返回 (rate, probs) 方便损失函数使用
        if return_probs:
            return rate, probs

        return rate

    
    def compress(self, indices):
        """
        压缩：将索引序列压缩为比特流

        使用注入函数或默认实现

        """
        if self.compress_injection is not None:
            return self.compress_injection(indices)
        raise NotImplementedError(
            "完整压缩实现需要算术编码器。"
            "在实际应用中，应使用 constriction 或 range_coder 库。"
        )
    
    def decompress(self, strings, shape):
        """
        解压：将比特流恢复为索引序列

        使用注入函数或默认实现

        """
        if self.decompress_injection is not None:
            return self.decompress_injection(strings, shape)
        raise NotImplementedError(
            "完整解压实现需要算术解码器。"
            "在实际应用中，应使用 constriction 或 range_coder 库。"
        )
    
    def _get_conditionals(self, indices):
        """获取每个位置的条件概率（用于压缩/解压）

        
        注意：由于 Transformer 使用因果掩码，forward 返回的 probs[t] 
        已经是 p(x_t | x_<t) 的条件概率，无需循环计算

        """
        # 直接使用 forward 返回的概率，避免重复计算

        _, probs = self.forward(indices, return_probs=True)
        # probs: [batch, seq_len, vocab_size]
        return probs


# ============================================================================
# 4. 训练函数和评估指标

# ============================================================================

class RateDistortionLoss(nn.Module):
    """率失真损失（这里只有率损失，因为是无损压缩）"""
    def __init__(self, lmbda=0.01):
        super().__init__()
        self.lmbda = lmbda

    
    def forward(self, output, target):
        """
        参数:
            output: 模型输出 (rate, probs)
            target: 目标索引（包含BOS token，用于计算准确率）

        返回:
            loss: 率损失
            rate: 率损失
            accuracy: 准确率（只计算非BOS位置的准确率）
        """
        rate = output[0] 
        
        # 计算准确率（辅助指标）
        probs = output[1]  # [batch, seq_len, vocab_size]
        preds = torch.argmax(probs, dim=-1)  # [batch, seq_len]
        
        # 只计算非BOS位置的准确率（target[:, 1:] 对应 preds[:, :-1]）
        # 因为输入是 [BOS, d0, ..., d254]，输出预测是 [d0, d1, ..., d255]
        target_shifted = target[:, 1:]  # [batch, seq_len-1]
        preds_shifted = preds[:, :-1]  # [batch, seq_len-1]
        accuracy = (preds_shifted == target_shifted).float().mean()
        
        # 总损失 = 率损失
        loss = rate
        
        return {
            'loss': loss,
            'rate': rate,
            'accuracy': accuracy
        }

def train_epoch(model, dataloader, criterion, optimizer, device, epoch,
                grad_accum_steps: int = 1, use_amp: bool = False,
                last_seq: int = 0):
    """训练一个 epoch，支持梯度累积与混合精度"""
    model.train()
    total_loss = 0
    total_rate = 0
    total_acc = 0
    num_batches = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f'Epoch {epoch:03d} [训练]')
    for batch_idx, indices in enumerate(pbar):
        indices = indices.to(device, non_blocking=True)  # [batch, seq_len]
        
        # === 数据预处理: 增加 BOS ===
        bos_token_id = model.transformer.bos_token_id
        bos_column = torch.full((indices.size(0), 1), bos_token_id, dtype=torch.long, device=device)
        full_sequence = torch.cat([bos_column, indices], dim=1)  # [batch, seq_len+1]
        
        input_seq = full_sequence[:, :-1]  # [batch, seq_len]
        target_seq = full_sequence[:, 1:]   # [batch, seq_len]
        
        seq_len = input_seq.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(device)
        
        # 前向 + 反向（支持 AMP）
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            logits, probs = model.transformer(input_seq, mask=mask, use_rope=True)  # [batch, seq_len, num_codes]
            # 仅对末尾 last_seq 个位置计算损失；0 表示使用全部
            tail = last_seq if last_seq and last_seq > 0 else logits.size(1)
            tail = min(tail, logits.size(1))
            logits_tail = logits[:, -tail:, :]  # [batch, tail, vocab]
            target_tail = target_seq[:, -tail:]  # [batch, tail]
            logits_flat = logits_tail.reshape(-1, logits_tail.size(-1))  # [batch * tail, vocab]
            target_flat = target_tail.reshape(-1)  # [batch * tail]
            nll_loss = F.cross_entropy(logits_flat, target_flat, reduction='mean')
            rate = nll_loss / 0.6931471805599453  # bits
            preds = torch.argmax(logits_tail, dim=-1)  # [batch, tail]
            accuracy = (preds == target_tail).float().mean()
            loss = rate / grad_accum_steps  # 梯度累积需缩放

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度累积：每 grad_accum_steps 更新一次
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

        # 统计（用未除 grad_accum 的 rate/acc）
        total_loss += loss.item() * grad_accum_steps
        total_rate += rate.item()
        total_acc += accuracy.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{total_loss/num_batches:.4f}',
            'rate': f'{total_rate/num_batches:.4f}',
            'acc': f'{total_acc/num_batches:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_rate = total_rate / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_rate, avg_acc




def validate(model, dataloader, criterion, device, args):
    """验证（参考 train_trans.py 的方式）"""
    model.eval()
    total_loss = 0.0
    total_rate = 0.0
    total_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for indices in tqdm(dataloader, desc='[验证]'):
            indices = indices.to(device)  # [batch, seq_len]
            
            # === 数据预处理: 增加 BOS ===
            bos_token_id = model.transformer.bos_token_id
            bos_column = torch.full((indices.size(0), 1), bos_token_id, dtype=torch.long, device=device)
            full_sequence = torch.cat([bos_column, indices], dim=1)  # [batch, seq_len+1]
            
            # === 参考 train_trans.py: 输入是 inputs[:, :-1]，目标是 inputs[:, 1:] ===
            input_seq = full_sequence[:, :-1]  # [batch, seq_len]
            target_seq = full_sequence[:, 1:]   # [batch, seq_len]
            
            # 生成因果掩码
            seq_len = input_seq.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(device)
            
            # 前向传播
            logits, probs = model.transformer(input_seq, mask=mask, use_rope=True)  # [batch, seq_len, num_codes]
            
            # 仅对末尾 last_seq 个位置计算损失；0 表示使用全部
            tail = args.last_seq if args.last_seq and args.last_seq > 0 else logits.size(1)
            tail = min(tail, logits.size(1))
            logits_tail = logits[:, -tail:, :]  # [batch, tail, vocab]
            target_tail = target_seq[:, -tail:]  # [batch, tail]
            logits_flat = logits_tail.reshape(-1, logits_tail.size(-1))  # [batch * tail, vocab]
            target_flat = target_tail.reshape(-1)  # [batch * tail]
            
            nll_loss = F.cross_entropy(logits_flat, target_flat, reduction='mean')
            rate = nll_loss / 0.6931471805599453  # 转换为 bits
            
            # 计算准确率
            preds = torch.argmax(logits_tail, dim=-1)  # [batch, tail]
            accuracy = (preds == target_tail).float().mean()
            
            loss = rate

            # 统计
            total_loss += loss.item()
            total_rate += rate.item()
            total_acc += accuracy.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_rate = total_rate / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_rate, avg_acc




# ============================================================================
# 5. 主训练脚本

# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='训练 Transformer 压缩 VQ-VAE 索引')
    parser.add_argument('--vocab_size', type=int, default=1024, help='codebook 大小')
    parser.add_argument('--seq_len', type=int, default=8192, help='序列长度 (16x16=256)')
    parser.add_argument('--batch_size', type=int, default=1, help='批大小')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                       help='梯度累积步数，用于在有限显存下模拟更大batch')
    parser.add_argument('--use_amp', action='store_true',
                       help='启用混合精度训练以节省显存')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--d_model', type=int, default=512, help='Transformer 隐藏维度')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer 层数')
    parser.add_argument('--num_samples', type=int, default=100, help='训练样本数')
    parser.add_argument('--last_seq', type=int, default=0,
                        help='仅在损失中使用末尾 last_seq 个位置（0 表示使用全部）')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--output_dir', type=str, default='/data/huyang/vqgan_result/64/transformer', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda:5', help='设备 (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--compression_method', type=str, default='ans', 
                       choices=['ans', 'range_coding', 'bitpacking'],
                       help='压缩方法 (ans, range_coding, bitpacking)')
    
    # 数据路径参数

    parser.add_argument('--data_dir', type=str, default='/data/huyang/data/test',
                       help='包含 .npy 文件的目录路径（用于加载真实的 VQ codebook index 数据，如果未指定 train_data_dir 和 val_data_dir 则使用此路径）')
    parser.add_argument('--train_data_dir', type=str, default="/data/huyang/data/test",
                       help='训练集数据目录路径（包含 .npy 文件），如果指定则使用此路径而不是 data_dir')
    parser.add_argument('--val_data_dir', type=str, default="/data/huyang/data/test",
                       help='验证集数据目录路径（包含 .npy 文件），如果指定则使用此路径而不是从 data_dir 随机划分')
    parser.add_argument('--train_num_samples', type=int, default=None,
                       help='训练集样本数量（如果为 None，则使用所有可用数据）')
    parser.add_argument('--val_num_samples', type=int, default=None,
                       help='验证集样本数量（如果为 None，则使用所有可用数据）')
    parser.add_argument('--resume', type=str, default=None,
                        help='从指定路径的检查点继续训练（例如: outputs/best_model.pth）')
    parser.add_argument('--test_only',type=bool,default=False,help='是否只测试，如果为 True，则不训练')
    parser.add_argument('--save_dir', type=str, default="/data/huyang/vqgan_result/64/transformer",
                        help='模型保存目录（如果指定，则模型将保存到此目录而不是 output_dir）')
    
    args = parser.parse_args()
    
    # 保存 val_data_dir 的引用，用于后续清理判断
    original_val_data_dir = args.val_data_dir
    
    # 设置随机种子

    set_seed(args.seed)
    
    # 创建输出目录

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 设备设置

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")



    model = TransformerEntropyModel(
        num_codes=args.vocab_size,  # VQGAN codebook大小
        max_seq_len=args.seq_len,  # 原始序列长度（不包含BOS）
        d_model=args.d_model,
        num_layers=args.num_layers,
        # compress_injection=compress_injection,
        # decompress_injection=decompress_injection

    ).to(device)
    
    # ========================================================================
    # 创建数据集

    # ========================================================================
    print(f"\n{'='*60}")
    print("创建数据集")
    print('='*60)
    
    # 确定数据目录

    train_data_dir = args.train_data_dir 

    val_data_dir = args.val_data_dir 

    
    # 创建数据注入函数

    train_data_injection_fn = None

    if get_data_injection is not None:
        try:
            train_data_injection_fn = get_data_injection(data_dir=train_data_dir, seq_len=args.seq_len)
            print(f"✓ 训练集使用真实数据: {train_data_dir}")
        except Exception as e:
            print(f"⚠ 加载训练集真实数据失败: {e}")
            print("  将使用模拟数据")
            train_data_injection_fn = None

    else:
        print("⚠ 未导入 data_prepare，训练集将使用模拟数据")
    
    val_data_injection_fn = None

    if get_data_injection is not None and val_data_dir != train_data_dir:
        try:
            val_data_injection_fn = get_data_injection(data_dir=val_data_dir, seq_len=args.seq_len)
            print(f"✓ 验证集使用真实数据: {val_data_dir}")
        except Exception as e:
            print(f"⚠ 加载验证集真实数据失败: {e}")
            print("  将使用与训练集相同的数据注入函数")
            val_data_injection_fn = train_data_injection_fn

    elif val_data_dir == train_data_dir:
        val_data_injection_fn = train_data_injection_fn

        print(f"✓ 验证集使用与训练集相同的数据路径: {val_data_dir}")
    else:
        print("⚠ 未导入 data_prepare，验证集将使用模拟数据")
    
    # 创建完整数据集

    train_num_samples = args.train_num_samples if args.train_num_samples is not None else args.num_samples

    train_dataset = VQIndexDataset(
        num_samples=train_num_samples,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        spatial_correlation=True,
        data_injection=train_data_injection_fn,
        max_size=1000

    )
    
    # ========== 清理数据生成过程中的临时数据 ==========
    # 数据已经加载到数据集中，可以清理数据注入函数中的临时数据
    if train_data_injection_fn is not None:
        print("清理训练数据生成过程中的临时数据...")
        # 如果 data_injection 返回的是 numpy 数组，数据集已经复制了数据
        # 这里主要是清理引用，让 Python 的垃圾回收器可以回收内存
        train_data_injection_fn = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("训练数据临时数据清理完成")
    
    # 判断是否使用相同的数据源
    # 由于 data_injection 现在返回 numpy 数组，不能直接比较数组
    # 应该比较数据目录路径来判断是否使用相同数据源
    use_same_data_source = (val_data_dir == train_data_dir)
    
    if use_same_data_source:
        # 从训练集划分验证集


        val_dataset = train_dataset
        # print(f"从训练集中划分验证集: 训练集 {train_size} 样本, 验证集 {val_size} 样本")
    else:
        # 创建独立验证集

        val_num_samples = args.val_num_samples if args.val_num_samples is not None else int(args.num_samples * args.val_ratio)
        val_dataset = VQIndexDataset(
            num_samples=val_num_samples,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            spatial_correlation=True,
            data_injection=val_data_injection_fn

        )
        print(f"创建独立验证集: {len(val_dataset)} 样本")
    
    # 创建数据加载器

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False

    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False

    )
    
    print(f"训练集大小: {len(train_dataset)}")
    # print(f"验证集大小: {len(val_dataset)}")
    print(f"词汇表大小: {args.vocab_size}")
    print(f"序列长度: {args.seq_len}")  
    print(f"固定长度编码: {np.log2(args.vocab_size):.2f} bits/index")
    
    # ========================================================================
    # 创建模型（使用注入的压缩/解压函数）

    # ========================================================================
    print(f"\n{'='*60}")
    print("创建模型")
    print('='*60)
    
    # 创建注入函数（如果可用）

    compress_injection = None

    decompress_injection = None

    
    # try:
    #     from transformer.encode_methods import create_compress_injection, create_decompress_injection

    #     compress_injection = create_compress_injection(
    #         method=args.compression_method, 
    #         vocab_size=args.vocab_size

    #     )
    #     decompress_injection = create_decompress_injection(
    #         method=args.compression_method

    #     )
    #     print(f"✓ 使用 {args.compression_method} 压缩方法")
    # except ImportError as e:
    #     print(f"⚠ 无法导入 encode_methods: {e}")
    #     print("  将使用默认的 NotImplementedError 实现")
    # except Exception as e:
    #     print(f"⚠ 创建注入函数失败: {e}")
    #     print("  将使用默认的 NotImplementedError 实现")
    
    
    
    # 计算模型参数

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型架构: Transformer ({args.num_layers} 层, {args.d_model} 隐藏维度)")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 如果指定了 --resume，加载预训练模型
    start_epoch = 1
    if args.resume is not None and os.path.exists(args.resume):
        print(f"\n{'='*60}")
        print(f"从检查点加载模型: {args.resume}")
        print('='*60)
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            
            # 加载模型权重
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✓ 模型权重加载成功")
            else:
                # 如果没有 'model_state_dict'，尝试直接加载（可能是旧格式）
                model.load_state_dict(checkpoint)
                print("✓ 模型权重加载成功（旧格式）")
            
            # 获取起始epoch（如果存在）
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"✓ 将从第 {start_epoch} 轮继续训练")
            
            # 打印检查点信息
            if 'val_rate' in checkpoint:
                print(f"  检查点验证率: {checkpoint['val_rate']:.4f}")
            if 'args' in checkpoint:
                print(f"  检查点参数: vocab_size={checkpoint['args'].get('vocab_size', 'N/A')}, "
                      f"seq_len={checkpoint['args'].get('seq_len', 'N/A')}, "
                      f"d_model={checkpoint['args'].get('d_model', 'N/A')}")
        except Exception as e:
            print(f"⚠ 加载检查点失败: {e}")
            print("  将从头开始训练")
            start_epoch = 1
    elif args.resume is not None:
        print(f"⚠ 警告: 指定的检查点文件不存在: {args.resume}")
        print("  将从头开始训练")
    
    best_model_path = output_dir / 'best_model.pth'
    if not args.test_only:
        train(args, device, model, train_loader, val_loader, output_dir, best_model_path, 
          compress_injection, decompress_injection, total_params, start_epoch=start_epoch)
    # test_sample(args, device, train_loader, model, best_model_path)



def train(args, device, model, train_loader, val_loader, output_dir, best_model_path,
          compress_injection, decompress_injection, total_params, start_epoch=1):
    """
    训练函数
    
    参数:
        args: 命令行参数
        device: 设备
        model: Transformer模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        output_dir: 输出目录
        best_model_path: 最佳模型保存路径
        compress_injection: 压缩注入函数
        decompress_injection: 解压注入函数
        total_params: 模型总参数数量
        start_epoch: 起始训练轮数（用于继续训练）
    """
    # ========================================================================
    # 训练设置

    # ========================================================================
    criterion = RateDistortionLoss(lmbda=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率调度器

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5

    )
    
    # ========================================================================
    # 加载检查点（如果继续训练）
    # ========================================================================
    train_history = {
        'loss': [], 'rate': [], 'acc': [],
        'val_loss': [], 'val_rate': [], 'val_acc': []
    }
    best_val_rate = float('inf')
    
    if args.resume is not None and os.path.exists(args.resume) and start_epoch > 1:
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            
            # 加载优化器状态
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ 优化器状态加载成功")
            
            # 加载训练历史
            if 'train_history' in checkpoint:
                train_history = checkpoint['train_history']
                print(f"✓ 训练历史加载成功（已有 {len(train_history['loss'])} 轮记录）")
            
            # 加载最佳验证率
            if 'val_rate' in checkpoint:
                best_val_rate = checkpoint['val_rate']
                print(f"✓ 最佳验证率: {best_val_rate:.4f}")
        except Exception as e:
            print(f"⚠ 加载训练状态失败: {e}")
            print("  将从头开始训练状态")
    
    # ========================================================================
    # 训练循环

    # ========================================================================
    print(f"\n{'='*60}")
    if start_epoch > 1:
        print(f"继续训练 (从第 {start_epoch} 轮开始)")
    else:
        print("开始训练")
    print('='*60)
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # 训练

        train_loss, train_rate, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            grad_accum_steps=args.grad_accum_steps, use_amp=args.use_amp,
            last_seq=args.last_seq
        )
        
        # 验证

        val_loss, val_rate, val_acc = validate(
            model, val_loader, criterion, device, args
        )


        # val_loss, val_rate, val_acc = 1,1,1
        
        # 更新学习率

        scheduler.step(val_loss)
        
        # 保存历史

        train_history['loss'].append(train_loss)
        train_history['rate'].append(train_rate)
        train_history['acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_rate'].append(val_rate)
        train_history['val_acc'].append(val_acc)
        
        # 打印结果

        print(f"训练 - 损失: {train_loss:.4f}, 率: {train_rate:.4f}, 准确率: {train_acc:.4f}")
        print(f"验证 - 损失: {val_loss:.4f}, 率: {val_rate:.4f}, 准确率: {val_acc:.4f}")
        
        # 保存最佳模型

        if val_rate <= best_val_rate:
            best_val_rate = val_rate
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rate': val_rate,
                'train_history': train_history,
                'args': vars(args),
                'compress_injection': compress_injection is not None,
                'decompress_injection': decompress_injection is not None,
                'compression_method': args.compression_method
            }, best_model_path)
            print(f"✓ 保存最佳模型 (验证率: {val_rate:.4f})")
    
        # test_sample(args, device, train_loader, model, best_model_path)
    # ========================================================================
    # 评估最终模型

    # ========================================================================
    print(f"\n{'='*60}")
    print("最终评估")
    print('='*60)
    
    # 加载最佳模型

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估压缩率

    avg_bits, compression_ratio, fixed_bits = evaluate_compression_rate(
        model, val_loader, device, args.vocab_size

    )
    
    print(f"评估结果:")
    print(f"  平均比特率: {avg_bits:.4f} bits/index")
    print(f"  固定长度编码: {fixed_bits:.4f} bits/index")
    print(f"  压缩比: {compression_ratio:.2f}:1")
    print(f"  压缩率提升: {(1 - avg_bits/fixed_bits)*100:.1f}%")
    
    # ========================================================================
    # 可视化训练曲线

    # ========================================================================
    plt.figure(figsize=(15, 5))
    
    # 损失曲线

    plt.subplot(1, 3, 1)
    plt.plot(train_history['loss'], label='训练损失')
    plt.plot(train_history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.title('训练损失曲线')
    plt.grid(True, alpha=0.3)
    
    # 率曲线

    plt.subplot(1, 3, 2)
    plt.plot(train_history['rate'], label='训练率')
    plt.plot(train_history['val_rate'], label='验证率')
    plt.axhline(y=fixed_bits, color='r', linestyle='--', label=f'固定长度 ({fixed_bits:.2f} bits)')
    plt.xlabel('Epoch')
    plt.ylabel('率 (bits/index)')
    plt.legend()
    plt.title('率曲线')
    plt.grid(True, alpha=0.3)
    
    # 准确率曲线

    plt.subplot(1, 3, 3)
    plt.plot(train_history['acc'], label='训练准确率')
    plt.plot(train_history['val_acc'], label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    plt.title('准确率曲线')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()
    
    print(f"\n训练曲线已保存至: {output_dir / 'training_curves.png'}")
    
    # ========================================================================
    # 保存最终模型和配置

    # ========================================================================
    final_model_path = output_dir / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'train_history': train_history,
        'compression_stats': {
            'avg_bits_per_index': avg_bits,
            'compression_ratio': compression_ratio,
            'fixed_bits_per_index': fixed_bits

        },
        'compress_injection': compress_injection is not None,
        'decompress_injection': decompress_injection is not None,
        'compression_method': args.compression_method

    }, final_model_path)
    
    # 保存配置

    config = {
        'vocab_size': args.vocab_size,
        'seq_len': args.seq_len,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'model_params': total_params,
        'compression_stats': {
            'avg_bits_per_index': avg_bits,
            'compression_ratio': compression_ratio

        },
        'compression_method': args.compression_method,
        'injection_enabled': compress_injection is not None

    }
    
    import json

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print("训练完成!")
    print('='*60)
    print(f"输出目录: {output_dir}")
    print(f"最佳模型: {best_model_path}")
    print(f"最终模型: {final_model_path}")
    print(f"配置文件: {output_dir / 'config.json'}")
    
    # test_sample(args, device, train_loader, model, best_model_path)
    




if __name__ == "__main__":
    main()
    
