from collections import OrderedDict
from typing import Tuple

import attr
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


@attr.s(repr=False, eq=False)
class VectorQuantizer(nn.Module):
    # 标准向量量化器，使用最近邻查找 + Straight-Through Estimator
    num_embeddings: int = attr.ib(default=8192, validator=lambda i, a, x: x >= 512)
    embedding_dim: int = attr.ib(default=256, validator=lambda i, a, x: x > 128)
    commitment_cost: float = attr.ib(default=0.25, validator=lambda i, a, x: x >= 0.0)

    def __attrs_post_init__(self):
        super().__init__()
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.p_depth_last = 'b c h w -> b h w c'
        self.p_space_last = 'b h w c -> b c h w'
        self.p_group = 'b h w c -> (b h w) c'
        self.p_flatten = '(b h w) c -> b h w c'
        self.p_code = '(b h w) -> b h w'

    def _forward(self, inputs, flat_input, **kwargs):
        # inputs 形状: [b, h, w, c]；flat_input 已展平为 [b*h*w, c]

        # 计算每个位置与 codebook 中每个向量的欧式距离
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # 选取最近的 code（硬分配 one-hot 编码）
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 查表得到量化后的向量，并恢复空间形状
        quantized = rearrange(torch.matmul(encodings, self.embedding.weight), self.p_flatten,
                              **kwargs)

        # 量化损失：codebook 更新 (q_latent) + 输入承诺损失 (e_latent)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-Through：前向用量化结果，反向梯度直接传给输入
        quantized = inputs + (quantized - inputs).detach()
        # 聚类使用情况的困惑度，衡量 codebook 利用率
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encoding_indices

    def forward(self, inputs):
        # BCHW -> BHWC
        inputs = rearrange(inputs, self.p_depth_last)
        # BHWC -> (B*H*W)C
        flat_input = rearrange(inputs, self.p_group)
        kwargs = {
            'b': inputs.size()[0],
            'h': inputs.size()[1],
            'w': inputs.size()[2],
            'c': inputs.size()[3]
        }
        loss, quantized, perplexity, encoding_indices = self._forward(inputs, flat_input, **kwargs)

        # convert quantized from BHWC -> BCHW
        quantized = rearrange(quantized, self.p_flatten)
        encoding_indices = rearrange(encoding_indices.squeeze(), self.p_code, b=kwargs['b'], h=kwargs['h'], w=kwargs['w'])
        return loss, quantized, perplexity, encoding_indices


@attr.s(repr=False, eq=False)
class VectorQuantizerEMA(nn.Module):
    # 使用 EMA 更新 codebook 的向量量化器，适合稳定训练
    num_embeddings: int = attr.ib(default=8192, validator=lambda i, a, x: x >= 512)
    embedding_dim: int = attr.ib(default=256, validator=lambda i, a, x: x > 128)
    commitment_cost: float = attr.ib(default=0.25, validator=lambda i, a, x: x >= 0.0)
    decay: float = attr.ib(default=0.99, validator=lambda i, a, x: 0.9 <= x < 1.0)
    epsilon: float = attr.ib(default=1e-5, validator=lambda i, a, x: x > 0)

    def __attrs_post_init__(self):
        super().__init__()
        # 输入形状为 [B, C, H, W]，其中 C = embedding_dim
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()

        self.register_buffer('ema_cluster_size', torch.zeros(self.num_embeddings))
        self.ema_w = nn.Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))
        self.ema_w.data.normal_()

        self.p_depth_last = 'b c h w -> b h w c'
        self.p_space_last = 'b h w c -> b c h w'
        self.p_group = 'b h w c -> (b h w) c'
        self.p_flatten = '(b h w) c -> b h w c'
        self.p_code = '(b h w) -> b h w'

    def _forward(self, inputs: torch.Tensor, flat_input: torch.Tensor, **kwargs) -> Tuple[torch.Tensor]:
        # inputs 形状: [b, h, w, c]；flat_input 形状: [b*h*w, c]

        # 计算与 codebook 的距离
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # 选取最近的 code
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 量化并恢复空间形状
        quantized = rearrange(torch.matmul(encodings, self.embedding.weight), self.p_flatten,
                              **kwargs)

        # EMA 更新 codebook：聚类计数 + 权重滑动平均
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                     (1 - self.decay) * torch.sum(encodings, 0)

            # Laplace 平滑，避免除零
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = ((self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)

            self.embedding.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))

        # 仅包含承诺损失，codebook 由 EMA 更新
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        # Straight-Through：前向用量化结果，梯度传回输入
        quantized = inputs + (quantized - inputs).detach()
        # 困惑度用于观察 codebook 的利用率
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encoding_indices

    def forward(self, inputs) -> Tuple[torch.Tensor]:
        # BCHW -> BHWC
        inputs = rearrange(inputs, self.p_depth_last)
        # BHWC -> (B*H*W)C
        flat_input = rearrange(inputs, self.p_group)

        kwargs = {
            'b': inputs.size()[0],
            'h': inputs.size()[1],
            'w': inputs.size()[2],
            'c': inputs.size()[3]
        }
        loss, quantized, perplexity, encoding_indices = self._forward(inputs, flat_input, **kwargs)

        # convert quantized from BHWC -> BCHW
        quantized = rearrange(quantized, self.p_space_last)

        encoding_indices = rearrange(encoding_indices.squeeze(), self.p_code, b=kwargs['b'], h=kwargs['h'], w=kwargs['w'])

        return loss, quantized, perplexity, encoding_indices

