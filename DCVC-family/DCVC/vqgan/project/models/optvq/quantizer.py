# ------------------------------------------------------------------------------
# OptVQ: Preventing Local Pitfalls in Vector Quantization via Optimal Transport
# Copyright (c) 2024 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch.distributed as dist

class VectorQuantizer(nn.Module):
    def __init__(self, n_e: int = 1024, e_dim: int = 128, 
                 beta: float = 1.0, use_norm: bool = False,
                 use_proj: bool = True, fix_codes: bool = False,
                 loss_q_type: str = "ce",
                 num_head: int = 1,
                 start_quantize_steps: int = None,
                 logger=None,
                 current_step: int = 0,
                 enable_stats: bool = False):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.loss_q_type = loss_q_type
        self.num_head = num_head
        self.start_quantize_steps = start_quantize_steps
        self.code_dim = self.e_dim // self.num_head
        self.logger = logger  # 可选的 logger，用于记录统计信息
        self.current_step = current_step  # 当前训练步数
        self.enable_stats = enable_stats  # 是否启用统计信息记录

        self.norm = lambda x: F.normalize(x, p=2.0, dim=-1, eps=1e-6) if use_norm else x
        assert not use_norm, f"use_norm=True is no longer supported! Because the norm operation without theorectical analysis may cause unpredictable unstability."
        self.use_proj = use_proj

        self.embedding = nn.Embedding(num_embeddings=n_e, embedding_dim=self.code_dim)
        if use_proj:
            self.proj = nn.Linear(self.code_dim, self.code_dim)
            torch.nn.init.normal_(self.proj.weight, std=self.code_dim ** -0.5)
        if fix_codes:
            self.embedding.weight.requires_grad = False
    
    def set_current_step(self, step: int):
        """设置当前训练步数（用于 start_quantize_steps）"""
        self.current_step = step

    def reshape_input(self, x: Tensor):
        """
        (B, C, H, W) / (B, T, C) -> (B, T, C)
        """
        if x.ndim == 4:
            _, C, H, W = x.size()
            x = x.permute(0, 2, 3, 1).contiguous().view(-1, H * W, C)
            return x, {"size": (H, W)}
        elif x.ndim == 3:
            return x, None
        else:
            raise ValueError("Invalid input shape!")

    def recover_output(self, x: Tensor, info):
        if info is not None:
            H, W = info["size"]
            if x.ndim == 3: # features (B, T, C) -> (B, C, H, W)
                C = x.size(2)
                return x.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
            elif x.ndim == 2: # indices (B, T) -> (B, H, W)
                return x.view(-1, H, W)
            else:
                raise ValueError("Invalid input shape!")
        else: # features (B, T, C) or indices (B, T)
            return x
    
    def get_codebook(self, return_numpy: bool = True):
        embed = self.proj(self.embedding.weight) if self.use_proj else self.embedding.weight
        if return_numpy:
            return embed.data.cpu().numpy()
        else:
            return embed.data

    def quantize_input(self, query, reference):
        # compute the distance matrix
        query2ref = torch.cdist(query, reference, p=2.0) # (B1, B2)

        # find the nearest embedding
        indices = torch.argmin(query2ref, dim=-1) # (B1,)
        nearest_ref = reference[indices] # (B1, C)
            
        return indices, nearest_ref, query2ref

    def compute_codebook_loss(self, query, indices, nearest_ref, beta: float, query2ref):
        # compute the loss
        if self.loss_q_type == "l2":
            loss = torch.mean((query - nearest_ref.detach()).pow(2)) + \
                   torch.mean((nearest_ref - query.detach()).pow(2)) * beta
        elif self.loss_q_type == "l1":
            loss = torch.mean((query - nearest_ref.detach()).abs()) + \
                   torch.mean((nearest_ref - query.detach()).abs()) * beta
        elif self.loss_q_type == "ce":
            loss = F.cross_entropy(- query2ref, indices)

        return loss
    
    def compute_quantized_output(self, x, x_q):
        if self.start_quantize_steps is not None:
            if self.training and self.current_step < self.start_quantize_steps:
                if self.logger is not None:
                    self.logger.log_metrics(self.current_step, {"params/quantize_ratio": 0.0})
                return x
            else:
                if self.logger is not None:
                    self.logger.log_metrics(self.current_step, {"params/quantize_ratio": 1.0})
                return x + (x_q - x).detach()
        else:
            if self.logger is not None:
                self.logger.log_metrics(self.current_step, {"params/quantize_ratio": 1.0})
            return x + (x_q - x).detach()  

    def embed_code(self, code, size=None, code_format="image"):
        """
        Args:
            code_format (str): "image" (B x nH x H x W) or "sequence" (B x T x nH)
        """
        code = code.view(code.shape[0], -1)
        B, dim = code.size()
        if size is not None:
            H, W = size[0], size[1]
        else:
            H_W = int(dim / self.num_head)
            H = W = int(math.sqrt(H_W))
        assert H * W * self.num_head == dim

        embed = self.proj(self.embedding.weight) if self.use_proj else self.embedding.weight
        embed = self.norm(embed)
        x_q = embed[code] # (B, TxnH, dC)

        if code_format == "image":
            x_q = x_q.view(B, self.num_head, H_W, -1)
            x_q = x_q.permute(0, 2, 1, 3).contiguous().view(B, H_W, -1)
        elif code_format == "sequence":
            x_q = x_q.view(B, H_W, -1)
        x_q = self.recover_output(x_q, {"size": (H, W)})
        return x_q

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x: Tensor):
        """
        量化输入张量x，返回量化后的结果、量化损失和索引。

        Args:
            x (Tensor): 输入特征张量。shape 可以是 (B, C, H, W) 或 (B, T, C)
        Returns:
            x_q (Tensor): 量化后的特征，shape 与输入一致
            loss (Tensor): 码本量化损失
            indices (Tensor): 每个patch对应的码本索引
        """
        x = x.float()  # 确保输入为float类型（量化通常在float上操作）
        
        # reshape_input 做输入整理与格式兼容，info用于recover_output恢复shape
        x, info = self.reshape_input(x)
        B, T, C = x.size()  # B=batch数, T=patch数量/序列长度, C=通道数
        
        # 将输入展平为二维 (B*T, C)，便于与码本做对比
        x = x.view(-1, C)
        
        # 根据是否使用投影层（proj）选用嵌入表，embed shape = (num_code, dC)
        embed = self.proj(self.embedding.weight) if self.use_proj else self.embedding.weight

        # 若是多头向量量化，将x继续reshape兼容多头，每个头独立量化
        if self.num_head > 1:
            x = x.view(-1, self.code_dim)  # (B*T*nH, dC)，每个头一个向量子空间

        # 对输入x和嵌入表进行层归一化（提升数值稳定性和训练效率）
        x, embed = self.norm(x), self.norm(embed)

        # 量化操作：获得最近邻索引、对应码本向量，以及查询与引用的距离（或相似度）
        indices, x_q, query2ref = self.quantize_input(x, embed)
        
        # 计算码本相关损失（如VQ/VQ-commitment损失、L1或cross-entropy等）
        loss = self.compute_codebook_loss(
            query=x, 
            indices=indices, 
            nearest_ref=x_q, 
            beta=self.beta, 
            query2ref=query2ref
        )

        # 可选：记录一些训练时的统计量，便于监控码本利用率、范数分布等
        # if self.training and self.enable_stats and self.logger is not None:
        #     with torch.no_grad():
        #         # num_unique：当前batch激活的码本种类数量
        #         num_unique = torch.unique(indices).size(0)
        #         # x_norm_mean：输入query向量范数均值
        #         x_norm_mean = torch.mean(x.norm(dim=-1))
        #         # embed_norm_mean：码本向量范数均值
        #         embed_norm_mean = torch.mean(embed.norm(dim=-1))
        #         # diff_norm_mean：query与最近码本向量距离均值
        #         diff_norm_mean = torch.mean((x_q - x).norm(dim=-1))
        #         # x2e_mean：query到嵌入的距离均值（通常与query2ref直接相关）
        #         x2e_mean = query2ref.mean()
        #         # 汇总统计信息，并利用logger记录
        #         stats_dict = {
        #             "params/num_unique": num_unique.item() if isinstance(num_unique, torch.Tensor) else num_unique,
        #             "params/x_norm": x_norm_mean.item(),
        #             "params/embed_norm": embed_norm_mean.item(),
        #             "params/diff_norm": diff_norm_mean.item(),
        #             "params/x2e_mean": x2e_mean.item()
        #         }
        #         self.logger.log_metrics(self.current_step, stats_dict)
    
        # 生成最终量化输出。根据训练阶段是否启用量化进行软/硬量化跳跃
        # 输出reshape回(B,T,C)
        x_q = self.compute_quantized_output(x, x_q).view(B, T, C)
        # 码本索引也reshape为(B,T,num_head)
        indices = indices.view(B, T, self.num_head)

        # 利用info恢复到输入前的空间尺寸（如还原到(B,C,H,W)）
        x_q = self.recover_output(x_q, info)
        indices = self.recover_output(indices, info)
        
        return x_q, loss, indices

def sinkhorn(cost: Tensor, n_iters: int = 3, epsilon: float = 1, is_distributed: bool = False):
    """
    Sinkhorn 算法实现，用于概率最优传输的近似优化。

    参数:
        cost (Tensor): 距离或代价矩阵，形状为 (B, K)，B 表示样本数，K 表示簇/码本数。
        n_iters (int): Sinkhorn 归一化的迭代次数，典型值为3~5。
        epsilon (float): 平滑/温度参数，控制softmax的锐利程度。
        is_distributed (bool): 是否在分布式环境下运行（DDP），如果为True则归一化操作跨多卡同步。

    返回:
        Q (Tensor): Sinkhorn归一化的soft assignment矩阵，形状为 (B, K)，每一行和为1。
    """
    # (1) 使用 Gibbs 分布（softmin）将代价矩阵转化为概率矩阵：
    #    Q(b, k) = exp(-epsilon * cost(b, k))
    #    转置后 Q shape: (K, B)，便于后续按 prototype(row) 和 sample(col) 归一化
    Q = torch.exp(- cost * epsilon).t() # (K, B)

    # (2) 获取 B, K（样本数与prototype/簇数）。分布式时，B要乘世界卡数
    if is_distributed:
        B = Q.size(1) * dist.get_world_size()
    else:
        B = Q.size(1)
    K = Q.size(0)

    # (3) 首先把所有概率之和归一化，使所有元素和为1（Q变成概率分布）
    sum_Q = torch.sum(Q)
    if is_distributed:
        dist.all_reduce(sum_Q)  # 分布式同步sum
    Q /= (sum_Q + 1e-8)        # 防止除以0

    # (4) 迭代n_iters次，使得最终Q每行分布都等于1/K，每列分布都等于1/B
    for _ in range(n_iters):
        # (4.1) 行归一化，使每个prototype被分配的概率总和为1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)     # 每行的和，shape (K, 1)
        if is_distributed:
            dist.all_reduce(sum_of_rows)                    # 分布式同步sum
        Q /= (sum_of_rows + 1e-8)                           # 防止除0，归一化每行和为1
        Q /= K                                              # 所有prototype加起来和为1

        # (4.2) 列归一化，使每个query/sample分配到的概率总和为1/B
        sum_of_cols = torch.sum(Q, dim=0, keepdim=True)     # 每列的和，shape (1, B)
        Q /= (sum_of_cols + 1e-8)                           # 归一化每列
        Q /= B                                              # 所有sample加起来和为1

    # (5) 为了保证每一列的和为1（K*Q[:,b]加起来为1），最后再恢复回原始样本总数的比例
    Q *= B

    # (6) 转置使结果变回 (B, K)，每一行表示一个query映射到每个prototype的概率
    return Q.t() # (B, K)

class VectorQuantizerSinkhorn(VectorQuantizer):
    def __init__(self, epsilon: float = 10.0, n_iters: int = 5, 
                 normalize_mode: str = "all", use_prob: bool = True,
                 *args, **kwargs):
        super(VectorQuantizerSinkhorn, self).__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.normalize_mode = normalize_mode
        self.use_prob = use_prob

    def normalize(self, A, dim, mode="all"):
        if mode == "all":
            A = (A - A.mean()) / (A.std() + 1e-6)
            A = A - A.min()
        elif mode == "dim":
            A = A / math.sqrt(dim)
        elif mode == "null":
            pass
        return A

    def quantize_input(self, query, reference):
        """
        对输入的查询向量 query 与参考向量 reference 进行最优分配（量化）。
        
        主要流程分为以下几步：
        1. 计算距离矩阵，获得每个查询向量与所有参考向量（如码本向量）的欧氏距离。
        2. 对距离进行归一化处理，根据指定的模式（如"all"、"dim"、"null"），保证数值范围适合后续处理。
        3. 使用Sinkhorn算法对归一化后的距离进行最优传输分配，获得概率型分配矩阵Q。
        4. 根据 use_prob 标志决定选取方式：
           - 若 use_prob = True，则使用多项式采样，从概率分配Q中采样离散索引（即软量化）。
           - 否则，直接选择概率最大的位置（即硬量化）。
        5. 根据最终获得的索引，从参考向量中选择与每个查询向量最匹配的码本项。
        6. 如果训练模式下并且需要统计，支持定期（如每1000步）记录直方图，便于分析码本利用率。
        7. 返回三个内容：每个query的编码索引、所选出的参考向量，以及原始距离矩阵。

        参数说明:
            query (Tensor): shape (B1, D) 查询向量或特征
            reference (Tensor): shape (B2, D) 码本/参考向量

        返回:
            indices (LongTensor): shape (B1,) 每个查询向量的最佳（或采样）参考向量索引
            nearest_ref (Tensor): shape (B1, D) 选出的参考向量结果
            query2ref (Tensor): shape (B1, B2) 查询向量与码本的距离矩阵
        """

        # (1) 计算距离矩阵。torch.cdist 计算 L2 距离。
        query2ref = torch.cdist(query, reference, p=2.0)  # (B1, B2)，每行表示一个query与所有reference的距离

        # (2) 归一化距离并使用Sinkhorn算法进行分配
        with torch.no_grad():
            # 判断是否为分布式训练
            is_distributed = dist.is_initialized() and dist.get_world_size() > 1
            # 归一化距离，使得后续数值更平稳
            normalized_cost = self.normalize(
                query2ref,
                dim=reference.size(1),
                mode=self.normalize_mode
            )
            # 使用Sinkhorn算法，获得soft assignment概率分配矩阵Q
            Q = sinkhorn(
                normalized_cost,
                n_iters=self.n_iters,
                epsilon=self.epsilon,
                is_distributed=is_distributed
            )

        # (3) 根据使用概率采样还是最大值，获得每个query的最终分配索引
        if self.use_prob:
            # 避免概率全0导致采样失败，对每行最大值额外加一点
            max_q_id = torch.argmax(Q, dim=-1)
            Q[torch.arange(Q.size(0)), max_q_id] += 1e-8
            # 多项式采样，每个query按概率Q[i, :]采样一个reference
            indices = torch.multinomial(Q, num_samples=1).squeeze()
        else:
            # 直接取概率最大的reference索引
            indices = torch.argmax(Q, dim=-1)
        # 选取最终分配的参考向量
        nearest_ref = reference[indices]

        # (4) 可选：训练时每1000步收集一次量化分配统计信息（如用于tensorboard等分析）
        if self.training and self.enable_stats and self.logger is not None:
            if self.current_step % 1000 == 0:
                # 若logger支持此类统计，则可在此添加具体记录代码
                # 当前版本留空，仅作为扩展点
                pass

        # (5) 返回
        return indices, nearest_ref, query2ref

class Identity(VectorQuantizer):
    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x: Tensor):
        x = x.float()
        loss_q = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # compute statistics (可选)
        if self.training and self.enable_stats and self.logger is not None:
            with torch.no_grad():
                x_flatten, _ = self.reshape_input(x)
                x_norm_mean = torch.mean(x_flatten.norm(dim=-1))
                self.logger.log_metrics(self.current_step, {"params/x_norm": x_norm_mean.item()})
        
        return x, loss_q, None