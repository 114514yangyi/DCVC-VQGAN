"""
VQ-VAE2 主模型 (用于视频序列)

该模型实现了完整的 VQ-VAE2 架构，用于视频数据的自编码和量化：
1. Encoder: 将视频帧编码为潜在特征
2. Vector Quantizer: 将连续特征量化为离散码本索引
3. Decoder: 将量化特征解码为重建的视频帧

训练目标：
- 重建损失：使重建的视频尽可能接近原始视频
- VQ损失：使编码特征尽可能接近码本中的向量
"""

from typing import Tuple, Any
import types
import math

import attr
import torch
from einops import rearrange
from torch import nn

from models.vq_vae.layer import VectorQuantizerEMA, VectorQuantizer
from models.gan.encoder import Encoder
from models.gan.decoder import Decoder
from models.optvq.quantizer import VectorQuantizer as OptVQVectorQuantizer, VectorQuantizerSinkhorn
import torch.nn.functional as F


class OptVQQuantizerAdapter(nn.Module):
    """
    OptVQ 量化器的适配器，使其兼容原有的 VQ-VAE 接口
    
    原有接口: forward(inputs) -> (loss, quantized, perplexity, encoding_indices)
    OptVQ 接口: forward(x) -> (x_q, loss, indices)
    
    该适配器将 OptVQ 的输出转换为原有格式
    """
    def __init__(self, optvq_quantizer, commitment_cost: float = 0.25):
        super().__init__()
        self.optvq_quantizer = optvq_quantizer
        self.commitment_cost = commitment_cost
        
        # 为了兼容 decode 方法，需要提供 embedding 属性
        # OptVQ 使用 proj(embedding.weight) 作为实际的码本
        self._setup_embedding_proxy()
    
    def set_current_step(self, step: int):
        """设置当前训练步数（用于 OptVQ 的 start_quantize_steps 和统计信息）"""
        self.optvq_quantizer.set_current_step(step)
    
    def _setup_embedding_proxy(self):
        """设置 embedding 代理，用于 decode 方法"""
        # 创建一个代理对象，模拟 embedding 的行为
        class EmbeddingProxy:
            def __init__(self, quantizer):
                self.quantizer = quantizer
            
            def __getattr__(self, name):
                # 如果访问 weight，返回实际的码本权重
                if name == 'weight':
                    # 获取基础 embedding 权重
                    if self.quantizer.use_proj:
                        embed_weight = self.quantizer.proj(self.quantizer.embedding.weight)
                    else:
                        embed_weight = self.quantizer.embedding.weight
                    
                    # 如果使用多头，需要将多个头拼接起来
                    # OptVQ 中，每个头的维度是 code_dim = e_dim // num_head
                    # 在 forward 中，多个头会被拼接成 e_dim 维度
                    # 但在 decode 时，我们需要 (n_e, e_dim) 的权重矩阵
                    if self.quantizer.num_head > 1:
                        # 多头情况：embed_weight 形状是 (n_e, code_dim)
                        # 需要重复 num_head 次并拼接，得到 (n_e, e_dim)
                        code_dim = embed_weight.shape[1]
                        e_dim = self.quantizer.e_dim
                        # 将每个 embedding 重复 num_head 次
                        embed_weight = embed_weight.unsqueeze(1).repeat(1, self.quantizer.num_head, 1)
                        embed_weight = embed_weight.view(self.quantizer.n_e, e_dim)
                    
                    return embed_weight
                return getattr(self.quantizer.embedding, name)
        
        self.embedding = EmbeddingProxy(self.optvq_quantizer)
        # 为了兼容原有代码中的 num_embeddings 访问
        self.num_embeddings = self.optvq_quantizer.n_e
    
    def forward(self, inputs: torch.Tensor):
        """
        适配 OptVQ 的接口到原有接口
        
        Args:
            inputs: 输入张量，形状为 (batch, n_init, h', w')
        
        Returns:
            Tuple[loss, quantized, perplexity, encoding_indices]:
            - loss: VQ 损失（标量）
            - quantized: 量化后的特征，形状与输入相同
            - perplexity: 困惑度（标量）
            - encoding_indices: 码本索引，形状为 (batch, h', w')
        """
        # 调用 OptVQ 量化器
        # OptVQ 的 forward 返回 (x_q, loss, indices)
        quantized, loss, encoding_indices = self.optvq_quantizer(inputs)
        
        # 计算困惑度（perplexity）
        # 统计每个码本索引的使用频率
        if encoding_indices is not None:
            # encoding_indices 形状: (batch, h', w')
            flat_indices = encoding_indices.view(-1)  # (batch * h' * w',)
            
            # 计算每个码本向量的使用频率
            unique_indices, counts = torch.unique(flat_indices, return_counts=True)
            probs = torch.zeros(self.num_embeddings, device=inputs.device)
            probs[unique_indices] = counts.float() / flat_indices.numel()
            
            # 计算困惑度：exp(-sum(p * log(p)))
            # 只考虑被使用的码本向量
            used_probs = probs[probs > 0]
            if len(used_probs) > 0:
                perplexity = torch.exp(-torch.sum(used_probs * torch.log(used_probs + 1e-10)))
            else:
                perplexity = torch.tensor(1.0, device=inputs.device)
        else:
            perplexity = torch.tensor(1.0, device=inputs.device)
        
        # 返回格式: (loss, quantized, perplexity, encoding_indices)
        return loss, quantized, perplexity, encoding_indices
    
    @property
    def p_flatten(self):
        """为了兼容 decode 方法中的 p_flatten"""
        return '(b h w) c -> b h w c'
    
    @property
    def p_space_last(self):
        """为了兼容 decode 方法中的 p_space_last"""
        return 'b h w c -> b c h w'
    
    def embed_code(self, code, size=None, code_format="image"):
        """
        为了兼容，提供 embed_code 方法（OptVQ 的原生方法）
        """
        return self.optvq_quantizer.embed_code(code, size=size, code_format=code_format)


@attr.s(repr=False, eq=False)
class VqVae2(nn.Module):
    """
    VQ-VAE2 主模型类
    
    完整的 VQ-VAE2 架构，包含编码器、向量量化器和解码器三个主要组件。
    该模型专门设计用于处理视频序列数据。
    
    工作流程：
    1. 输入: (batch * sequence_length, channels, height, width) - 视频帧序列
    2. Encoder: 编码为 (batch, n_init, h', w') - 潜在特征
    3. Vector Quantizer: 量化为离散码本索引，输出量化特征
    4. Decoder: 解码为 (batch * sequence_length, channels, height, width) - 重建视频帧
    
    损失函数：
    - 重建损失 (Reconstruction Loss): MSE(重建帧, 原始帧)
    - VQ损失 (Vector Quantization Loss): 包含 commitment loss 和 codebook loss
    """
    
    # ========== 模型架构参数 ==========
    group_count: int = attr.ib()
    """
    编码/解码组的数量
    决定了模型的深度和空间下采样/上采样的层级数
    通常设置为 3-5
    """
    
    n_hid: int = attr.ib(default=256, validator=lambda i, a, x: x >= 64)
    """
    初始隐藏特征维度
    编码器和解码器都使用这个值作为基础特征维度
    后续组的特征维度会基于此值递增或递减
    """
    
    n_init: int = attr.ib(default=128, validator=lambda i, a, x: x >= 8)
    """
    编码器输出的特征维度，也是 codebook 的嵌入维度
    这个值决定了量化向量的维度，影响模型的表达能力
    通常设置为 64-1024
    """
    
    
    n_blk_per_group: int = attr.ib(default=2, validator=lambda i, a, x: x >= 1)
    """
    每个编码/解码组中包含的块数量
    每个块都是一个残差块，用于特征提取或重建
    通常设置为 1-3
    """
    
    # ========== 输入输出参数 ==========
    input_channels: int = attr.ib(default=3, validator=lambda i, a, x: x >= 1)
    """
    输入图像的通道数
    通常为 3 (RGB) 或 1 (灰度)
    """
    
    output_channels: int = attr.ib(default=3, validator=lambda i, a, x: x >= 1)
    """
    输出图像的通道数
    应该与 input_channels 相同
    """
    
    # ========== 向量量化参数 ==========
    vocab_size: int = attr.ib(default=8192, validator=lambda i, a, x: x >= 512)
    """
    码本大小，即离散向量的数量
    这个值决定了模型的离散表示能力
    通常设置为 512-8192，更大的值可以表示更丰富的特征
    """
    
    commitment_cost: float = attr.ib(default=0.25, validator=lambda i, a, x: x >= 0.0)
    """
    Commitment cost，用于 VQ 损失的计算
    控制编码器输出与量化向量之间的对齐强度
    较大的值会强制编码器输出更接近码本向量
    通常设置为 0.1-0.5
    """
    
    decay: float = attr.ib(default=0.99, validator=lambda i, a, x: x >= 0.0)
    """
    EMA (指数移动平均) 衰减率，用于 VectorQuantizerEMA
    控制码本向量的更新速度
    当 decay > 0 时使用 EMA 更新，否则使用标准 VQ
    通常设置为 0.99-0.999
    如果设置为 0，则使用标准的 VectorQuantizer（不使用 EMA）
    """
    
    # ========== OptVQ 量化器参数 ==========
    use_optvq: bool = attr.ib(default=True)
    """
    是否使用 OptVQ 量化器（Optimal Transport Vector Quantization）
    True: 使用 OptVQ 的 VectorQuantizer
    False: 使用原始的 VectorQuantizerEMA/VectorQuantizer
    """
    
    optvq_use_sinkhorn: bool = attr.ib(default=True)
    """
    是否使用 Sinkhorn 算法（OptVQ 的高级版本）
    True: 使用 VectorQuantizerSinkhorn（基于最优传输）
    False: 使用标准的 VectorQuantizer
    """
    
    optvq_beta: float = attr.ib(default=1.0)
    """
    OptVQ 的 beta 参数，控制损失权重
    """
    
    optvq_use_proj: bool = attr.ib(default=True)
    """
    是否使用投影层（OptVQ 的特性）
    """
    
    optvq_num_head: int = attr.ib(default=1)
    """
    OptVQ 的多头数量
    """
    
    optvq_logger: Any = attr.ib(default=None)
    """
    可选的 logger 对象，用于记录 OptVQ 的统计信息
    如果为 None，则不记录统计信息
    """
    
    optvq_enable_stats: bool = attr.ib(default=True)
    """
    是否启用 OptVQ 的统计信息记录
    """
    
    # ========== 视频序列参数 ==========
    sequence_length: int = attr.ib(default=16, validator=lambda i, a, x: x > 0)
    """
    视频序列的长度，即一次处理多少帧
    输入格式是 (batch * sequence_length, channels, h, w)
    通常设置为 8-32
    """
    
    # ========== 下采样/上采样控制 ==========
    downsample: bool = attr.ib(default=True)
    """
    是否在编码器中进行空间下采样
    True: 每个编码组后使用 MaxPool2d 进行2倍下采样
    False: 不进行下采样，保持空间尺寸不变
    """
    
    upsample: bool = attr.ib(default=True)
    """
    是否在解码器中进行空间上采样
    True: 每个解码组后使用 Upsample 进行2倍上采样
    False: 不进行上采样，保持空间尺寸不变
    应该与 downsample 配对使用
    """

    def __attrs_post_init__(self):
        """
        初始化模型组件
        
        创建三个主要组件：
        1. Encoder: 编码器，将视频帧编码为潜在特征
        2. Vector Quantizer: 向量量化器，将连续特征量化为离散码本索引
        3. Decoder: 解码器，将量化特征解码为重建的视频帧
        """
        super().__init__()

        # ========== 创建 args 对象用于 GAN encoder/decoder ==========
        # GAN 的 encoder 和 decoder 需要 args 对象
        args = types.SimpleNamespace()
        args.image_channels = self.input_channels
        args.latent_dim = self.n_init

        # ========== 创建编码器 ==========
        # 使用 GAN 的 Encoder，处理单帧图像
        # 输入: (batch * sequence_length, input_channels, height, width)
        # 输出: (batch * sequence_length, latent_dim, h', w')
        self.encoder = Encoder(args)


        # ========== 创建向量量化器 ==========
        # 向量量化器将连续的潜在特征量化为离散的码本索引
        # 输入: (batch, n_init, h', w')
        # 输出: 量化后的特征 (batch, n_init, h', w')，以及 VQ 损失和困惑度
        if self.use_optvq:
            # 使用 OptVQ 量化器（Optimal Transport Vector Quantization）
            if self.optvq_use_sinkhorn:
                # 使用 Sinkhorn 算法（基于最优传输）
                optvq_quantizer = VectorQuantizerSinkhorn(
                    n_e=self.vocab_size,
                    e_dim=self.n_init,
                    beta=self.optvq_beta,
                    use_proj=self.optvq_use_proj,
                    num_head=self.optvq_num_head,
                    loss_q_type="l2",  # 使用 L2 损失
                    logger=self.optvq_logger,
                    current_step=0,
                    enable_stats=self.optvq_enable_stats
                )
            else:
                # 使用标准的 OptVQ 量化器
                optvq_quantizer = OptVQVectorQuantizer(
                    n_e=self.vocab_size,
                    e_dim=self.n_init,
                    beta=self.optvq_beta,
                    use_proj=self.optvq_use_proj,
                    num_head=self.optvq_num_head,
                    loss_q_type="l2",  # 使用 L2 损失
                    logger=self.optvq_logger,
                    current_step=0,
                    enable_stats=self.optvq_enable_stats
                )
            # 创建适配器以兼容原有接口
            self.vq_vae = OptVQQuantizerAdapter(optvq_quantizer, self.commitment_cost)
        elif self.decay > 0.0:
            # 使用 EMA (指数移动平均) 更新码本向量
            # 这种方法可以更稳定地更新码本，通常效果更好
            self.vq_vae = VectorQuantizerEMA(
                num_embeddings=self.vocab_size,      # 码本大小
                embedding_dim=self.n_init,          # 嵌入维度
                commitment_cost=self.commitment_cost,  # commitment cost
                decay=self.decay                     # EMA 衰减率
            )
        else:
            # 使用标准的向量量化器（不使用 EMA）
            # 码本向量通过梯度更新
            self.vq_vae = VectorQuantizer(
                num_embeddings=self.vocab_size,
                embedding_dim=self.n_init,
                commitment_cost=self.commitment_cost
            )


        # ========== 创建解码器 ==========
        # 使用 GAN 的 Decoder，处理单帧图像
        # 输入: (batch * sequence_length, latent_dim, h', w')
        # 输出: (batch * sequence_length, output_channels, height, width)
        self.decoder = Decoder(args)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        前向传播（训练时使用）
        
        Args:
            x: 输入视频帧序列，形状为 (batch * sequence_length, channels, height, width)
               例如：batch=4, sequence_length=16 时，输入形状为 (64, 3, 256, 256)
        
        Returns:
            Tuple[loss, x_recon, perplexity, encoding_indices]:
            - loss: VQ 损失（标量），用于训练向量量化器
            - x_recon: 重建的视频帧，形状与输入相同
            - perplexity: 困惑度（标量），衡量码本的使用均匀程度
            - encoding_indices: 码本索引
        """
        # 步骤1: 编码每一帧
        # GAN encoder 处理单帧，输入: (batch * seq_length, channels, h, w)
        # 输出: (batch * seq_length, latent_dim, h', w')
        z = self.encoder(x)
        
        # 步骤2: 向量量化
        # 将连续的潜在特征量化为离散的码本索引
        # 输入: (batch*sequence_length, n_init, h', w')
        # 输出: 
        #   - loss: VQ 损失（标量）
        #   - quantized: 量化后的特征 (batch*sequence_length, n_init, h', w')
        #   - perplexity: 困惑度（标量）
        #   - codes: 码本索引 (batch*sequence_length, h', w')
        loss, quantized, perplexity, encoding_indices = self.vq_vae(z)
        
        # 步骤3: 解码
        # GAN decoder 处理单帧，输入: (batch*sequence_length, n_init, h', w')
        # 输出: (batch*sequence_length, channels, h, w)
        x_recon = self.decoder(quantized)
        
        # 返回量化索引，便于统计码本使用情况与比特率估计
        return loss, x_recon, perplexity, encoding_indices

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码函数（推理时使用）
        
        将输入视频帧编码为码本索引，不进行梯度计算
        
        Args:
            x: 输入视频帧序列，形状为 (batch * sequence_length, channels, height, width)
        
        Returns:
            codes: 码本索引，形状为 (batch, h', w')
                  每个位置的值是 [0, vocab_size-1] 的整数，表示该位置对应的码本向量索引
        """
        # 步骤1: 编码每一帧
        z = self.encoder(x)
        
        # 步骤2: 获取码本索引
        _, _, _, encoding_indices = self.vq_vae(z)
        return encoding_indices

    @torch.no_grad()
    def decode(self, encode_indices: torch.Tensor) -> torch.Tensor:
        """
        解码函数（推理时使用）
        
        从码本索引重建视频帧，不进行梯度计算
        
        Args:
            encode_indices: 码本索引，形状为 (batch, h', w') 或 (batch, h', w', num_head)
                           每个位置的值是 [0, vocab_size-1] 的整数
        
        Returns:
            重建的视频帧，形状为 (batch * sequence_length, channels, height, width)
        """
        # 如果使用的是 OptVQ 量化器，使用其原生的 embed_code 方法
        if hasattr(self.vq_vae, 'optvq_quantizer'):
            # OptVQ 的 embed_code 方法可以处理多头情况
            # encode_indices 形状可能是:
            #   - (batch, h', w') 当 num_head=1
            #   - (batch, h', w', num_head) 当 num_head>1
            #   - (batch, num_head, h', w') 当 num_head>1 且 recover_output 处理错误时
            b = encode_indices.shape[0]
            optvq_quantizer = self.vq_vae.optvq_quantizer
            actual_num_head = optvq_quantizer.num_head
            
            # 处理不同的输入形状
            if encode_indices.ndim == 3:
                # 可能是 (batch, h', w') 或 (batch, num_head, h') 或 (batch, h', num_head)
                # 检查是否是 (batch, num_head, h') 的情况（recover_output 的错误输出）
                if encode_indices.shape[1] == actual_num_head and actual_num_head > 1:
                    # (batch, num_head, h') -> 需要转换为 (batch, h', num_head)
                    # 但我们需要知道 h' 和 w'，这里假设 h' = w' = sqrt(shape[2])
                    h_w = encode_indices.shape[2]
                    h = w = int(math.sqrt(h_w))
                    if h * w != h_w:
                        raise ValueError(f"Cannot infer spatial dimensions from shape {encode_indices.shape}")
                    # 转换为 (batch, h', w', num_head)
                    encode_indices = encode_indices.permute(0, 2, 1).contiguous()  # (batch, h', num_head)
                    encode_indices = encode_indices.view(b, h, w, actual_num_head)
                    code_flat = encode_indices.view(b, -1)  # (batch, h'*w'*num_head)
                else:
                    # (batch, h', w') - num_head=1 的情况
                    h, w = encode_indices.shape[1], encode_indices.shape[2]
                    code_flat = encode_indices.view(b, -1)  # (batch, h'*w')
            elif encode_indices.ndim == 4:
                # 可能是 (batch, h', w', num_head) 或 (batch, num_head, h', w')
                if encode_indices.shape[1] == actual_num_head and actual_num_head > 1:
                    # (batch, num_head, h', w') -> 转换为 (batch, h', w', num_head)
                    encode_indices = encode_indices.permute(0, 2, 3, 1).contiguous()
                h, w = encode_indices.shape[1], encode_indices.shape[2]
                code_flat = encode_indices.view(b, -1)  # (batch, h'*w'*num_head)
            else:
                raise ValueError(f"Unexpected encode_indices shape: {encode_indices.shape}")
            
            # 验证维度并调用 embed_code
            dim = code_flat.shape[1]
            expected_dim = h * w * actual_num_head
            if dim == expected_dim:
                # 维度匹配，可以安全传递 size
                quantized = self.vq_vae.embed_code(
                    code_flat, 
                    size=(h, w), 
                    code_format="image"
                )
            else:
                # 维度不匹配，让 embed_code 自动推断空间维度
                # 这可以避免断言错误
                quantized = self.vq_vae.embed_code(
                    code_flat, 
                    size=None, 
                    code_format="image"
                )
            # quantized 形状: (batch, n_init, h', w')
        else:
            # 使用原有的解码方法（兼容原始 VectorQuantizer）
            # 获取空间维度
            b, h, w = encode_indices.size()
            
            # 步骤1: 将索引展平并转换为 one-hot 编码
            # 从 (batch, h, w) 转换为 (batch * h * w, 1)
            encode_indices_flat = rearrange(encode_indices, 'b h w -> (b h w) 1').to(torch.int64)
            
            # 创建 one-hot 编码矩阵
            # 形状: (batch * h * w, vocab_size)
            encodings = torch.zeros(
                encode_indices_flat.shape[0], 
                self.vq_vae.num_embeddings, 
                device=encode_indices.device
            )
            # 将对应位置的索引设置为1
            encodings.scatter_(1, encode_indices_flat, 1)
            
            # 步骤2: 通过 one-hot 编码和码本权重矩阵的乘积，获取量化向量
            # encodings @ embedding.weight: (batch * h * w, vocab_size) @ (vocab_size, n_init)
            #                                -> (batch * h * w, n_init)
            # 然后重新排列为 (batch, h, w, n_init)
            quantized = rearrange(
                torch.matmul(encodings, self.vq_vae.embedding.weight), 
                self.vq_vae.p_flatten,
                b=b, h=h, w=w
            )
            # 转换为 (batch, n_init, h, w)
            quantized = rearrange(quantized, self.vq_vae.p_space_last)
        
        # 步骤3: 解码为视频帧
        # 输入: (batch, n_init, h, w)
        # 输出: (batch * sequence_length, channels, height, width)
        return self.decoder(quantized)
