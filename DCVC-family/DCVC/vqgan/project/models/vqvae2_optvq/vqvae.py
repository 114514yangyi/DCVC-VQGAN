import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log2
from typing import Tuple

from models.vqvae2.helper import HelperModule
from models.optvq.quantizer import VectorQuantizerSinkhorn

class ReZero(HelperModule):
    def build(self, in_channels: int, res_channels: int):
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x) * self.alpha + x

class ResidualStack(HelperModule):
    def build(self, in_channels: int, res_channels: int, nb_layers: int):
        self.stack = nn.Sequential(*[ReZero(in_channels, res_channels) 
                        for _ in range(nb_layers)
                    ])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.stack(x)

class Encoder(HelperModule):
    def build(self, 
            in_channels: int, hidden_channels: int, 
            res_channels: int, nb_res_layers: int,
            downscale_factor: int,
        ):
        assert log2(downscale_factor) % 1 == 0, "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))
        layers = []
        c_channel, n_channel = in_channels, hidden_channels // 2
        for _ in range(downscale_steps):
            layers.append(nn.Sequential(
                nn.Conv2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, hidden_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        layers.append(ResidualStack(n_channel, res_channels, nb_res_layers))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

class Decoder(HelperModule):
    def build(self, 
            in_channels: int, hidden_channels: int, out_channels: int,
            res_channels: int, nb_res_layers: int,
            upscale_factor: int,
        ):
        assert log2(upscale_factor) % 1 == 0, "Downscale must be a power of 2"
        upscale_steps = int(log2(upscale_factor))
        layers = [nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1)]
        layers.append(ResidualStack(hidden_channels, res_channels, nb_res_layers))
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        for _ in range(upscale_steps):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, out_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        # layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

"""
    Almost directly taken from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
    No reason to reinvent this rather complex mechanism.

    Essentially handles the "discrete" part of the network, and training through EMA rather than 
    third term in loss function.
"""
class CodeLayer(HelperModule):
    def build(self, in_channels: int, embed_dim: int, nb_entries: int,
              epsilon: float = 10.0, n_iters: int = 5,
              normalize_mode: str = "all", use_prob: bool = True,
              beta: float = 1.0, loss_q_type: str = "ce",
              use_norm: bool = False, use_proj: bool = True,
              logger=None, enable_stats: bool = False):
        """
        使用 OptVQ (VectorQuantizerSinkhorn) 的 CodeLayer
        
        Args:
            in_channels: 输入通道数
            embed_dim: embedding 维度
            nb_entries: codebook 大小
            epsilon: Sinkhorn 算法的 epsilon 参数
            n_iters: Sinkhorn 算法的迭代次数
            normalize_mode: 归一化模式 ("all", "dim", "null")
            use_prob: 是否使用概率采样
            beta: VQ 损失的 beta 参数
            loss_q_type: 损失类型 ("ce", "l2", "l1")
            use_norm: 是否使用归一化（已废弃，保持 False）
            use_proj: 是否使用投影层
            logger: 可选的 logger（用于统计信息）
            enable_stats: 是否启用统计信息记录
        """
        self.conv_in = nn.Conv2d(in_channels, embed_dim, 1)
        
        self.dim = embed_dim
        self.n_embed = nb_entries
        
        # 初始化 OptVQ 量化器
        self.quantizer = VectorQuantizerSinkhorn(
            n_e=nb_entries,
            e_dim=embed_dim,
            beta=beta,
            use_norm=use_norm,
            use_proj=use_proj,
            loss_q_type=loss_q_type,
            num_head=1,  # 单头
            epsilon=epsilon,
            n_iters=n_iters,
            normalize_mode=normalize_mode,
            use_prob=use_prob,
            logger=logger,
            current_step=0,
            enable_stats=enable_stats
        )

    def set_current_step(self, step: int):
        """设置当前训练步数（用于 OptVQ 的 start_quantize_steps）"""
        if hasattr(self.quantizer, 'set_current_step'):
            self.quantizer.set_current_step(step)

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.Tensor, torch.LongTensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, C, H, W)
            
        Returns:
            quantize: 量化后的特征 (B, C, H, W)
            diff: VQ 损失（标量 tensor）
            embed_ind: 编码索引 (B, H, W)
        """
        # 通过 conv_in 投影到 embed_dim
        x_proj = self.conv_in(x.float())  # (B, embed_dim, H, W)
        
        # 使用 OptVQ 量化器进行量化
        # VectorQuantizerSinkhorn.forward 接受 (B, C, H, W) 格式
        x_q, vq_loss, indices = self.quantizer(x_proj)
        
        # 计算 diff（用于兼容原有代码）
        # 离着不能够这样子只计算传播到encoder的误差,这样子就没有办法更新codebook了
        # diff 是量化误差的均值（保持为 tensor 以兼容原有代码）
        # diff = (x_q.detach() - x_proj).pow(2).mean()
        
        # 使用 stop-gradient 技巧
        quantize = x_proj + (x_q - x_proj).detach()
        
        return quantize, vq_loss, indices

    def embed_code(self, embed_id: torch.LongTensor) -> torch.FloatTensor:
        """
        从编码索引恢复量化特征
        
        Args:
            embed_id: 编码索引 (B, H, W) 或 (B, H*W)
            
        Returns:
            量化后的特征 (B, H, W, C) - 需要 permute 到 (B, C, H, W)
        """
        # 获取特征的空间尺寸
        if embed_id.ndim == 2:
            # (B, H*W) -> 需要推断 H, W
            B = embed_id.shape[0]
            H_W = embed_id.shape[1]
            # 假设是正方形
            H = W = int(H_W ** 0.5)
            assert H * W == H_W, f"Cannot infer spatial size from {embed_id.shape}"
            embed_id = embed_id.view(B, H, W)
        elif embed_id.ndim == 3:
            B, H, W = embed_id.shape
        else:
            raise ValueError(f"Unexpected embed_id shape: {embed_id.shape}")
        
        # 使用 OptVQ 的 embed_code 方法
        # 注意：VectorQuantizerSinkhorn.embed_code 返回 (B, C, H, W)
        # 但我们需要 (B, H, W, C) 格式以兼容原有代码
        x_q = self.quantizer.embed_code(embed_id, size=(H, W), code_format="image")
        
        # 转换为 (B, H, W, C) 格式
        # x_q 当前是 (B, C, H, W)，需要 permute 到 (B, H, W, C)
        x_q = x_q.permute(0, 2, 3, 1)
        
        return x_q

class Upscaler(HelperModule):
    def build(self,
            embed_dim: int,
            scaling_rates: list[int],
        ):

        self.stages = nn.ModuleList()
        for sr in scaling_rates:
            upscale_steps = int(log2(sr))
            layers = []
            for _ in range(upscale_steps):
                layers.append(nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(embed_dim))
                layers.append(nn.ReLU(inplace=True))
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x: torch.FloatTensor, stage: int) -> torch.FloatTensor:
        return self.stages[stage](x)

"""
    Main VQ-VAE-2 Module, capable of support arbitrary number of levels
    TODO: A lot of this class could do with a refactor. It works, but at what cost?
    TODO: Add disrete code decoding function
"""
class VQVAE(HelperModule):
    def build(self,
            in_channels: int                = 3,
            hidden_channels: int            = 128,
            res_channels: int               = 32,
            nb_res_layers: int              = 2,
            nb_levels: int                  = 3,
            embed_dim: int                  = 64,
            nb_entries: int                 = 512,
            scaling_rates: list[int]        = [8, 4, 2],
            # OptVQ 相关参数（传递给 CodeLayer）
            epsilon: float                  = 10.0,
            n_iters: int                    = 5,
            normalize_mode: str            = "all",
            use_prob: bool                  = True,
            beta: float                     = 1.0,
            loss_q_type: str                = "ce",
            use_norm: bool                  = False,
            use_proj: bool                  = True,
            logger=None,
            enable_stats: bool              = False
        ):
        self.nb_levels = nb_levels
        assert len(scaling_rates) == nb_levels, "Number of scaling rates not equal to number of levels!"

        self.encoders = nn.ModuleList([Encoder(in_channels, hidden_channels, res_channels, nb_res_layers, scaling_rates[0])])
        for i, sr in enumerate(scaling_rates[1:]):
            self.encoders.append(Encoder(hidden_channels, hidden_channels, res_channels, nb_res_layers, sr))

        # 准备 OptVQ 参数
        code_layer_kwargs = {
            'epsilon': epsilon,
            'n_iters': n_iters,
            'normalize_mode': normalize_mode,
            'use_prob': use_prob,
            'beta': beta,
            'loss_q_type': loss_q_type,
            'use_norm': use_norm,
            'use_proj': use_proj,
            'logger': logger,
            'enable_stats': enable_stats
        }
        
        self.codebooks = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.codebooks.append(CodeLayer(hidden_channels+embed_dim, embed_dim, nb_entries, **code_layer_kwargs))
        self.codebooks.append(CodeLayer(hidden_channels, embed_dim, nb_entries, **code_layer_kwargs))

        self.decoders = nn.ModuleList([Decoder(embed_dim*nb_levels, hidden_channels, in_channels, res_channels, nb_res_layers, scaling_rates[0])])
        for i, sr in enumerate(scaling_rates[1:]):
            self.decoders.append(Decoder(embed_dim*(nb_levels-1-i), hidden_channels, embed_dim, res_channels, nb_res_layers, sr))

        self.upscalers = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.upscalers.append(Upscaler(embed_dim, scaling_rates[1:len(scaling_rates) - i][::-1]))
        
        # 为了兼容训练脚本，添加以下属性
        # encoder: 指向第一个 encoder（用于测试和分析）
        self.encoder = self.encoders[0]
        # decoder: 指向最后一个 decoder（用于损失计算）
        self.decoder = self.decoders[0]
        # output_channels: 输出通道数
        self.output_channels = in_channels
        # vocab_size: 码本大小（使用第一个 codebook 的大小）
        self.vocab_size = nb_entries
        # vq_vae: 指向 codebooks（为了兼容训练脚本中的 vq_vae 访问）
        # 创建一个包装器，使 codebooks 可以像单个模块一样访问
        self.vq_vae = self.codebooks

    def forward(self, x):
        encoder_outputs = []
        code_outputs = []
        decoder_outputs = []
        upscale_counts = []
        id_outputs = []
        diffs = []

        for enc in self.encoders:
            if len(encoder_outputs):
                encoder_outputs.append(enc(encoder_outputs[-1]))
            else:
                encoder_outputs.append(enc(x))

        for l in range(self.nb_levels-1, -1, -1):
            codebook, decoder = self.codebooks[l], self.decoders[l]

            if len(decoder_outputs): # if we have previous levels to condition on
                code_q, code_d, emb_id = codebook(torch.cat([encoder_outputs[l], decoder_outputs[-1]], axis=1))
            else:
                code_q, code_d, emb_id = codebook(encoder_outputs[l])
            diffs.append(code_d)
            id_outputs.append(emb_id)

            code_outputs = [self.upscalers[i](c, upscale_counts[i]) for i, c in enumerate(code_outputs)]
            upscale_counts = [u+1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1)))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        # 返回原始格式（用于兼容原有代码）
        return decoder_outputs[-1], diffs, encoder_outputs, decoder_outputs, id_outputs
    
    def forward_training(self, x):
        """
        训练时使用的前向传播，返回与 VqVae2 兼容的格式
        
        Returns:
            Tuple[vq_loss, images_recon, perplexity, encoding_indices]:
            - vq_loss: VQ 损失（所有层级的 diff 之和）
            - images_recon: 重建图像
            - perplexity: 困惑度（平均）
            - encoding_indices: 码本索引（最后一层的索引）
        """
        images_recon, diffs, encoder_outputs, decoder_outputs, id_outputs = self.forward(x)
        
        # 计算总 VQ 损失（所有层级的 diff 之和）
        vq_loss = sum(diffs)
        
        # 计算困惑度：基于最后一层的 codebook 使用情况
        # 困惑度 = exp(H)，其中 H 是熵
        if id_outputs and len(id_outputs) > 0:
            # 使用最后一层的编码索引
            encoding_indices = id_outputs[-1]
            
            # 计算每个 codebook entry 的使用频率
            flat_indices = encoding_indices.flatten()
            vocab_size = self.codebooks[-1].n_embed
            
            # 计算每个 entry 的使用次数
            counts = torch.bincount(flat_indices, minlength=vocab_size).float()
            # 计算概率分布
            probs = counts / (counts.sum() + 1e-10)
            # 只考虑被使用的 entries
            used_probs = probs[probs > 0]
            
            if len(used_probs) > 0:
                # 计算熵：H = -sum(p * log(p))
                entropy = -torch.sum(used_probs * torch.log(used_probs + 1e-10))
                # 困惑度 = exp(H)
                perplexity = torch.exp(entropy)
            else:
                perplexity = torch.tensor(1.0, device=x.device)
        else:
            perplexity = torch.tensor(1.0, device=x.device)
            encoding_indices = None
        
        return vq_loss, images_recon, perplexity, encoding_indices

    def decode_codes(self, *cs):
        decoder_outputs = []
        code_outputs = []
        upscale_counts = []

        for l in range(self.nb_levels - 1, -1, -1):
            codebook, decoder = self.codebooks[l], self.decoders[l]
            code_q = codebook.embed_code(cs[l]).permute(0, 3, 1, 2)
            code_outputs = [self.upscalers[i](c, upscale_counts[i]) for i, c in enumerate(code_outputs)]
            upscale_counts = [u+1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1)))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        return decoder_outputs[-1]

if __name__ == '__main__':
    from helper import get_parameter_count
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nb_levels = 10 
    net = VQVAE(nb_levels=nb_levels, scaling_rates=[2]*nb_levels).to(device)
    print(f"Number of trainable parameters: {get_parameter_count(net)}")

    x = torch.randn(1, 3, 1024, 1024).to(device)
    _, diffs, enc_out, dec_out = net(x)
    print('\n'.join(str(y.shape) for y in enc_out))
    print()
    print('\n'.join(str(y.shape) for y in dec_out))
    print()
    print('\n'.join(str(y) for y in diffs))
