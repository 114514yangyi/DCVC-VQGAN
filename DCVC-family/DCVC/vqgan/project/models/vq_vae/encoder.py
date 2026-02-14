"""
VQ-VAE2 编码器模块 (用于视频序列)

该模块实现了用于视频数据的编码器，包含两个主要部分：
1. 空间编码 (Spatial Encoding): 对每一帧进行空间特征提取
2. 时间编码 (Temporal Encoding): 对多帧序列进行时间特征融合

输入格式: (batch * sequence_length, channels, height, width)
输出格式: (batch, n_init, h', w') - 其中 h', w' 是下采样后的空间维度
"""

from collections import OrderedDict
from functools import partial
from typing import Tuple

import attr
import torch
from einops import rearrange
from torch import nn





@attr.s(repr=False, eq=False)
class EncoderBlock(nn.Module):
    n_in: int = attr.ib(validator=lambda i, a, x: x >= 1)
    n_out: int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 == 0)
    n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)

    def __attrs_post_init__(self) -> None:
        super().__init__()
        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers ** 2)

        self.id_path = nn.Conv2d(in_channels=self.n_in, out_channels=self.n_out, kernel_size=1) \
            if self.n_in != self.n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([
            ('relu_1', nn.ReLU()),
            ('conv_1', nn.Conv2d(in_channels=self.n_in, out_channels=self.n_hid, kernel_size=3, padding=1)),
            ('relu_2', nn.ReLU()),
            ('conv_2', nn.Conv2d(in_channels=self.n_hid, out_channels=self.n_hid, kernel_size=3, padding=1)),
            ('relu_3', nn.ReLU()),
            ('conv_3', nn.Conv2d(in_channels=self.n_hid, out_channels=self.n_hid, kernel_size=3, padding=1)),
            ('relu_4', nn.ReLU()),
            ('conv_4', nn.Conv2d(in_channels=self.n_hid, out_channels=self.n_out, kernel_size=1)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


@attr.s(repr=False, eq=False)
class Encoder2(nn.Module):
    """
    VQ-VAE2 视频编码器
    
    该编码器采用分层编码策略：
    - 首先对每一帧进行空间编码（提取空间特征）
    - 然后将多帧特征在时间维度上融合（提取时间特征）
    
    架构设计：
    - 空间编码：使用多个编码组（groups），每个组包含多个编码块（blocks）
    - 时间编码：将多帧的空间特征在通道维度上拼接，然后通过分组卷积融合
    """
    
    # ========== 模型架构参数 ==========
    group_count: int = attr.ib()
    """
    编码组的数量，决定了空间下采样的层级数
    例如：group_count=4 表示有4个编码组，空间尺寸会下采样 (group_count-1) 次
    """
    
    n_hid: int = attr.ib(default=256, validator=lambda i, a, x: x >= 64)
    """
    初始隐藏特征维度，也是第一个编码组的特征维度
    后续组的特征维度会逐渐增加：n_hid, 2*n_hid, 3*n_hid, ...
    """
    
    n_blk_per_group: int = attr.ib(default=2, validator=lambda i, a, x: x >= 1)
    """
    每个编码组中包含的编码块（EncoderBlock）数量
    每个编码块都是一个残差块，用于特征提取
    """
    
    input_channels: int = attr.ib(default=3, validator=lambda i, a, x: x >= 1)
    """
    输入图像的通道数，通常为3（RGB）或1（灰度）
    """
    
    # ========== 输出和序列参数 ==========
    n_init: int = attr.ib(default=128, validator=lambda i, a, x: x > 0)
    """
    编码器输出的特征维度，也是 codebook 的嵌入维度
    这个维度决定了量化向量的维度
    """
    
    sequence_length: int = attr.ib(default=16, validator=lambda i, a, x: x > 0)
    """
    视频序列的长度，即一次处理多少帧
    输入格式是 (batch * sequence_length, channels, h, w)
    """
    
    # ========== 下采样控制 ==========
    downsample: bool = attr.ib(default=True)
    """
    是否在空间编码过程中进行下采样
    True: 每个编码组后使用 MaxPool2d 进行2倍下采样
    False: 不进行下采样，保持空间尺寸不变
    """

    def __attrs_post_init__(self):
        """
        初始化编码器架构
        
        构建过程：
        1. 创建空间编码器：处理每一帧的空间特征
        2. 创建时间编码器：融合多帧的时间特征
        """
        super().__init__()

        # ========== 准备构建函数 ==========
        blk_range = range(self.n_blk_per_group)  # 每个组中的块索引范围
        n_layers = self.group_count * self.n_blk_per_group  # 总层数，用于残差块的权重归一化
        
        # 卷积层创建函数
        make_conv = nn.Conv2d
        # 编码块创建函数（使用偏函数固定总层数）
        make_blk = partial(EncoderBlock, n_layers=n_layers)

        def make_grp(gid: int, n: int, n_prev, downsample: bool = True) -> Tuple[str, nn.Sequential]:
            """
            创建一个编码组
            
            Args:
                gid: 组ID
                n: 当前组的输出特征维度
                n_prev: 前一个组的输出特征维度（用于第一个块的输入）
                downsample: 是否在该组后下采样
            
            Returns:
                (组名, Sequential模块)
            """
            # 创建该组中的所有编码块
            # 第一个块的输入维度是 n_prev，后续块的输入输出都是 n
            blks = [(f'block_{i + 1}', make_blk(n_in=n_prev if i == 0 else n, n_out=n)) for
                    i in blk_range]
            # 如果需要下采样，添加最大池化层（2倍下采样）
            if downsample:
                blks += [('pool', nn.MaxPool2d(kernel_size=2))]
            return f'spatial_{gid}', nn.Sequential(OrderedDict(blks))

        # ========== 构建空间编码器 ==========
        # 空间编码器处理每一帧的空间特征
        # 输入: (batch * seq_length, input_channels, h, w)
        # 输出: (batch * seq_length, n * sequence_length, h', w')
        # 其中 n = group_count * n_hid，h', w' 是下采样后的尺寸
        
        # 第一层：输入卷积层，将输入通道转换为初始隐藏维度
        # 使用较大的卷积核 (7x7) 来捕获更大的感受野
        encode_blks_spatial = [('input', make_conv(
            in_channels=self.input_channels, 
            out_channels=self.n_hid, 
            kernel_size=7, 
            padding=3  # 保持空间尺寸不变
        ))]
        
        # # 构建多个编码组
        # # 每个组的特征维度逐渐增加：n_hid, 2*n_hid, 3*n_hid, ...
        # n, n_prev = self.n_hid, self.n_hid  # n: 当前组输出维度, n_prev: 前一组输出维度
        # for gid in range(1, self.group_count):
        #     # 当前组的输出维度 = (组ID + 1) * n_hid
        #     n = (gid + 1) * self.n_hid
        #     encode_blks_spatial.append(make_grp(gid=gid, n=n, n_prev=n_prev, downsample=self.downsample))
        #     n_prev = n  # 更新前一组维度

                # 构建多个编码组
        # 每个组的特征维度逐渐增加：n_hid, 2*n_hid, 3*n_hid, ...
        n, n_prev = self.n_hid, self.n_hid  # n: 当前组输出维度, n_prev: 前一组输出维度
        for gid in range(1, self.group_count):
            # 当前组的输出维度 = (组ID + 1) * n_hid
            n = (gid+1) * self.n_hid
            encode_blks_spatial.append(make_grp(gid=gid, n=n, n_prev=n_prev, downsample=self.downsample))
            n_prev = n  # 更新前一组维度
        
        # 最后一组：不进行下采样（因为后面要进行时间编码）
        encode_blks_spatial.append(make_grp(gid=self.group_count, n=n, n_prev=n_prev, downsample=False))

        # ========== 构建时间编码器 ==========
        # 时间编码器将多帧的空间特征在时间维度上融合
        # 输入形状: (batch * seq_length, n, h', w') 
        #   经过 rearrange 后: (batch, n * seq_length, h', w')
        # 输出形状: (batch, n_init, h', w')
        # 
        # 使用分组卷积 (groups=n) 来减少参数量，同时保持特征分离
        # 这样每个时间步的特征都被独立处理，然后通过分组卷积融合
        encode_blks_tempo = [
            (f'tempo', nn.Sequential(OrderedDict([
                ('relu', nn.ReLU()),  # 激活函数
                ('conv', make_conv(
                    in_channels=n ,  # 输入：所有帧的特征拼接
                    out_channels=self.n_init,              # 输出：codebook 维度
                    kernel_size=3, 
                    padding=1,  # 保持空间尺寸
                    groups=n  # 分组卷积，减少参数量
                ))
            ])))]

        # 组合成完整的模块
        self.blocks_spatial = nn.Sequential(OrderedDict(encode_blks_spatial))
        self.blocks_tempo = nn.Sequential(OrderedDict(encode_blks_tempo))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 输入张量，形状为 (batch * sequence_length, channels, height, width)
                   例如：batch=4, sequence_length=16 时，输入形状为 (64, 3, 256, 256)
        
        Returns:
            编码后的特征，形状为 (batch, n_init, h', w')
            其中 h', w' 是下采样后的空间维度
        """
        # 步骤1: 空间编码
        # 对每一帧独立进行空间特征提取
        # 输入: (batch * seq_length, channels, h, w)
        # 输出: (batch * seq_length, n, h', w') 其中 n = group_count * n_hid
        z = self.blocks_spatial(inputs)
        
        # 步骤2: 重新排列维度，将时间维度融合到通道维度
        # 从 (batch * seq_length, n, h', w') 
        # 转换为 (batch, n * seq_length, h', w')
        # 这样可以将多帧的特征在通道维度上拼接，便于后续的时间编码
        # z = rearrange(z, '(b d) c h w -> b (c d) h w', d=self.sequence_length)
        
        # 步骤3: 时间编码
        # 通过分组卷积融合多帧的时间特征
        # 输入: (batch, n * seq_length, h', w')
        # 输出: (batch, n_init, h', w')
        return self.blocks_tempo(z)
