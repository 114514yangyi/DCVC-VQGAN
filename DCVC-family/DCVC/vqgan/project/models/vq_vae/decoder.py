"""
VQ-VAE2 解码器模块 (用于视频序列)

该模块实现了用于视频数据的解码器，是编码器的逆过程：
1. 时间解码 (Temporal Decoding): 将融合的时间特征分解为多帧特征
2. 空间解码 (Spatial Decoding): 对每一帧进行空间特征重建

输入格式: (batch, n_init, h', w') - 量化后的特征
输出格式: (batch * sequence_length, channels, height, width) - 重建的视频帧
"""

from collections import OrderedDict
from functools import partial
from typing import Tuple

import attr
import torch
from einops import rearrange
from torch import nn



@attr.s(eq=False, repr=False)
class DecoderBlock(nn.Module):
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
            ('conv_1', nn.Conv2d(in_channels=self.n_in, out_channels=self.n_hid, kernel_size=1)),
            ('relu_2', nn.ReLU()),
            ('conv_2', nn.Conv2d(in_channels=self.n_hid, out_channels=self.n_hid, kernel_size=3, padding=1)),
            ('relu_3', nn.ReLU()),
            ('conv_3', nn.Conv2d(in_channels=self.n_hid, out_channels=self.n_hid, kernel_size=3, padding=1)),
            ('relu_4', nn.ReLU()),
            ('conv_4', nn.Conv2d(in_channels=self.n_hid, out_channels=self.n_out, kernel_size=3, padding=1))]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


@attr.s(eq=False, repr=False)
class Decoder2(nn.Module):
    """
    VQ-VAE2 视频解码器
    
    该解码器采用分层解码策略，与编码器对称：
    - 首先进行时间解码（将融合特征分解为多帧特征）
    - 然后对每一帧进行空间解码（重建空间特征）
    
    架构设计：
    - 时间解码：通过分组卷积将 codebook 维度扩展为多帧特征
    - 空间解码：使用多个解码组（groups），每个组包含多个解码块（blocks）
    - 特征维度逐渐减少，空间尺寸逐渐上采样
    """
    
    # ========== 模型架构参数 ==========
    group_count: int = attr.ib()
    """
    解码组的数量，应该与编码器的 group_count 相同
    决定了空间上采样的层级数
    """
    
    # ========== 输入和输出参数 ==========
    n_init: int = attr.ib(default=128, validator=lambda i, a, x: x >= 8)
    """
    输入特征维度，即 codebook 的嵌入维度
    必须与编码器的 n_init 相同
    """
    
    n_hid: int = attr.ib(default=256, validator=lambda i, a, x: x >= 64)
    """
    隐藏特征维度，也是最后一个解码组的特征维度
    应该与编码器的 n_hid 相同
    """
    
    n_blk_per_group: int = attr.ib(default=2, validator=lambda i, a, x: x >= 1)
    """
    每个解码组中包含的解码块（DecoderBlock）数量
    应该与编码器的 n_blk_per_group 相同
    """
    
    output_channels: int = attr.ib(default=3, validator=lambda i, a, x: x >= 1)
    """
    输出图像的通道数，通常为3（RGB）或1（灰度）
    应该与编码器的 input_channels 相同
    """
    
    # ========== 序列参数 ==========
    sequence_length: int = attr.ib(default=16, validator=lambda i, a, x: x > 0)
    """
    视频序列的长度，必须与编码器的 sequence_length 相同
    """
    
    # ========== 上采样控制 ==========
    upsample: bool = attr.ib(default=True)
    """
    是否在空间解码过程中进行上采样
    True: 每个解码组后使用 Upsample 进行2倍上采样
    False: 不进行上采样，保持空间尺寸不变
    """

    def __attrs_post_init__(self) -> None:
        """
        初始化解码器架构
        
        构建过程：
        1. 创建时间解码器：将融合特征分解为多帧特征
        2. 创建空间解码器：重建每一帧的空间特征
        """
        super().__init__()

        # ========== 准备构建函数 ==========
        blk_range = range(self.n_blk_per_group)  # 每个组中的块索引范围
        n_layers = self.group_count * self.n_blk_per_group  # 总层数，用于残差块的权重归一化
        
        # 卷积层创建函数
        make_conv = nn.Conv2d
        # 解码块创建函数（使用偏函数固定总层数）
        make_blk = partial(DecoderBlock, n_layers=n_layers)

        def make_grp(gid: int, n: int, n_prev: int, upsample: bool = True) -> Tuple[str, nn.Sequential]:
            """
            创建一个解码组
            
            Args:
                gid: 组ID
                n: 当前组的输出特征维度
                n_prev: 前一个组的输出特征维度（用于第一个块的输入）
                upsample: 是否在该组后上采样
            
            Returns:
                (组名, Sequential模块)
            """
            # 创建该组中的所有解码块
            # 第一个块的输入维度是 n_prev，后续块的输入输出都是 n
            blks = [(f'block_{i + 1}', make_blk(n_in=n_prev if i == 0 else n, n_out=n)) for
                    i in blk_range]
            # 如果需要上采样，添加上采样层（2倍上采样）
            if upsample:
                blks += [('upsample', nn.Upsample(scale_factor=2, mode='nearest'))]
            return f'spatial_{gid}', nn.Sequential(OrderedDict(blks))

        # ========== 构建时间解码器 ==========
        # 时间解码器将融合的时间特征分解为多帧特征
        # 输入: (batch, n_init, h', w')
        # 输出: (batch, n * sequence_length, h', w')
        # 其中 n = group_count * n_hid
        n = (self.group_count) * self.n_hid  # 每帧的特征维度
        
        # 使用分组卷积将 codebook 维度扩展为多帧特征
        # groups=n 表示将输出通道分成 n 组，每组处理一个时间步的特征
        decode_blks_tempo = [
            (f'tempo', nn.Sequential(OrderedDict([
                ('conv', make_conv(
                    in_channels=self.n_init,                    # 输入：codebook 维度
                    out_channels=n,      # 输出：所有帧的特征拼接
                    kernel_size=3, 
                    padding=1,  # 保持空间尺寸
                    groups=n  # 分组卷积，每个组处理一个时间步
                )),
                ('relu', nn.ReLU())  # 激活函数
            ])))]

        # ========== 构建空间解码器 ==========
        # 空间解码器重建每一帧的空间特征
        # 输入: (batch, n * sequence_length, h', w')
        #   经过 rearrange 后: (batch * sequence_length, n, h', w')
        # 输出: (batch * sequence_length, output_channels, h, w)
        
        decode_blks_spatial = []
        # 第一个解码组的输入维度是 n = group_count * n_hid
        n_prev = (self.group_count) * self.n_hid
        # 特征维度逐渐减少：group_count*n_hid, (group_count-1)*n_hid, ..., n_hid
        for gid in range(1, self.group_count):
            # 当前组的输出维度 = (group_count - gid) * n_hid
            n = (self.group_count - gid) * self.n_hid
            decode_blks_spatial.append(make_grp(gid=gid, n=n, n_prev=n_prev, upsample=self.upsample))
            n_prev = n  # 更新前一组维度
        
        # 最后一组：输出维度为 n_hid，不进行上采样
        decode_blks_spatial.append(make_grp(gid=self.group_count, n=self.n_hid, n_prev=n_prev, upsample=False))
        
        # 输出层：将特征维度转换为输出通道数
        decode_blks_spatial.append(
            ('output', nn.Sequential(OrderedDict([
                ('relu', nn.ReLU()),  # 激活函数
                ('conv', make_conv(
                    in_channels=self.n_hid, 
                    out_channels=self.output_channels, 
                    kernel_size=1  # 1x1 卷积，只改变通道数
                )),
            ]))))
        
        # 组合成完整的模块
        self.blocks_tempo = nn.Sequential(OrderedDict(decode_blks_tempo))
        self.blocks_spatial = nn.Sequential(OrderedDict(decode_blks_spatial))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, n_init, h', w')
               这是经过量化后的特征，来自编码器的输出
        
        Returns:
            重建的视频帧，形状为 (batch * sequence_length, channels, height, width)
            例如：batch=4, sequence_length=16 时，输出形状为 (64, 3, 256, 256)
        """
        # 步骤1: 时间解码
        # 将融合的时间特征分解为多帧特征
        # 输入: (batch, n_init, h', w')
        # 输出: (batch, n * sequence_length, h', w')
        # 其中 n = group_count * n_hid
        z = self.blocks_tempo(x)
        
        # 步骤2: 重新排列维度，将通道维度中的时间信息分离出来
        # 从 (batch, n * sequence_length, h', w')
        # 转换为 (batch * sequence_length, n, h', w')
        # 这样可以将多帧特征分离，便于后续的空间解码
        # z = rearrange(z, 'b (c d) h w -> (b d) c h w', d=self.sequence_length)
        
        # 步骤3: 空间解码
        # 对每一帧独立进行空间特征重建
        # 输入: (batch * sequence_length, n, h', w')
        # 输出: (batch * sequence_length, output_channels, h, w)
        return self.blocks_spatial(z)
