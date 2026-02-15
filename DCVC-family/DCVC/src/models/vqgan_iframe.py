"""
VQGAN I-frame 模型包装器

提供与 CompressAI I-frame 模型兼容的接口，使 VQGAN 模型可以作为 I-frame 压缩模型使用。
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# 添加 VQGAN 项目路径
_VQGAN_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../vqgan/VQGAN")
)
if _VQGAN_ROOT not in sys.path:
    sys.path.insert(0, _VQGAN_ROOT)

from models.model_adapter import create_model


class VQGANIFrameModel(nn.Module):
    """
    VQGAN I-frame 模型包装器
    提供与 CompressAI I-frame 模型兼容的接口
    
    接口兼容性：
    - 输入：[B, C, H, W] Tensor，值范围 [0, 1]
    - 输出：dict，包含 'x_hat'（重建图像）和 'likelihoods'（用于 BPP 计算）
    """
    
    def __init__(self, vq_config_path, vq_checkpoint_path, device):
        super().__init__()
        self.device = device
        self.vq_config_path = vq_config_path
        self.vq_checkpoint_path = vq_checkpoint_path
        
        # 加载配置
        with open(vq_config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 加载 VQGAN 模型
        self.vqgan_model = self._load_vqgan_model()
        self.vqgan_model.eval()
        self.vqgan_model = self.vqgan_model.to(device)
        
        # 归一化和反归一化
        # VQGAN 期望输入范围 [-1, 1]
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.unnormalize = transforms.Normalize(
            mean=[-1.0, -1.0, -1.0],
            std=[2.0, 2.0, 2.0]
        )
        
        # 获取 codebook 大小（用于 BPP 计算）
        self.codebook_size = self._get_codebook_size()
    
    def _load_vqgan_model(self):
        """加载 VQGAN 模型"""
        model_args = self.config.get("model_args", {})
        model = create_model(config_path=self.vq_config_path, model_args=model_args)
        
        # 加载 checkpoint
        ckpt = torch.load(self.vq_checkpoint_path, map_location=self.device)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        
        return model
    
    def _get_codebook_size(self):
        """获取 codebook 大小（n_embed）"""
        model_args = self.config.get("model_args", {})
        codebook_size = model_args.get("n_embed", 1024)  # 默认 1024
        return codebook_size
    
    def forward(self, x):
        """
        前向传播，兼容 CompressAI 接口
        
        输入:
            x: [B, C, H, W] Tensor，值范围 [0, 1]
        
        输出:
            dict: {
                'x_hat': 重建图像 [B, C, H, W]，值范围 [0, 1]
                'likelihoods': 用于 BPP 计算的似然值（字典）
            }
        """
        # 1. 归一化到 [-1, 1]
        x_normalized = self.normalize(x)
        
        # 2. 编码和解码
        with torch.no_grad():
            inner_model = self.vqgan_model._get_model()
            quant, emb_loss, info = inner_model.encode(x_normalized)
            x_hat_normalized = inner_model.decode(quant)
        
        # 3. 反归一化到 [0, 1]
        x_hat = self.unnormalize(x_hat_normalized)
        x_hat = torch.clamp(x_hat, 0, 1)
        
        # 4. 计算 BPP（使用最简单的均匀分布假设）
        likelihoods = self._calculate_likelihoods(info, x.shape)
        
        return {
            'x_hat': x_hat,
            'likelihoods': likelihoods,
            'indices': info[2] if info is not None and len(info) >= 3 else None  # 可选：保存 indices 用于调试
        }
    
    def _calculate_likelihoods(self, info, input_shape):
        """
        计算 codebook indices 的似然值
        
        使用最简单的均匀分布假设：
        - 每个 codebook index 出现的概率 = 1 / codebook_size
        - BPP = log2(codebook_size) / (H * W) * (H' * W') / (H * W)
        - 简化后：BPP = log2(codebook_size) * (H' * W') / (H * W)
        
        为了兼容 CompressAI 的 BPP 计算方式，我们返回一个与输入图像尺寸匹配的 likelihood。
        由于 VQGAN 使用下采样（通常 4 倍），我们需要将 latent 空间的概率映射到像素空间。
        
        输入:
            info: VQGAN encode 返回的 info 元组，info[2] 是 indices
            input_shape: 输入图像的形状 [B, C, H, W]
        
        输出:
            dict: {'y': likelihood tensor}，形状与 CompressAI 兼容
        """
        B, C, H, W = input_shape
        
        # 获取 indices 和下采样后的尺寸
        if info is not None and len(info) >= 3:
            indices = info[2]  # [B, H', W'] 或 [B*H'*W']
            if indices.ndim == 1:
                # VQGAN 通常下采样 4 倍
                h_down = H // 4
                w_down = W // 4
                indices = indices.view(B, h_down, w_down)
            else:
                h_down, w_down = indices.shape[1], indices.shape[2]
        else:
            # 如果没有 indices，使用下采样后的尺寸估算
            h_down = H // 4
            w_down = W // 4
        
        # 均匀分布假设：每个 index 的概率 = 1 / codebook_size
        uniform_prob = 1.0 / self.codebook_size
        
        # 为了兼容 CompressAI 的 BPP 计算方式，我们需要返回一个与输入图像尺寸相关的 likelihood
        # CompressAI 的 likelihoods 通常是按像素计算的，但 VQGAN 是按下采样的 latent 空间计算的
        # 我们创建一个形状为 (B, 1, H, W) 的 likelihood，每个位置的值代表该像素对应的 latent 位置的概率
        # 由于下采样，每个 latent 位置对应 4x4=16 个像素
        
        # 方法：创建一个 (B, 1, H, W) 的 likelihood，每个 latent 位置的值均匀分布到对应的像素区域
        # 每个 latent 位置的概率是 uniform_prob，对应 16 个像素
        # 所以每个像素对应的概率也是 uniform_prob（因为每个像素都属于某个 latent 位置）
        
        # 创建与输入图像尺寸匹配的 likelihood
        # 形状: (B, 1, H, W)，每个位置的值是 uniform_prob
        likelihood = torch.full(
            (B, 1, H, W),
            uniform_prob,
            dtype=torch.float32,
            device=self.device
        )
        
        return {'y': likelihood}
    
    def compress(self, x):
        """
        压缩接口（可选，用于真实压缩）
        当前实现仅返回编码结果，不进行实际熵编码
        """
        # 可以在这里实现实际的熵编码逻辑
        # 例如使用 Transformer 或其他熵编码模型
        return self.forward(x)
    
    def decompress(self, strings):
        """
        解压缩接口（可选，用于真实解压缩）
        当前实现暂不支持
        """
        raise NotImplementedError("VQGAN decompress not implemented yet")
