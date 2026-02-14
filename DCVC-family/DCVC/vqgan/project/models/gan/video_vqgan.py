import torch
from torch import nn


class VQGANLossHelper:
    """
    视频版 VQGAN 的损失计算辅助函数。

    - adopt_weight: 在指定步数前关闭判别器损失
    - calculate_lambda: 按照论文中的做法，使用最后一层权重梯度比值自适应调节 GAN 权重
    """

    @staticmethod
    def adopt_weight(disc_factor: float, step: int, threshold: int, value: float = 0.0) -> float:
        if step < threshold:
            return value
        return disc_factor

    @staticmethod
    def calculate_lambda(decoder: nn.Module, perceptual_loss: torch.Tensor, gan_loss: torch.Tensor,
                         clamp_max: float = 1e2) -> torch.Tensor:
        """
        根据论文 `Taming Transformers` 的策略，使用 decoder 最后一层权重的梯度范数比例
        来动态平衡 GAN 与感知重建损失。
        """
        # 选取 decoder 中最后一个需要梯度的权重参数
        target_param = None
        for p in decoder.parameters():
            if p.requires_grad:
                target_param = p
        if target_param is None:
            # fallback，避免训练中断
            print("No target parameter found for lambda calculation")
            return perceptual_loss.detach() * 0 + 1.0

        p_grads = torch.autograd.grad(perceptual_loss, target_param, retain_graph=True, allow_unused=True)[0]
        g_grads = torch.autograd.grad(gan_loss, target_param, retain_graph=True, allow_unused=True)[0]

        if p_grads is None or g_grads is None:
            print("No gradient found for lambda calculation 2")
            return perceptual_loss.detach() * 0 + 1.0

        lam = torch.norm(p_grads) / (torch.norm(g_grads) + 1e-4)
        lam = torch.clamp(lam, 0, clamp_max).detach()
        return 0.8 * lam

