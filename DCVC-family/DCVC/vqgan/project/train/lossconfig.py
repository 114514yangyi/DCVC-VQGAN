"""
损失计算配置模块

将所有损失计算逻辑封装到 LossCalculator 类中，包括：
- 重建损失（MSE, L1）
- VQ 损失
- 感知损失（LPIPS）
- GAN 损失（生成器和判别器）
"""
import json
import os
import math
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from types import SimpleNamespace
from models.gan.video_vqgan import VQGANLossHelper
from models.gan.discriminator import Discriminator
from models.gan.lpips import LPIPS


def load_config(config_path: str = 'param.json') -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，默认为 'param.json'
    
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def get_train_args_from_config(config_path: str = 'param.json') -> Dict[str, Any]:
    """
    从配置文件中获取训练参数（train_args）
    
    Args:
        config_path: 配置文件路径，默认为 'param.json'
    
    Returns:
        train_args 字典
    """
    config = load_config(config_path)
    return config.get('train_args', {})


class LossCalculator:
    """
    损失计算器
    
    统一管理所有损失的计算，包括：
    - 重建损失（MSE, L1）
    - VQ 损失
    - 感知损失（LPIPS）
    - GAN 损失（生成器和判别器）
    - Taming VQGAN 损失（如果使用 taming 模型）
    """
    
    def __init__(
        self,
        decoder: Optional[torch.nn.Module] = None,
        num_steps: int = 100000,
        # 用于初始化判别器和感知损失函数的参数
        output_channels: int = 3,
        device: Optional[torch.device] = None,
        base_lr: float = 1e-4,
        config_path: str = 'param.json',
        model: Optional[torch.nn.Module] = None  # 添加模型参数，用于检测 taming 模型
    ):
        """
        初始化损失计算器
        
        从配置文件中读取以下参数：
        - use_gan: 是否使用 GAN 训练
        - disc_factor: 判别器损失权重因子
        - disc_start: 开始使用判别器的步数
        - pixel_loss_factor: 像素损失权重
        - perceptual_loss_factor: 感知损失权重
        - disc_lr: 判别器学习率
        - disc_beta1: 判别器优化器的 beta1
        - disc_beta2: 判别器优化器的 beta2
        - use_taming_loss: 是否使用 taming 模型的损失函数（如果模型是 taming 模型）
        
        Args:
            decoder: 解码器模型（用于计算 lambda_gan）
            num_steps: 总训练步数（用于噪声衰减计算）
            output_channels: 模型输出通道数（用于初始化判别器）
            device: 设备（如果为 None，则从 decoder 获取）
            base_lr: 基础学习率（用于判别器，如果配置文件中 disc_lr 为 None）
            config_path: 配置文件路径，默认为 'param.json'
            model: 模型对象（用于检测是否是 taming 模型）
        """
        # 从配置文件读取参数
        train_args = get_train_args_from_config(config_path)
        
        self.model = model
        self.use_gan = train_args.get('use_gan', True)
        self.disc_factor = train_args.get('disc_factor', 1.0)
        self.disc_start = train_args.get('disc_start', 20000)
        self.pixel_loss_factor = train_args.get('pixel_loss_factor', 1.0)
        self.perceptual_loss_factor = train_args.get('perceptual_loss_factor', 1.0)
        self.use_taming_loss = train_args.get('use_taming_loss', None)  # None 表示自动检测
        disc_lr = train_args.get('disc_lr', None)
        disc_beta1 = train_args.get('disc_beta1', 0.5)
        disc_beta2 = train_args.get('disc_beta2', 0.9)
        
        # 如果配置文件中 disc_lr 为 None，使用 base_lr
        if disc_lr is None:
            disc_lr = base_lr
        
        self.decoder = decoder
        self.num_steps = num_steps
        
        # 检测是否是 taming 模型
        self.is_taming_model = self._is_taming_model()
        
        # 保存原始的 use_gan 设置（用于 taming 模型）
        original_use_gan = self.use_gan
        
        # 如果使用 taming 模型且未明确禁用 taming 损失，则使用 taming 损失
        if self.is_taming_model and self.use_taming_loss is not False:
            self.use_taming_loss = True
            # 从 taming 模型中获取损失函数和判别器
            taming_model = self._get_taming_model()
            self.taming_loss_fn = taming_model.loss
            self.discriminator = taming_model.loss.discriminator
            self.perceptual_loss_fn = None  # taming 损失函数内部已包含感知损失
            self.optimizer_disc = None  # taming 模型的判别器优化器在模型内部管理
            
            # 尊重用户的 use_gan 设置
            # 如果 use_gan=False，通过设置 disc_factor=0 来禁用判别器影响
            # 但判别器仍然会被创建（因为 taming 损失函数总是创建判别器）
            if not original_use_gan:
                # 如果用户明确禁用 GAN，将 disc_factor 设置为 0
                # 这样判别器损失不会影响总损失，但判别器仍然存在（因为损失函数已创建）
                self.disc_factor = 0.0
                # 注意：taming 损失函数内部的 disc_factor 也需要设置为 0
                # 这会在损失计算时通过 disc_factor 参数控制
                print("警告: 检测到 use_gan=False，但 taming 模型的损失函数已包含判别器。"
                      "将 disc_factor 设置为 0 以禁用判别器影响。"
                      "注意：判别器仍会被创建，但不会影响训练。")
            else:
                # 如果 use_gan=True，保持原有设置
                self.use_gan = True
        else:
            # 使用传统的损失计算方式
            self.use_taming_loss = False
            self.taming_loss_fn = None
        # 初始化判别器和感知损失函数
        self.discriminator = None
        self.perceptual_loss_fn = None
        self.optimizer_disc = None
        
        if self.use_gan:
            # 确定设备
            if device is None:
                if decoder is not None:
                    device = next(decoder.parameters()).device
                else:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 初始化判别器
            disc_args = SimpleNamespace(image_channels=output_channels)
            self.discriminator = Discriminator(disc_args).to(device)
            
            # 初始化感知损失函数
            self.perceptual_loss_fn = LPIPS().eval().to(device)
            
            # 初始化判别器优化器
            disc_lr = disc_lr if disc_lr is not None else base_lr
            self.optimizer_disc = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=disc_lr,
                betas=(disc_beta1, disc_beta2)
            )
    
    def _is_taming_model(self) -> bool:
        """检测模型是否是 taming 模型"""
        if self.model is None:
            return False
        
        # 检查模型是否是 TamingVQGANAdapter
        from models.model_adapter import TamingVQGANAdapter
        if isinstance(self.model, TamingVQGANAdapter):
            return True
        
        # 检查内部模型是否是 taming 模型
        try:
            model = self.model._get_model() if hasattr(self.model, '_get_model') else self.model
            from models.taming.models.vqgan import VQModel, EMAVQ
            return isinstance(model, (VQModel, EMAVQ))
        except:
            return False
    
    def _get_taming_model(self):
        """获取 taming 模型对象"""
        if hasattr(self.model, '_get_model'):
            return self.model._get_model()
        return self.model
    
    def get_discriminator(self) -> Optional[torch.nn.Module]:
        """获取判别器模型"""
        return self.discriminator
    
    def get_perceptual_loss_fn(self) -> Optional[torch.nn.Module]:
        """获取感知损失函数"""
        return self.perceptual_loss_fn
    
    def get_optimizer_disc(self) -> Optional[torch.optim.Optimizer]:
        """获取判别器优化器"""
        return self.optimizer_disc
    
    def step_discriminator(self, gan_loss: torch.Tensor, step: int):
        """
        执行判别器优化步骤
        
        注意：对于 taming 模型，判别器优化在模型内部管理，这里不需要执行。
        
        Args:
            gan_loss: 判别器损失
            step: 当前训练步数
        """
        # taming 模型的判别器优化在模型内部管理
        if self.use_taming_loss:
            return
        
        # 传统模型的判别器优化
        if self.use_gan and self.optimizer_disc is not None and step % 3 == 0:
            self.optimizer_disc.zero_grad()
            gan_loss.backward()
            self.optimizer_disc.step()
    
    def get_taming_optimizers(self):
        """
        获取 taming 模型的优化器（如果使用 taming 模型）
        
        注意：taming 模型应该自己管理学习率（通过 configure_optimizers），
        这样更符合原模型的实现方式。
        
        Returns:
            (opt_ae, opt_disc): 自编码器优化器和判别器优化器，如果不是 taming 模型则返回 None
        """
        if not self.use_taming_loss:
            return None, None
        
        taming_model = self._get_taming_model()
        
        # 直接调用 taming 模型的 configure_optimizers 方法
        # 这样可以让模型自己管理学习率，更符合原实现
        # configure_optimizers 返回 (optimizers, schedulers)
        optimizers, schedulers = taming_model.configure_optimizers()
        
        # taming 模型返回两个优化器：[opt_ae, opt_disc]
        if isinstance(optimizers, list) and len(optimizers) >= 2:
            opt_ae = optimizers[0]
            opt_disc = optimizers[1]
        elif isinstance(optimizers, list) and len(optimizers) == 1:
            # 某些变体可能只有一个优化器（如 VQNoDiscModel）
            opt_ae = optimizers[0]
            opt_disc = None
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.warning("Taming 模型只返回一个优化器，判别器优化器为 None")
            else:
                print("警告: Taming 模型只返回一个优化器，判别器优化器为 None")
        else:
            # 如果返回单个优化器（不是列表）
            opt_ae = optimizers
            opt_disc = None
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.warning("Taming 模型返回单个优化器，判别器优化器为 None")
            else:
                print("警告: Taming 模型返回单个优化器，判别器优化器为 None")
        
        # 记录使用的学习率（从优化器中获取）
        if opt_ae is not None:
            ae_lr = opt_ae.param_groups[0]['lr'] if opt_ae.param_groups else None
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.info(f"Taming 模型自编码器学习率: {ae_lr}")
            else:
                print(f"Taming 模型自编码器学习率: {ae_lr}")
        if opt_disc is not None:
            disc_lr = opt_disc.param_groups[0]['lr'] if opt_disc.param_groups else None
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.info(f"Taming 模型判别器学习率: {disc_lr}")
            else:
                print(f"Taming 模型判别器学习率: {disc_lr}")
        
        return opt_ae, opt_disc
    
    def load_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        从 checkpoint 加载判别器和优化器状态
        
        Args:
            checkpoint: checkpoint 字典
        """
        if self.use_gan and self.discriminator is not None:
            disc_state = checkpoint.get('discriminator_state')
            opt_disc_state = checkpoint.get('optimizer_disc')
            if disc_state is not None:
                self.discriminator.load_state_dict(disc_state)
            if opt_disc_state is not None and self.optimizer_disc is not None:
                self.optimizer_disc.load_state_dict(opt_disc_state)
    
    def get_checkpoint_state(self) -> Dict[str, Any]:
        """
        获取判别器和优化器的状态，用于保存 checkpoint
        
        Returns:
            包含判别器和优化器状态的字典
        """
        state = {}
        if self.use_gan and self.discriminator is not None:
            state['discriminator_state'] = self.discriminator.state_dict()
        if self.use_gan and self.optimizer_disc is not None:
            state['optimizer_disc'] = self.optimizer_disc.state_dict()
        return state
    
    def compute_reconstruction_loss(
        self,
        images: torch.Tensor,
        images_recon: torch.Tensor,
        use_l1: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算重建损失
        
        Args:
            images: 原始图像 [B*D, C, H, W]
            images_recon: 重建图像 [B*D, C, H, W]
            use_l1: 是否使用 L1 损失，否则使用 MSE
        
        Returns:
            (recon_error, pixel_loss): MSE 损失和 L1 损失
        """
        recon_error = F.mse_loss(images_recon, images)
        pixel_loss = F.l1_loss(images_recon, images) if use_l1 else recon_error
        return recon_error, pixel_loss
    
    def compute_perceptual_loss(
        self,
        images: torch.Tensor,
        images_recon: torch.Tensor
    ) -> torch.Tensor:
        """
        计算感知损失（LPIPS）
        
        Args:
            images: 原始图像 [B*D, C, H, W]
            images_recon: 重建图像 [B*D, C, H, W]
        
        Returns:
            perceptual_loss: 感知损失标量
        """
        if self.perceptual_loss_fn is None:
            device = images.device
            return torch.tensor(0.0, device=device)
        return self.perceptual_loss_fn(images, images_recon).mean()
    
    def compute_gan_generator_loss(
        self,
        images_recon: torch.Tensor,
        step: int
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        计算 GAN 生成器损失
        
        Args:
            images_recon: 重建图像 [B*D, C, H, W]
            step: 当前训练步数
        
        Returns:
            (g_loss, lambda_gan, disc_factor): 生成器损失、lambda 值和判别器权重因子
        """
        if not self.use_gan or self.discriminator is None:
            device = images_recon.device
            return torch.tensor(0.0, device=device), 0.0, 0.0
        
        # 计算判别器权重因子
        disc_factor = VQGANLossHelper.adopt_weight(self.disc_factor, step, self.disc_start)
        
        # 生成器损失：让判别器认为生成图像是真实的
        g_fake = self.discriminator(images_recon)
        g_loss = -g_fake.mean()
        
        # 计算 lambda_gan（需要感知损失和像素损失，这里先返回0，在 compute_total_loss 中计算）
        lambda_gan = 0.0
        
        return g_loss, lambda_gan, disc_factor
    
    def compute_gan_discriminator_loss(
        self,
        images: torch.Tensor,
        images_recon: torch.Tensor,
        step: int,
        disc_factor: float
    ) -> torch.Tensor:
        """
        计算 GAN 判别器损失
        
        Args:
            images: 原始图像 [B*D, C, H, W]
            images_recon: 重建图像 [B*D, C, H, W]
            step: 当前训练步数
            disc_factor: 判别器权重因子
        
        Returns:
            gan_loss: 判别器损失
        """
        if not self.use_gan or self.discriminator is None:
            device = images.device
            return torch.tensor(0.0, device=device)
        
        # 给判别器输入添加逐渐下降的噪声
        noise_scale = max(0, 0.01 * (1 - 2 * step / self.num_steps))
        images_noisy = images + torch.randn_like(images) * noise_scale
        images_recon_noisy = images_recon + torch.randn_like(images_recon) * noise_scale
        
        # 判别器损失：区分真实图像和生成图像
        d_real = self.discriminator(images_noisy.detach())
        d_fake = self.discriminator(images_recon_noisy.detach())
        d_loss_real = F.relu(1. - d_real).mean()
        d_loss_fake = F.relu(1. + d_fake).mean()
        gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)
        
        return gan_loss
    
    def compute_total_loss(
        self,
        images: torch.Tensor,
        images_recon: torch.Tensor,
        vq_loss: torch.Tensor,
        step: int,
        logger: Optional[Any] = None,
        optimizer_idx: int = 0  # 添加优化器索引，用于 taming 模型的双优化器训练
    ) -> Dict[str, Any]:
        """
        计算总损失和所有子损失
        
        Args:
            images: 原始图像 [B*D, C, H, W]
            images_recon: 重建图像 [B*D, C, H, W]
            vq_loss: VQ 损失
            step: 当前训练步数
            logger: 日志记录器（可选）
            optimizer_idx: 优化器索引（0=自编码器，1=判别器），用于 taming 模型
        
        Returns:
            包含所有损失的字典：
            {
                'total_loss': 总损失,
                'recon_error': MSE 重建损失,
                'pixel_loss': L1 像素损失,
                'vq_loss': VQ 损失,
                'perceptual_loss': 感知损失,
                'perceptual_rec_loss': 感知重建损失（加权后的感知+像素损失）,
                'g_loss': GAN 生成器损失,
                'gan_loss': GAN 判别器损失,
                'disc_factor': 判别器权重因子,
                'lambda_gan': GAN lambda 值,
                'log_dict': 详细日志字典（taming 模型使用）
            }
        """
        # 如果使用 taming 损失函数，直接调用 taming 的损失计算
        if self.use_taming_loss:
            return self._compute_taming_loss(images, images_recon, vq_loss, step, optimizer_idx)
        
        # 否则使用传统的损失计算方式
        device = images.device
        
        # 计算重建损失
        recon_error, pixel_loss = self.compute_reconstruction_loss(
            images, images_recon, use_l1=self.use_gan
        )
        
        # 初始化所有损失
        perceptual_loss = torch.tensor(0.0, device=device)
        perceptual_rec_loss = torch.tensor(0.0, device=device)
        g_loss = torch.tensor(0.0, device=device)
        gan_loss = torch.tensor(0.0, device=device)
        disc_factor = 0.0
        lambda_gan = 0.0
        
        if self.use_gan:
            # 计算判别器权重因子
            disc_factor = VQGANLossHelper.adopt_weight(self.disc_factor, step, self.disc_start)
            
            # 计算感知损失
            perceptual_loss = self.compute_perceptual_loss(images, images_recon)
            
            # 计算感知重建损失（加权后的感知+像素损失）
            perceptual_rec_loss = (
                self.perceptual_loss_factor * perceptual_loss +
                self.pixel_loss_factor * pixel_loss
            )
            
            # 计算生成器损失
            g_fake = self.discriminator(images_recon)
            g_loss = -g_fake.mean()
            
            # 计算 lambda_gan
            lambda_gan = VQGANLossHelper.calculate_lambda(
                decoder=self.decoder,
                perceptual_loss=perceptual_rec_loss,
                gan_loss=g_loss
            )
            
            # 增加一个缓慢缩放
            lambda_gan = lambda_gan * min(1.0, math.fabs(step - self.disc_start) / (2 * self.disc_start))
            
            # 计算总损失（VQ-GAN）
            total_loss = perceptual_rec_loss + vq_loss + disc_factor * lambda_gan * g_loss
        else:
            # 不使用 GAN，使用简单的 VQ-VAE 损失:这里只使用简单的L2损失
            total_loss = recon_error + vq_loss
        
        # 计算判别器损失（用于判别器训练）
        gan_loss = self.compute_gan_discriminator_loss(
            images, images_recon, step, disc_factor
        )
        
        return {
            'total_loss': total_loss,
            'recon_error': recon_error,
            'pixel_loss': pixel_loss,
            'vq_loss': vq_loss,
            'perceptual_loss': perceptual_loss,
            'perceptual_rec_loss': perceptual_rec_loss,
            'g_loss': g_loss,
            'gan_loss': gan_loss,
            'disc_factor': disc_factor,
            'lambda_gan': lambda_gan
        }
    
    def _compute_taming_loss(
        self,
        images: torch.Tensor,
        images_recon: torch.Tensor,
        vq_loss: torch.Tensor,
        step: int,
        optimizer_idx: int
    ) -> Dict[str, Any]:
        """
        使用 taming 模型的损失函数计算损失
        
        Args:
            images: 原始图像
            images_recon: 重建图像
            vq_loss: VQ 损失（codebook_loss）
            step: 当前训练步数
            optimizer_idx: 优化器索引（0=自编码器，1=判别器）
        
        Returns:
            损失字典
        """
        taming_model = self._get_taming_model()
        
        # 获取解码器的最后一层（用于自适应权重计算）
        last_layer = taming_model.get_last_layer() if hasattr(taming_model, 'get_last_layer') else None
        
        # 调用 taming 损失函数
        # taming 的损失函数期望 vq_loss 是标量或形状为 [B] 的张量
        if vq_loss.dim() > 0:
            # 如果是多维张量，取均值
            codebook_loss = vq_loss.mean()
        else:
            codebook_loss = vq_loss
        
        # 如果 use_gan=False，临时修改损失函数的 disc_factor 为 0
        # 这样可以禁用判别器的影响（但判别器仍然会被创建，因为损失函数已初始化）
        original_disc_factor = None
        if not self.use_gan and hasattr(self.taming_loss_fn, 'disc_factor'):
            original_disc_factor = self.taming_loss_fn.disc_factor
            self.taming_loss_fn.disc_factor = 0.0
        
        # 调用 taming 损失函数
        loss, log_dict = self.taming_loss_fn(
            codebook_loss=codebook_loss,
            inputs=images,
            reconstructions=images_recon,
            optimizer_idx=optimizer_idx,
            global_step=step,
            last_layer=last_layer,
            cond=None,
            split="train"
        )
        
        # 恢复原始的 disc_factor（如果修改过）
        if original_disc_factor is not None:
            self.taming_loss_fn.disc_factor = original_disc_factor
        
        # 提取关键损失值（用于兼容性）
        total_loss = loss
        recon_error = log_dict.get('train/nll_loss', torch.tensor(0.0, device=images.device))
        pixel_loss = log_dict.get('train/rec_loss', recon_error)
        perceptual_loss = log_dict.get('train/p_loss', torch.tensor(0.0, device=images.device))
        g_loss = log_dict.get('train/g_loss', torch.tensor(0.0, device=images.device))
        gan_loss = log_dict.get('train/disc_loss', torch.tensor(0.0, device=images.device))
        disc_factor = log_dict.get('train/disc_factor', torch.tensor(0.0, device=images.device))
        

        if optimizer_idx==0:

            return {
                'total_loss': total_loss,
                'recon_error': recon_error,
                'pixel_loss': pixel_loss,
                'vq_loss': codebook_loss,
                'perceptual_loss': perceptual_loss,
                'perceptual_rec_loss': pixel_loss,  # taming 模型中已包含感知损失
                'g_loss': g_loss,
                'gan_loss': gan_loss,
                'disc_factor': disc_factor.item() if isinstance(disc_factor, torch.Tensor) else disc_factor,
                'lambda_gan': log_dict.get('train/d_weight', torch.tensor(0.0, device=images.device)),
                'log_dict': log_dict  # 保存完整的日志字典
            }
    
        else :

            return {
                'total_loss': total_loss,
                'gan_loss': gan_loss,
                'disc_factor': disc_factor.item() if isinstance(disc_factor, torch.Tensor) else disc_factor,
                'lambda_gan': log_dict.get('train/d_weight', torch.tensor(0.0, device=images.device)),
                'log_dict': log_dict  # 保存完整的日志字典
            }
    def get_loss_metrics(self, loss_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        从损失字典中提取用于日志记录的指标
        
        Args:
            loss_dict: compute_total_loss 返回的损失字典
        
        Returns:
            包含所有损失值的字典（转换为 Python float）
        """
        metrics = {
            'loss/total': loss_dict['total_loss'].item() if isinstance(loss_dict['total_loss'], torch.Tensor) else loss_dict['total_loss'],
            'loss/recon': loss_dict['recon_error'].item() if isinstance(loss_dict['recon_error'], torch.Tensor) else loss_dict['recon_error'],
            'loss/vq': loss_dict['vq_loss'].item() if isinstance(loss_dict['vq_loss'], torch.Tensor) else loss_dict['vq_loss'],
        }
        
        # 对于 taming 模型，从 log_dict 中提取额外的损失信息
        if self.use_taming_loss and 'log_dict' in loss_dict:
            log_dict = loss_dict['log_dict']
            # 提取 taming 模型的所有损失信息
            for key, value in log_dict.items():
                if isinstance(value, torch.Tensor):
                    # 只记录标量张量
                    if value.numel() == 1:
                        metrics[key] = value.item()
                elif isinstance(value, (int, float)):
                    metrics[key] = float(value)
        
        # 传统模型的损失信息
        if 'perceptual_loss' in loss_dict:
            perceptual_loss = loss_dict['perceptual_loss']
            if isinstance(perceptual_loss, torch.Tensor) and perceptual_loss.numel() > 0:
                metrics['loss/perceptual'] = perceptual_loss.item() if perceptual_loss.numel() == 1 else perceptual_loss.mean().item()
        
        if 'pixel_loss' in loss_dict:
            pixel_loss = loss_dict['pixel_loss']
            if isinstance(pixel_loss, torch.Tensor) and pixel_loss.numel() > 0:
                metrics['loss/pixel_l1'] = pixel_loss.item() if pixel_loss.numel() == 1 else pixel_loss.mean().item()
        
        if self.use_gan or (self.use_taming_loss and self.use_gan):
            if 'g_loss' in loss_dict:
                g_loss = loss_dict['g_loss']
                if isinstance(g_loss, torch.Tensor) and g_loss.numel() > 0:
                    metrics['loss/gan_g'] = g_loss.item() if g_loss.numel() == 1 else g_loss.mean().item()
            
            if 'gan_loss' in loss_dict:
                gan_loss = loss_dict['gan_loss']
                if isinstance(gan_loss, torch.Tensor) and gan_loss.numel() > 0:
                    metrics['loss/gan_d'] = gan_loss.item() if gan_loss.numel() == 1 else gan_loss.mean().item()
        
        return metrics

