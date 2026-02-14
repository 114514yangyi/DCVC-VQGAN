"""
模型适配器模块

提供统一的模型接口，支持多种 VQ-VAE 模型实现。
适配器层负责：
1. 从配置文件读取模型类型和参数
2. 创建对应的具体模型实例
3. 提供统一的接口供训练脚本使用
4. 支持扩展，方便添加新的模型实现
"""
import json
import os
import logging
from typing import Tuple, Optional, Dict, Any
import torch
from torch import nn

logger = logging.getLogger(__name__)


class BaseVqVaeAdapter(nn.Module):
    """
    VQ-VAE 模型适配器基类
    
    定义统一的接口，所有具体的模型适配器都应该继承此类。
    这样可以确保不同的模型实现都能被训练脚本统一使用。
    """
    
    def __init__(self, model: nn.Module):
        """
        初始化适配器
        
        Args:
            model: 具体的模型实例
        """
        super().__init__()
        # 使用 object.__setattr__ 直接设置 _model，避免 nn.Module 的 __setattr__ 拦截
        object.__setattr__(self, '_model', model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch*sequence, channels, height, width)
        
        Returns:
            Tuple[vq_loss, images_recon, perplexity, encoding_indices]:
            - vq_loss: VQ 损失（标量）
            - images_recon: 重建图像，形状与输入相同
            - perplexity: 困惑度（标量）
            - encoding_indices: 码本索引（可选）
        """
        return self._get_model()(x)
    
    @property
    def encoder(self) -> nn.Module:
        """获取编码器模块（兼容 encoder 和 encoders）"""
        model = self._get_model()
        # 如果模型有 encoder 属性（单数），直接返回
        if hasattr(model, 'encoder'):
            return model.encoder
        # 如果模型有 encoders 属性（复数），返回第一个 encoder
        elif hasattr(model, 'encoders') and len(model.encoders) > 0:
            return model.encoders[0]
        else:
            raise AttributeError(f"模型 {type(model).__name__} 没有 encoder 或 encoders 属性")
    
    @property
    def decoder(self) -> nn.Module:
        """获取解码器模块"""
        return self._get_model().decoder
    
    @property
    def output_channels(self) -> int:
        """获取输出通道数"""
        return self._get_model().output_channels
    
    @property
    def vocab_size(self) -> Optional[int]:
        """获取词汇表大小（如果模型支持）"""
        return getattr(self._get_model(), 'vocab_size', None)
    
    @property
    def vq_vae(self) -> Optional[nn.Module]:
        """获取向量量化器（如果模型支持）"""
        return getattr(self._get_model(), 'vq_vae', None)
    
    def parameters(self, recurse: bool = True):
        """获取模型参数"""
        return self._get_model().parameters(recurse=recurse)
    
    def named_parameters(self, prefix='', recurse=True):
        """获取命名的模型参数"""
        return self._get_model().named_parameters(prefix, recurse)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """获取模型状态字典"""
        return self._get_model().state_dict(destination, prefix, keep_vars)
    
    def load_state_dict(self, state_dict, strict: bool = True):
        """加载模型状态字典"""
        return self._get_model().load_state_dict(state_dict, strict)
    
    def train(self, mode: bool = True):
        """设置训练模式"""
        return self._get_model().train(mode)
    
    def eval(self):
        """设置评估模式"""
        return self._get_model().eval()
    
    def _get_model(self):
        """安全地获取内部模型，避免触发 __getattr__"""
        return object.__getattribute__(self, '_model')
    
    def to(self, device):
        """移动模型到指定设备"""
        # 使用 object.__getattribute__ 直接访问 _model，避免触发 __getattr__
        model = object.__getattribute__(self, '_model')
        object.__setattr__(self, '_model', model.to(device))
        return self
    
    def cuda(self, device=None):
        """移动模型到 CUDA 设备"""
        model = object.__getattribute__(self, '_model')
        object.__setattr__(self, '_model', model.cuda(device))
        return self
    
    def cpu(self):
        """移动模型到 CPU"""
        model = object.__getattribute__(self, '_model')
        object.__setattr__(self, '_model', model.cpu())
        return self
    
    def __getattr__(self, name):
        """
        代理所有其他属性访问到内部模型
        
        这样可以确保适配器能够访问模型的所有属性和方法
        """
        # 如果访问的是私有属性（以 _ 开头），直接抛出异常
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # 使用 object.__getattribute__ 安全地访问 _model
        try:
            model = object.__getattribute__(self, '_model')
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '_model'")
        
        return getattr(model, name)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码函数（推理时使用）
        
        Args:
            x: 输入张量
        
        Returns:
            码本索引
        """
        model = self._get_model()
        if hasattr(model, 'encode'):
            return model.encode(x)
        else:
            raise NotImplementedError(f"模型 {type(model)} 不支持 encode 方法")
    
    def decode(self, encode_indices: torch.Tensor) -> torch.Tensor:
        """
        解码函数（推理时使用）
        
        Args:
            encode_indices: 码本索引
        
        Returns:
            重建的图像
        """
        model = self._get_model()
        if hasattr(model, 'decode'):
            return model.decode(encode_indices)
        else:
            raise NotImplementedError(f"模型 {type(model)} 不支持 decode 方法")


class VqVae2Adapter(BaseVqVaeAdapter):
    """
    VqVae2 模型的适配器
    
    适配 models.vq_vae.vq_vae.VqVae2 模型
    """
    
    def __init__(self, model_args: Dict[str, Any], config_path: str = 'param.json'):
        """
        初始化 VqVae2 适配器
        
        Args:
            model_args: 模型参数字典
            config_path: 配置文件路径（用于读取额外配置）
        """
        from models.vq_vae.vq_vae import VqVae2
        
        # 从配置文件读取模型参数（如果存在）
        config = _load_config(config_path)
        model_config = config.get('model_args', {})
        
        # 合并配置，model_args 优先级更高
        merged_args = {**model_config, **model_args}

        merged_args['use_optvq'] = False
        
        # 创建 VqVae2 模型
        model = VqVae2(**merged_args)
        
        super().__init__(model)
        
        logger.info(f"创建 VqVae2 模型适配器")
        logger.info(f"模型参数: {merged_args}")


class VqVae2OptVQAdapter(BaseVqVaeAdapter):
    """
    VqVae OptVQ 模型的适配器
    
    适配 models.vq_vae.vq_vae.VqVae2 模型
    """
    
    def __init__(self, model_args: Dict[str, Any], config_path: str = 'param.json'):
        """
        初始化 VqVae2 适配器
        
        Args:
            model_args: 模型参数字典
            config_path: 配置文件路径（用于读取额外配置）
        """
        from models.vq_vae.vq_vae import VqVae2
        
        # 从配置文件读取模型参数（如果存在）
        config = _load_config(config_path)
        model_config = config.get('model_args', {})
        
        # 合并配置，model_args 优先级更高
        merged_args = {**model_config, **model_args}

        merged_args['use_optvq'] = True
        
        # 创建 VqVae2 模型
        model = VqVae2(**merged_args)
        
        super().__init__(model)
        
        logger.info(f"创建 VqVae2 模型适配器")
        logger.info(f"模型参数: {merged_args}")

class VQVAEAdapter(BaseVqVaeAdapter):
    """
    VQVAE 模型的适配器
    
    适配 models.vqvae2.vqvae.VQVAE 模型
    """
    
    # VQVAE 模型支持的参数列表
    _SUPPORTED_PARAMS = {
        'in_channels', 'hidden_channels', 'res_channels', 'nb_res_layers',
        'nb_levels', 'embed_dim', 'nb_entries', 'scaling_rates'
    }
    
    @staticmethod
    def _convert_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        将参数从 VqVae2 格式转换为 VQVAE 格式
        
        Args:
            params: 原始参数字典
        
        Returns:
            转换后的参数字典
        """
        converted = {}
        
        # 参数映射：从 VqVae2 格式到 VQVAE 格式
        param_mapping = {
            'input_channels': 'in_channels',
            'vocab_size': 'nb_entries',
            "n_init":"embed_dim",
            "n_hid":"hidden_channels",
            "scaling_rates":"scaling_rates",
        }
        
        # 直接映射的参数
        for old_key, new_key in param_mapping.items():
            if old_key in params:
                converted[new_key] = params[old_key]
        
        # 保留 VQVAE 支持的参数
        for key in VQVAEAdapter._SUPPORTED_PARAMS:
            if key in params:
                converted[key] = params[key]
        
        # 设置默认值（如果某些必需参数缺失）
        defaults = {
            'in_channels': 3,
            'hidden_channels': 128,
            'res_channels': 64,
            'nb_res_layers': 2,
            'nb_levels': 2,
            'embed_dim': 64,
            'nb_entries': 512,
            'scaling_rates': [ 4, 2]
        }
        
        # 如果 in_channels 未设置，尝试从 input_channels 获取
        if 'in_channels' not in converted and 'input_channels' in params:
            converted['in_channels'] = params['input_channels']
        
        # 如果 nb_entries 未设置，尝试从 vocab_size 获取
        if 'nb_entries' not in converted and 'vocab_size' in params:
            converted['nb_entries'] = params['vocab_size']
        
        # 应用默认值
        for key, default_value in defaults.items():
            if key not in converted:
                converted[key] = default_value
        
        return converted
    
    def __init__(self, model_args: Dict[str, Any], config_path: str = 'param.json'):
        """
        初始化 VQVAE 适配器
        
        Args:
            model_args: 模型参数字典
            config_path: 配置文件路径（用于读取额外配置）
        """
        from models.vqvae2.vqvae import VQVAE
        
        # 从配置文件读取模型参数（如果存在）
        config = _load_config(config_path)
        model_config = config.get('model_args', {})
        model_config['scaling_rates'] = config.get('scaling_rates', [4, 2])

        # 合并配置，model_args 优先级更高
        merged_args = {**model_config, **model_args}
        
        # 转换参数格式并过滤不支持的参数
        converted_args = self._convert_params(merged_args)
        
        # 创建 VQVAE 模型
        model = VQVAE(**converted_args)
        
        super().__init__(model)
        
        logger.info(f"创建 VQVAE 模型适配器")
        logger.info(f"原始参数: {merged_args}")
        logger.info(f"转换后参数: {converted_args}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播，转换为训练脚本期望的格式
        
        Args:
            x: 输入张量，形状为 (batch*sequence, channels, height, width)
        
        Returns:
            Tuple[vq_loss, images_recon, perplexity, encoding_indices]:
            - vq_loss: VQ 损失（标量）
            - images_recon: 重建图像，形状与输入相同
            - perplexity: 困惑度（标量）
            - encoding_indices: 码本索引（可选）
        """
        model = self._get_model()
        
        # 调用模型的 forward_training 方法（如果存在），否则使用 forward 并转换
        if hasattr(model, 'forward_training'):
            return model.forward_training(x)
        else:
            # 使用原始 forward 并转换格式
            images_recon, diffs, encoder_outputs, decoder_outputs, id_outputs = model.forward(x)
            
            # 计算总 VQ 损失（所有层级的 diff 之和）
            vq_loss = sum(diffs)
            
            # 计算平均困惑度（简化处理）
            # 注意：这里简化了困惑度的计算，实际可能需要更复杂的处理
            perplexity = torch.tensor(1.0, device=x.device)
            
            # 使用最后一层的编码索引
            encoding_indices = id_outputs[-1] if id_outputs else None
            
            return vq_loss, images_recon, perplexity, encoding_indices


class TamingVQGANAdapter(BaseVqVaeAdapter):
    """
    Taming Transformers VQGAN 模型的适配器
    
    适配 models.taming.models.vqgan.VQModel 和 EMAVQ 模型
    将 PyTorch Lightning 的 VQGAN 模型包装成符合用户训练体系的接口
    
    使用示例（在 param.json 中）:
    {
        "model_type": "TamingVQGAN",
        "model_args": {
            "model_variant": "EMAVQ",  // 或 "VQModel"
            "n_embed": 1024,
            "embed_dim": 256,
            "ddconfig": {
                "double_z": false,
                "z_channels": 256,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [1, 1, 2, 2, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [16],
                "dropout": 0.0
            },
            "lossconfig": {
                "target": "models.taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
                "params": {
                    "disc_conditional": false,
                    "disc_in_channels": 3,
                    "disc_start": 10000,
                    "disc_weight": 0.8,
                    "codebook_weight": 1.0
                }
            }
        }
    }
    """
    
    @staticmethod
    def _convert_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        将用户配置参数转换为 taming 模型需要的格式
        
        Args:
            params: 用户提供的参数字典
        
        Returns:
            转换后的参数字典
        """
        converted = params.copy()
        
        # 如果用户提供了简化的参数，转换为 taming 格式
        # 例如：如果提供了 input_channels，转换为 ddconfig.in_channels
        if 'input_channels' in converted and 'ddconfig' not in converted:
            if 'ddconfig' not in converted:
                converted['ddconfig'] = {}
            converted['ddconfig']['in_channels'] = converted.pop('input_channels')
        
        # 如果提供了 vocab_size，转换为 n_embed
        if 'vocab_size' in converted and 'n_embed' not in converted:
            converted['n_embed'] = converted.pop('vocab_size')
        
        # 如果提供了 n_init，转换为 embed_dim
        if 'n_init' in converted and 'embed_dim' not in converted:
            converted['embed_dim'] = converted.pop('n_init')
        
        return converted
    
    def __init__(self, model_args: Dict[str, Any], config_path: str = 'param.json'):
        """
        初始化 Taming VQGAN 适配器
        
        Args:
            model_args: 模型参数字典，需要包含：
                - ddconfig: 编解码器配置（字典）
                - lossconfig: 损失函数配置（可选，如果不提供则使用默认配置）
                - n_embed: 码本大小（例如 1024）
                - embed_dim: 量化特征维度（例如 256）
                - model_variant: 模型变体，可选 'VQModel' 或 'EMAVQ'（默认 'VQModel'）
                - ckpt_path: 可选的预训练 checkpoint 路径
            config_path: 配置文件路径
        """
        # 从配置文件读取模型参数（如果存在）
        config = _load_config(config_path)
        model_config = config.get('model_args', {})
        
        # 合并配置，model_args 优先级更高
        merged_args = {**model_config, **model_args}
        
        # 转换参数格式
        merged_args = self._convert_params(merged_args)
        
        # 导入 taming 模型
        from models.taming.models.vqgan import VQModel, EMAVQ
        
        # 获取模型变体（默认使用 VQModel）
        model_variant = merged_args.pop('model_variant', 'VQModel')
        
        # 提取必需的参数
        ddconfig = merged_args.pop('ddconfig', {})
        lossconfig = merged_args.pop('lossconfig', None)
        n_embed = merged_args.pop('n_embed', 1024)
        embed_dim = merged_args.pop('embed_dim', 256)
        ckpt_path = merged_args.pop('ckpt_path', None)
        
        # 如果没有提供 lossconfig，创建默认配置
        if lossconfig is None:
            # 使用默认的 VQLPIPSWithDiscriminator 配置
            lossconfig = {
                'target': 'models.taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator',
                'params': {
                    'disc_conditional': False,
                    'disc_in_channels': ddconfig.get('in_channels', 3),
                    'disc_start': 10000,
                    'disc_weight': 0.8,
                    'codebook_weight': 1.0,
                }
            }
        
        # 根据模型变体创建模型
        if model_variant == 'EMAVQ':
            model = EMAVQ(
                ddconfig=ddconfig,
                lossconfig=lossconfig,
                n_embed=n_embed,
                embed_dim=embed_dim,
                ckpt_path=ckpt_path,
                ignore_keys=[],
                image_key="image",
                colorize_nlabels=None,
                monitor=None,
                remap=None,
                sane_index_shape=False,
            )
        else:
            model = VQModel(
                ddconfig=ddconfig,
                lossconfig=lossconfig,
                n_embed=n_embed,
                embed_dim=embed_dim,
                ckpt_path=ckpt_path,
                ignore_keys=[],
                image_key="image",
                colorize_nlabels=None,
                monitor=None,
                remap=None,
                sane_index_shape=False,
            )
        
        # 移除 PyTorch Lightning 的自动优化功能
        # 因为用户的训练体系手动管理优化器
        model.automatic_optimization = False
        
        # 设置学习率（taming 模型的 configure_optimizers 需要这个属性）
        # 从配置中读取，如果没有则使用默认值
        learning_rate = merged_args.get('learning_rate', merged_args.get('lr', 1e-4))
        model.learning_rate = learning_rate
        
        super().__init__(model)
        
        logger.info(f"创建 Taming VQGAN 模型适配器 (variant: {model_variant})")
        logger.info(f"码本大小: {n_embed}, 量化维度: {embed_dim}")
        logger.info(f"学习率: {learning_rate}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播，转换为训练脚本期望的格式
        
        taming 的 VQModel.forward() 返回 (dec, diff)，
        需要转换为 (vq_loss, images_recon, perplexity, encoding_indices)
        
        Args:
            x: 输入张量，形状为 (batch*sequence, channels, height, width)
        
        Returns:
            Tuple[vq_loss, images_recon, perplexity, encoding_indices]:
            - vq_loss: VQ 损失（标量）
            - images_recon: 重建图像，形状与输入相同
            - perplexity: 困惑度（标量）
            - encoding_indices: 码本索引（可选）
        """
        model = self._get_model()
        
        # 调用 taming 模型的 encode 方法获取量化信息
        quant, emb_loss, info = model.encode(x)
        
        # 解码
        images_recon = model.decode(quant)
        
        # 从 info 中提取困惑度和编码索引
        # info 格式: (perplexity, encodings, encoding_indices)
        if info is not None and len(info) >= 3:
            perplexity = info[0]  # 困惑度
            encoding_indices = info[2]  # 编码索引
            
            # 如果 perplexity 是 None，计算一个默认值
            if perplexity is None:
                perplexity = torch.tensor(1.0, device=x.device)
            
            # 确保 encoding_indices 的形状正确
            if encoding_indices is not None:
                # 如果是一维的，可能需要 reshape
                if len(encoding_indices.shape) == 1:
                    # 尝试根据量化特征的空间维度 reshape
                    b, c, h, w = quant.shape
                    encoding_indices = encoding_indices.view(b, h, w)
        else:
            # 如果 info 格式不正确，使用默认值
            perplexity = torch.tensor(1.0, device=x.device)
            encoding_indices = None
            raise ValueError("无法从量化器获取编码索引")
        
        # emb_loss 就是 vq_loss
        vq_loss = emb_loss
        
        return vq_loss, images_recon, perplexity, encoding_indices
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码函数（推理时使用）
        
        Args:
            x: 输入张量
        
        Returns:
            码本索引
        """
        model = self._get_model()
        quant, emb_loss, info = model.encode(x)
        
        # 从 info 中提取编码索引
        if info is not None and len(info) >= 3:
            encoding_indices = info[2]
            # print(encoding_indices.shape)
            return encoding_indices
        else:
            raise ValueError("无法从量化器获取编码索引")
    
    def decode(self, encode_indices: torch.Tensor) -> torch.Tensor:
        """
        解码函数（推理时使用）
        
        Args:
            encode_indices: 码本索引
        
        Returns:
            重建的图像
        """
        model = self._get_model()
        return model.decode_code(encode_indices)
    
    @property
    def output_channels(self) -> int:
        """获取输出通道数"""
        model = self._get_model()
        return model.decoder.conv_out.out_channels
    
    @property
    def vocab_size(self) -> Optional[int]:
        """获取词汇表大小（码本大小）"""
        model = self._get_model()
        # 尝试不同的属性名（不同量化器可能使用不同的名称）
        if hasattr(model.quantize, 'n_e'):
            return model.quantize.n_e
        elif hasattr(model.quantize, 'num_tokens'):
            return model.quantize.num_tokens
        elif hasattr(model.quantize, 'n_embed'):
            return model.quantize.n_embed
        elif hasattr(model.quantize, 're_embed'):
            return model.quantize.re_embed
        else:
            return None


class VqvaeOptVQAdapter(BaseVqVaeAdapter):
    """
    VQVAE OptVQ 模型的适配器
    
    适配 models.vqvae2_optvq.vqvae.VQVAE 模型（使用 OptVQ 量化器）
    """
    
    # VQVAE OptVQ 模型支持的参数列表
    _SUPPORTED_PARAMS = {
        'in_channels', 'hidden_channels', 'res_channels', 'nb_res_layers',
        'nb_levels', 'embed_dim', 'nb_entries', 'scaling_rates',
        'epsilon', 'n_iters', 'normalize_mode', 'use_prob',
        'beta', 'loss_q_type', 'use_norm', 'use_proj'
    }
    
    @staticmethod
    def _convert_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        将参数从 VqVae2 格式转换为 VQVAE OptVQ 格式
        
        Args:
            params: 原始参数字典
        
        Returns:
            转换后的参数字典
        """
        converted = {}
        
        # 参数映射：从 VqVae2 格式到 VQVAE OptVQ 格式
        param_mapping = {
            'input_channels': 'in_channels',
            'vocab_size': 'nb_entries',
            "n_init":"embed_dim",
            "n_hid":"hidden_channels",
            "scaling_rates":"scaling_rates",
        }
        
        # 直接映射的参数
        for old_key, new_key in param_mapping.items():
            if old_key in params:
                converted[new_key] = params[old_key]
        
        # 保留 VQVAE OptVQ 支持的参数
        for key in VqvaeOptVQAdapter._SUPPORTED_PARAMS:
            if key in params:
                converted[key] = params[key]
        
        # 设置默认值（如果某些必需参数缺失）
        defaults = {
            'in_channels': 3,
            'hidden_channels': 128,
            'res_channels': 32,
            'nb_res_layers': 2,
            'nb_levels': 3,
            'embed_dim': 64,
            'nb_entries': 512,
            'scaling_rates': [8, 4, 2],
            # OptVQ 相关参数的默认值
            'epsilon': 10.0,
            'n_iters': 5,
            'normalize_mode': 'all',
            'use_prob': True,
            'beta': 1.0,
            'loss_q_type': 'ce',
            'use_norm': False,
            'use_proj': True
        }
        
        # 如果 in_channels 未设置，尝试从 input_channels 获取
        if 'in_channels' not in converted and 'input_channels' in params:
            converted['in_channels'] = params['input_channels']
        
        # 如果 nb_entries 未设置，尝试从 vocab_size 获取
        if 'nb_entries' not in converted and 'vocab_size' in params:
            converted['nb_entries'] = params['vocab_size']
        
        # 应用默认值
        for key, default_value in defaults.items():
            if key not in converted:
                converted[key] = default_value
        
        return converted
    
    def __init__(self, model_args: Dict[str, Any], config_path: str = 'param.json'):
        """
        初始化 VQVAE OptVQ 适配器
        
        Args:
            model_args: 模型参数字典
            config_path: 配置文件路径（用于读取额外配置）
        """
        from models.vqvae2_optvq.vqvae import VQVAE
        
        # 从配置文件读取模型参数（如果存在）
        config = _load_config(config_path)
        model_config = config.get('model_args', {})
        model_config['scaling_rates'] = config.get('scaling_rates', [4, 2])
        
        # 合并配置，model_args 优先级更高
        merged_args = {**model_config, **model_args}
        
        # 转换参数格式并过滤不支持的参数
        converted_args = self._convert_params(merged_args)
        
        # 创建 VQVAE OptVQ 模型
        model = VQVAE(**converted_args)
        
        super().__init__(model)
        
        logger.info(f"创建 VQVAE OptVQ 模型适配器")
        logger.info(f"原始参数: {merged_args}")
        logger.info(f"转换后参数: {converted_args}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播，转换为训练脚本期望的格式
        
        Args:
            x: 输入张量，形状为 (batch*sequence, channels, height, width)
        
        Returns:
            Tuple[vq_loss, images_recon, perplexity, encoding_indices]:
            - vq_loss: VQ 损失（标量）
            - images_recon: 重建图像，形状与输入相同
            - perplexity: 困惑度（标量）
            - encoding_indices: 码本索引（可选）
        """
        model = self._get_model()
        
        # 调用模型的 forward_training 方法（如果存在），否则使用 forward 并转换
        if hasattr(model, 'forward_training'):
            return model.forward_training(x)
        else:
            # 使用原始 forward 并转换格式
            images_recon, diffs, encoder_outputs, decoder_outputs, id_outputs = model.forward(x)
            
            # 计算总 VQ 损失（所有层级的 diff 之和）
            vq_loss = sum(diffs)
            
            # 计算平均困惑度（简化处理）
            # 注意：这里简化了困惑度的计算，实际可能需要更复杂的处理
            perplexity = torch.tensor(1.0, device=x.device)
            
            # 使用最后一层的编码索引
            encoding_indices = id_outputs[-1] if id_outputs else None
            
            return vq_loss, images_recon, perplexity, encoding_indices


def _load_config(config_path: str = 'param.json') -> Dict[str, Any]:
    """
    加载配置文件（内部函数）
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def create_model(
    model_args: Optional[Dict[str, Any]] = None,
    config_path: str = 'param.json'
) -> BaseVqVaeAdapter:
    """
    根据配置文件创建模型适配器
    
    这是工厂函数，负责：
    1. 从配置文件读取模型类型
    2. 根据模型类型创建对应的适配器
    3. 支持扩展，可以轻松添加新的模型类型
    
    Args:
        model_args: 模型参数字典（可选，会与配置文件中的参数合并）
        config_path: 配置文件路径，默认为 'param.json'
    
    Returns:
        BaseVqVaeAdapter: 模型适配器实例
    
    Raises:
        ValueError: 如果模型类型不支持
        FileNotFoundError: 如果配置文件不存在
    """
    # 加载配置文件
    config = _load_config(config_path)
    
    # 从配置文件读取模型类型（如果指定）
    model_type = config.get('model_type', 'VqVae2')  # 默认使用 VqVae2
    
    # 获取模型参数
    config_model_args = config.get('model_args', {})
    if model_args is None:
        model_args = config_model_args.copy()
    else:
        # 合并参数，model_args 优先级更高
        model_args = {**config_model_args, **model_args}
    
    logger.info(f"创建模型: {model_type}")
    logger.info(f"模型参数来源: 配置文件 + 传入参数")
    
    # 使用注册表查找适配器类
    adapter_class = _MODEL_REGISTRY.get(model_type)
    if adapter_class is None:
        available = ', '.join(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"不支持的模型类型: {model_type}。"
            f"支持的模型类型: {available}"
        )
    
    # 创建适配器实例
    return adapter_class(model_args, config_path)


# 模型注册表（用于扩展）
_MODEL_REGISTRY: Dict[str, type] = {
    'VqVae': VqVae2Adapter,
    'vq_vae': VqVae2Adapter,
    'vqvae': VqVae2Adapter,
    'VQVAE2': VQVAEAdapter,
    'vqvae2': VQVAEAdapter,
    'VQVAE2_OptVQ': VqvaeOptVQAdapter,
    'vqvae2_optvq': VqvaeOptVQAdapter,
    'Vqvae2OptVQ': VqvaeOptVQAdapter,
    'vqvae2optvq': VqvaeOptVQAdapter,
    'vqvaeoptvq': VqVae2OptVQAdapter,
    # Taming Transformers VQGAN 模型
    'TamingVQGAN': TamingVQGANAdapter,
    'taming_vqgan': TamingVQGANAdapter,
    'TamingVQModel': TamingVQGANAdapter,
    'taming_vqmodel': TamingVQGANAdapter,
    'EMAVQ': TamingVQGANAdapter,
    'emavq': TamingVQGANAdapter,
    'VQGAN': TamingVQGANAdapter,
    'vqgan': TamingVQGANAdapter,
}


def register_model(model_type: str, adapter_class: type):
    """
    注册新的模型适配器（用于扩展）
    
    Args:
        model_type: 模型类型名称
        adapter_class: 适配器类（必须继承自 BaseVqVaeAdapter）
    
    Example:
        >>> from models.model_adapter import register_model, BaseVqVaeAdapter
        >>> class MyModelAdapter(BaseVqVaeAdapter):
        ...     pass
        >>> register_model('MyModel', MyModelAdapter)
    """
    if not issubclass(adapter_class, BaseVqVaeAdapter):
        raise TypeError(f"适配器类必须继承自 BaseVqVaeAdapter")
    
    _MODEL_REGISTRY[model_type] = adapter_class
    logger.info(f"注册新模型类型: {model_type}")


def get_available_models() -> list:
    """
    获取所有可用的模型类型
    
    Returns:
        模型类型列表
    """
    return list(_MODEL_REGISTRY.keys())

