"""
VQGAN 模型核心实现（基于 taming-transformers）

文件结构概览
-------------
- VQModel          : 经典 VQGAN，含编码器/解码器、向量量化器、判别器损失等
- VQSegmentationModel : 语义分割场景的轻量变体（无判别器）
- VQNoDiscModel    : 去掉判别器的 VQ 模型
- GumbelVQ         : 使用 GumbelSoftmax 量化的变体
- EMAVQ            : 使用 EMA 码本更新的变体

主要流程
-------------
1) 编码：Encoder -> quant_conv -> VectorQuantizer (或 Gumbel/EMA 量化)
2) 解码：post_quant_conv -> Decoder
3) 训练：autoencoder 损失 + 可选判别器损失（lossconfig 控制）

常见配置字段
-------------
- ddconfig: 编解码器结构（通道、分辨率、残差块等）
- lossconfig: 损失与判别器配置（在 main 中 instantiate_from_config）
- n_embed: 码本大小
- embed_dim: 量化特征维度

注意事项
-------------
- quant_conv / post_quant_conv 用 1x1 卷积在编码特征维度与量化维度之间转换
- forward 返回 (重建, 量化损失)，训练步骤分开处理 AE 与判别器
- sane_index_shape 可让量化器返回 bhw 形状的索引（兼容下游需求）
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import importlib

# 优先从本地模块导入，如果失败则从安装的 taming 包导入
try:
    from models.taming.modules.diffusionmodules.model import Encoder, Decoder
    from models.taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
    from models.taming.modules.vqvae.quantize import GumbelQuantize
    from models.taming.modules.vqvae.quantize import EMAVectorQuantizer, ResidualEMAVectorQuantizer
except ImportError:
    # 如果本地导入失败，尝试从安装的 taming 包导入
    try:
        from taming.modules.diffusionmodules.model import Encoder, Decoder
        from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
        from taming.modules.vqvae.quantize import GumbelQuantize
        from taming.modules.vqvae.quantize import EMAVectorQuantizer, ResidualEMAVectorQuantizer
    except ImportError:
        # 如果 VectorQuantizer2 不存在，尝试使用 VectorQuantizer
        from taming.modules.diffusionmodules.model import Encoder, Decoder
        from taming.modules.vqvae.quantize import VectorQuantizer
        from taming.modules.vqvae.quantize import GumbelQuantize
        from taming.modules.vqvae.quantize import EMAVectorQuantizer, ResidualEMAVectorQuantizer


def get_obj_from_str(string, reload=False):
    """
    从字符串路径动态导入并获取类或函数对象
    
    Args:
        string: 类的完整路径，格式为 "module.path.ClassName"
        reload: 是否重新加载模块
    
    Returns:
        cls: 类对象（未实例化）
    """
    module, cls = string.rsplit(".", 1)
    
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    根据配置字典动态实例化对象
    
    Args:
        config: 配置字典，必须包含 "target" 键，可选包含 "params" 键
    
    Returns:
        instance: 实例化的对象
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class VQModel(pl.LightningModule):
    """
    VQGAN 核心模型类
    
    这是标准的 VQGAN 实现，包含：
    - 编码器（Encoder）：将图像编码为潜在特征
    - 向量量化器（VectorQuantizer）：将连续特征离散化为码本索引
    - 解码器（Decoder）：从量化特征重建图像
    - 判别器损失（通过 lossconfig 配置）：用于对抗训练提升重建质量
    
    训练流程：
    1. 编码器将输入图像编码为潜在特征 z
    2. quant_conv 将 z 投影到量化器所需的维度
    3. 向量量化器将连续特征量化为离散码本向量
    4. post_quant_conv 将量化后的特征投影回解码器输入维度
    5. 解码器从量化特征重建图像
    6. 计算重建损失 + 量化损失 + 判别器损失（可选）
    """
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=True,  # tell vector quantizer to return indices as bhw
                 ):
        """
        初始化 VQModel
        
        Args:
            ddconfig: 编解码器配置字典，包含通道数、分辨率等结构参数
            lossconfig: 损失函数配置，用于实例化损失计算器（包含判别器）
            n_embed: 码本大小，即离散向量的数量（例如 8192）
            embed_dim: 量化特征的维度，即每个码本向量的维度（例如 256）
            ckpt_path: 可选的预训练 checkpoint 路径，用于加载权重
            ignore_keys: 加载 checkpoint 时要忽略的键前缀列表（用于迁移学习）
            image_key: batch 字典中图像数据的键名，默认为 "image"
            colorize_nlabels: 用于语义分割的可视化，将标签映射到 RGB 颜色
            monitor: 监控的指标名称，用于模型检查点保存（例如 "val/rec_loss"）
            remap: 码本重映射配置（用于迁移学习或码本压缩）
            sane_index_shape: 如果为 True，量化器返回 (batch, height, width) 形状的索引
                             如果为 False，返回展平的索引（兼容不同下游任务）
        """
        super().__init__()
        # 保存图像键名，用于从 batch 中提取数据
        self.image_key = image_key
        
        # 初始化编码器和解码器，使用相同的配置（对称结构）
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        # 从配置实例化损失计算器（包含判别器、感知损失等）
        self.loss = instantiate_from_config(lossconfig)
        
        # 初始化向量量化器
        # beta=0.25 是 commitment loss 的权重，控制编码器输出与码本向量的对齐强度
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=True)
        
        # quant_conv: 1x1 卷积，将编码器输出的 z_channels 维特征投影到 embed_dim 维
        # 这是量化前的维度转换
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        
        # post_quant_conv: 1x1 卷积，将量化后的 embed_dim 维特征投影回 z_channels 维
        # 这是量化后的维度恢复，用于输入解码器
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        # 如果提供了 checkpoint 路径，加载预训练权重
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        # 重复设置 image_key（可能是冗余代码）
        self.image_key = image_key
        
        # 如果指定了 colorize_nlabels，初始化颜色映射矩阵（用于语义分割可视化）
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            # 创建随机颜色映射：将 n_labels 个类别映射到 RGB 3 通道
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        
        # 设置监控指标（用于模型检查点保存策略）
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        """
        从 checkpoint 文件加载模型权重
        
        这个方法支持部分加载，可以忽略某些键（例如判别器权重），
        常用于迁移学习或冻结部分模块的场景。
        
        Args:
            path: checkpoint 文件路径
            ignore_keys: 要忽略的键前缀列表，例如 ["loss.discriminator"] 会忽略所有判别器权重
        
        流程：
        1. 加载 checkpoint 文件中的 state_dict
        2. 遍历所有键，删除匹配 ignore_keys 前缀的键
        3. 使用 strict=False 加载，允许部分匹配（缺失的键会被忽略）
        """
        # 加载 checkpoint，使用 CPU 避免显存问题
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        
        # 删除需要忽略的键
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        
        # 使用 strict=False 加载，允许部分匹配
        # 这意味着如果某些键在当前模型中不存在，会被忽略
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        """
        编码函数：将输入图像编码为量化后的潜在特征
        
        流程：
        1. 编码器将图像编码为潜在特征 z (形状: [B, z_channels, H', W'])
        2. quant_conv 将 z 投影到量化器维度 (形状: [B, embed_dim, H', W'])
        3. 向量量化器将连续特征量化为离散码本向量
        
        Args:
            x: 输入图像张量，形状为 [B, C, H, W]
        
        Returns:
            quant: 量化后的特征，形状为 [B, embed_dim, H', W']
                  这些特征是从码本中查找的最接近的向量
            emb_loss: 量化损失（commitment loss + codebook loss）
            info: 量化器的额外信息（可能包含索引、困惑度等）
        """
        # 步骤1: 编码器将图像编码为潜在特征
        # 输出形状: [B, z_channels, H', W']，其中 H', W' 通常比 H, W 小（下采样）
        h = self.encoder(x)
        
        # 步骤2: 1x1 卷积将编码器输出维度转换为量化器输入维度
        # z_channels -> embed_dim
        h = self.quant_conv(h)
        
        # 步骤3: 向量量化
        # quant: 量化后的特征（码本向量的查找结果）
        # emb_loss: 量化损失 = commitment_loss + codebook_loss
        # info: 额外信息（可能包含索引、困惑度等）
        quant, emb_loss, info = self.quantize(h)

        # print(info[2].shape)
        # print(info[2])
        # print(info)
        
        return quant, emb_loss, info

    def decode(self, quant):
        """
        解码函数：从量化后的潜在特征重建图像
        
        流程：
        1. post_quant_conv 将量化特征从 embed_dim 维投影回 z_channels 维
        2. 解码器将潜在特征解码为重建图像
        
        Args:
            quant: 量化后的特征，形状为 [B, embed_dim, H', W']
        
        Returns:
            dec: 重建的图像，形状为 [B, C, H, W]（与原始输入图像尺寸相同）
        """
        # 步骤1: 1x1 卷积将量化特征维度恢复为解码器输入维度
        # embed_dim -> z_channels
        quant = self.post_quant_conv(quant)
        
        # 步骤2: 解码器将潜在特征解码为图像
        # 输出形状: [B, C, H, W]，与原始输入图像尺寸相同
        dec = self.decoder(quant)
        
        return dec

    def decode_code(self, code_b):
        """
        从码本索引直接解码图像
        
        这个方法允许从离散的码本索引（而不是连续特征）重建图像，
        常用于生成任务或压缩场景。
        
        Args:
            code_b: 码本索引，形状为 [B, H', W'] 或 [B*H'*W']（整数张量）
        
        Returns:
            dec: 重建的图像，形状为 [B, C, H, W]
        
        流程：
        1. 使用量化器的 embed_code 方法将索引转换为对应的码本向量
        2. 调用 decode 方法将码本向量解码为图像
        """
        # 步骤1: 将码本索引转换为对应的码本向量
        # code_b (索引) -> quant_b (码本向量)，形状: [B, embed_dim, H', W']
        quant_b = self.quantize.embed_code(code_b)
        
        # 步骤2: 解码为图像
        dec = self.decode(quant_b)
        
        return dec

    def forward(self, input):
        """
        前向传播：完整的编码-解码流程
        
        这是模型的主要前向传播函数，执行完整的重建流程。
        注意：这个方法不包含判别器损失的计算，判别器损失在 training_step 中单独处理。
        
        Args:
            input: 输入图像，形状为 [B, C, H, W]
        
        Returns:
            dec: 重建的图像，形状为 [B, C, H, W]
            diff: 量化损失（commitment loss + codebook loss），标量
        """
        # 步骤1: 编码并量化
        # quant: 量化后的特征
        # diff: 量化损失
        # _: 量化器的额外信息（这里不使用）
        quant, diff, _ = self.encode(input)
        
        # 步骤2: 解码为重建图像
        dec = self.decode(quant)
        
        return dec, diff

    def get_input(self, batch, k):
        """
        从 batch 中提取并预处理输入数据
        
        这个方法处理不同格式的输入数据，确保它们转换为模型期望的格式：
        - 形状: [B, H, W, C] -> [B, C, H, W]
        - 数据类型: 转换为 float32
        - 内存格式: 确保连续内存布局（提高性能）
        
        Args:
            batch: 数据批次字典
            k: 数据键名（通常是 "image"）
        
        Returns:
            x: 预处理后的图像张量，形状为 [B, C, H, W]，数据类型为 float32
        """
        # 从 batch 中提取数据
        x = batch[k]
        
        # 如果输入是 3 维的 [H, W, C]，添加 batch 维度
        if len(x.shape) == 3:
            x = x[..., None]  # [H, W, C] -> [H, W, C, 1]，后续会处理
        
        # 将数据从 [B, H, W, C] 转换为 [B, C, H, W]（PyTorch 标准格式）
        x = x.permute(0, 3, 1, 2)
        
        # 确保内存连续布局（提高后续操作效率）
        x = x.to(memory_format=torch.contiguous_format)
        
        # 转换为 float32 类型
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        训练步骤：处理一个批次的训练数据
        
        VQGAN 使用两个优化器（双优化器策略）：
        - optimizer_idx=0: 优化自编码器（编码器、解码器、量化器）
        - optimizer_idx=1: 优化判别器
        
        这种交替训练策略可以稳定 GAN 训练过程。
        
        Args:
            batch: 训练数据批次
            batch_idx: 批次索引
            optimizer_idx: 优化器索引（0=自编码器，1=判别器）
        
        Returns:
            loss: 当前优化器的损失值
        """
        # 步骤1: 提取并预处理输入图像
        x = self.get_input(batch, self.image_key)
        
        # 步骤2: 前向传播，获得重建图像和量化损失
        # xrec: 重建图像
        # qloss: 量化损失（commitment loss + codebook loss）
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # ========== 自编码器优化分支 ==========
            # 计算自编码器总损失：重建损失 + 量化损失 + 感知损失（如果有）+ 生成器对抗损失
            
            # loss 函数会根据 optimizer_idx 计算不同的损失：
            # - optimizer_idx=0: 返回自编码器损失（重建损失 + 量化损失 + 生成器对抗损失）
            # - optimizer_idx=1: 返回判别器损失
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            # 记录主要损失到进度条和日志
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            
            # 记录详细的损失字典（不显示在进度条）
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            
            return aeloss

        if optimizer_idx == 1:
            # ========== 判别器优化分支 ==========
            # 计算判别器损失：区分真实图像和重建图像
            
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            
            # 记录判别器损失
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            
            return discloss

    def validation_step(self, batch, batch_idx):
        """
        验证步骤：在验证集上评估模型性能
        
        验证时同时计算自编码器和判别器的损失，但不进行反向传播。
        主要用于监控模型性能和选择最佳 checkpoint。
        
        Args:
            batch: 验证数据批次
            batch_idx: 批次索引
        
        Returns:
            log_dict: 包含所有验证指标的字典
        """
        # 步骤1: 提取并预处理输入图像
        x = self.get_input(batch, self.image_key)
        
        # 步骤2: 前向传播（不计算梯度）
        xrec, qloss = self(x)
        
        # 步骤3: 计算自编码器损失（用于监控重建质量）
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        # 步骤4: 计算判别器损失（用于监控判别器性能）
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        
        # 步骤5: 提取重建损失（通常是最重要的指标）
        rec_loss = log_dict_ae["val/rec_loss"]
        
        # 步骤6: 记录关键指标到进度条和日志
        # sync_dist=True 确保在分布式训练时同步所有进程的指标
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        # 步骤7: 记录所有详细指标
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        
        return self.log_dict

    def configure_optimizers(self):
        """
        配置优化器：设置自编码器和判别器的优化器
        
        VQGAN 使用双优化器策略：
        1. 自编码器优化器：优化编码器、解码器、量化器和投影层
        2. 判别器优化器：优化判别器网络
        
        返回两个优化器列表，PyTorch Lightning 会交替调用它们。
        
        Returns:
            optimizers: 优化器列表 [opt_ae, opt_disc]
            schedulers: 学习率调度器列表（这里为空列表，表示不使用调度器）
        """
        lr = self.learning_rate
        
        # ========== 自编码器优化器 ==========
        # 优化所有自编码器相关参数：
        # - encoder: 编码器参数
        # - decoder: 解码器参数
        # - quantize: 量化器参数（码本向量）
        # - quant_conv: 量化前投影层
        # - post_quant_conv: 量化后投影层
        # betas=(0.5, 0.9) 是 GAN 训练常用的 Adam 参数
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        
        # ========== 判别器优化器 ==========
        # 只优化判别器参数（判别器在 loss 对象中）
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        
        # 返回两个优化器，PyTorch Lightning 会根据 optimizer_idx 选择使用哪个
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        """
        获取解码器的最后一层权重
        
        这个方法用于感知损失计算。感知损失（如 LPIPS）通常只对解码器的最后一层
        应用较小的权重，因为最后一层主要影响高频细节，而感知损失更关注整体结构。
        
        Returns:
            weight: 解码器最后一层卷积的权重张量
        """
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        """
        记录图像用于可视化（在训练/验证过程中调用）
        
        这个方法生成输入图像和重建图像的对比，用于监控训练进度。
        如果输入是多通道的（如语义分割），会使用颜色映射转换为 RGB。
        
        Args:
            batch: 数据批次
            **kwargs: 其他参数（可能包含 split="train" 或 "val"）
        
        Returns:
            log: 包含 "inputs" 和 "reconstructions" 的字典，用于日志记录
        """
        log = dict()
        
        # 步骤1: 提取并预处理输入图像
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        
        # 步骤2: 前向传播获得重建图像
        xrec, _ = self(x)
        
        # 步骤3: 如果输入是多通道的（如语义分割），转换为 RGB 可视化
        if x.shape[1] > 3:
            # 确保重建图像也是多通道的
            assert xrec.shape[1] > 3
            # 使用随机颜色映射将多通道数据转换为 RGB
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        
        # 步骤4: 保存到日志字典
        log["inputs"] = x
        log["reconstructions"] = xrec
        
        return log

    def to_rgb(self, x):
        """
        将多通道数据（如语义分割标签）转换为 RGB 图像用于可视化
        
        这个方法使用一个随机初始化的颜色映射矩阵，将多通道数据投影到 RGB 空间。
        主要用于语义分割任务的可视化。
        
        Args:
            x: 多通道输入，形状为 [B, num_channels, H, W]
        
        Returns:
            x: RGB 图像，形状为 [B, 3, H, W]，值域归一化到 [-1, 1]
        
        流程：
        1. 使用 1x1 卷积（颜色映射矩阵）将多通道数据投影到 3 通道
        2. 将值域归一化到 [-1, 1]（PyTorch 图像的标准范围）
        """
        # 确保这个方法只在语义分割任务中使用
        assert self.image_key == "segmentation"
        
        # 如果还没有颜色映射矩阵，创建一个随机初始化的
        # 形状: [3, num_channels, 1, 1]，用于 1x1 卷积
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        
        # 步骤1: 使用 1x1 卷积进行颜色映射
        # 将 num_channels 维投影到 3 维（RGB）
        x = F.conv2d(x, weight=self.colorize)
        
        # 步骤2: 归一化到 [-1, 1] 范围
        # 这是 PyTorch 图像的标准范围（与 tanh 激活函数输出一致）
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        
        return x


class VQSegmentationModel(VQModel):
    """
    VQGAN 的语义分割变体
    
    这个类专门用于语义分割任务，主要特点：
    - 没有判别器（简化训练，减少计算量）
    - 包含颜色映射功能，用于将分割标签可视化
    - 优化器只优化自编码器部分
    
    适用于需要高质量特征表示但不追求极致视觉质量的场景。
    """
    def __init__(self, n_labels, *args, **kwargs):
        """
        初始化语义分割模型
        
        Args:
            n_labels: 语义分割的类别数量
            *args, **kwargs: 传递给父类 VQModel 的其他参数
        """
        # 调用父类初始化
        super().__init__(*args, **kwargs)
        
        # 初始化颜色映射矩阵：将 n_labels 个类别映射到 RGB 3 通道
        # 形状: [3, n_labels, 1, 1]，用于 1x1 卷积
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        # 仅优化自编码器部分（无判别器）
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        """
        训练步骤（简化版，无判别器）
        
        与父类不同，这里不需要 optimizer_idx，因为只有一个优化器。
        只计算自编码器损失（重建损失 + 量化损失）。
        
        Args:
            batch: 训练数据批次
            batch_idx: 批次索引
        
        Returns:
            aeloss: 自编码器损失
        """
        # 步骤1: 提取输入数据
        x = self.get_input(batch, self.image_key)
        
        # 步骤2: 前向传播
        xrec, qloss = self(x)
        
        # 步骤3: 计算自编码器损失（不传入 optimizer_idx，因为无判别器）
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        
        # 步骤4: 记录损失
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        """
        记录图像用于可视化（语义分割专用版本）
        
        对于语义分割任务，需要特殊处理：
        1. 如果输出是 logits，先转换为类别索引
        2. 将类别索引转换为 one-hot 编码
        3. 使用颜色映射转换为 RGB 可视化
        
        Args:
            batch: 数据批次
            **kwargs: 其他参数
        
        Returns:
            log: 包含输入和重建图像的字典
        """
        log = dict()
        
        # 步骤1: 提取输入数据
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        
        # 步骤2: 前向传播（不计算梯度）
        xrec, _ = self(x)
        
        # 步骤3: 处理多通道输出（语义分割）
        if x.shape[1] > 3:
            # 确保重建输出也是多通道的
            assert xrec.shape[1] > 3
            
            # 步骤3a: 将 logits 转换为类别索引
            # 如果 xrec 是 logits [B, num_classes, H, W]，取 argmax 得到类别索引
            xrec = torch.argmax(xrec, dim=1, keepdim=True)  # [B, 1, H, W]
            
            # 步骤3b: 将类别索引转换为 one-hot 编码
            # 形状: [B, 1, H, W] -> [B, num_classes, H, W]
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            
            # 步骤3c: 调整维度顺序
            # [B, num_classes, H, W, 1] -> [B, num_classes, H, W]
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            
            # 步骤3d: 使用颜色映射转换为 RGB
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        
        # 步骤4: 保存到日志
        log["inputs"] = x
        log["reconstructions"] = xrec
        
        return log


class VQNoDiscModel(VQModel):
    """
    VQGAN 的无判别器版本
    
    这个类移除了判别器，只使用重建损失和量化损失进行训练。
    适用于：
    - 不需要对抗训练的简单重建任务
    - 计算资源受限的场景
    - 更稳定的训练过程（避免 GAN 训练的不稳定性）
    
    注意：这个类使用旧版本的 PyTorch Lightning API（TrainResult/EvalResult），
    可能与新版本不兼容。
    """
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        """
        初始化无判别器模型
        
        Args:
            ddconfig: 编解码器配置
            lossconfig: 损失函数配置（不包含判别器）
            n_embed: 码本大小
            embed_dim: 量化特征维度
            ckpt_path: 可选的 checkpoint 路径
            ignore_keys: 加载 checkpoint 时忽略的键
            image_key: 图像数据的键名
            colorize_nlabels: 颜色映射标签数（可选）
        """
        # 调用父类初始化（不包含判别器）
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        # 无判别器的纯 VQAE 优化器
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


class GumbelVQ(VQModel):
    """
    VQGAN 的 Gumbel-Softmax 量化变体
    
    这个类使用 Gumbel-Softmax 进行可微分的离散量化，而不是硬量化。
    主要特点：
    - 使用 Gumbel-Softmax 实现可微分的离散采样
    - 温度退火策略：训练初期温度高（更平滑），后期温度低（更离散）
    - 包含 KL 散度损失，鼓励码本使用均匀分布
    
    优势：
    - 可微分，训练更稳定
    - 可以学习更灵活的码本分布
    - 适合需要端到端训练的生成任务
    """
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):
        """
        初始化 GumbelVQ 模型
        
        Args:
            ddconfig: 编解码器配置
            lossconfig: 损失函数配置
            n_embed: 码本大小
            embed_dim: 量化特征维度
            temperature_scheduler_config: 温度调度器配置（控制 Gumbel-Softmax 的温度）
            ckpt_path: 可选的 checkpoint 路径
            ignore_keys: 加载 checkpoint 时忽略的键
            image_key: 图像数据的键名
            colorize_nlabels: 颜色映射标签数
            monitor: 监控指标名称
            kl_weight: KL 散度损失的权重（鼓励码本均匀使用）
            remap: 码本重映射配置
        """
        # 获取编码器输出的通道数
        z_channels = ddconfig["z_channels"]
        
        # 调用父类初始化（注意：ckpt_path=None，因为后面会单独处理）
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,  # 先不加载，后面单独处理
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        # 设置损失函数的类别数（用于某些损失计算）
        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        # 初始化 Gumbel 量化器（替换父类的标准量化器）
        # kl_weight: KL 散度权重，鼓励码本均匀使用
        # temp_init: 初始温度（通常为 1.0，训练过程中会逐渐降低）
        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        # 初始化温度调度器（用于温度退火）
        # 训练过程中，温度会从高到低逐渐降低，使量化从平滑变为离散
        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)

        # 如果提供了 checkpoint，现在加载（在替换量化器之后）
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        """
        更新 Gumbel-Softmax 的温度
        
        这个方法在每次训练步骤开始时调用，根据当前训练步数更新温度。
        温度退火策略：
        - 训练初期：温度高 -> Gumbel-Softmax 更平滑 -> 更容易探索
        - 训练后期：温度低 -> Gumbel-Softmax 更离散 -> 更接近硬量化
        
        这是 Gumbel-Softmax 训练的关键步骤。
        """
        # 根据当前训练步数计算新的温度值
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        # 返回量化前特征，方便自定义量化策略或可视化
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        训练步骤（GumbelVQ 版本）
        
        与标准 VQModel 的主要区别：
        1. 每次训练前更新温度（温度退火）
        2. 记录当前温度值用于监控
        
        Args:
            batch: 训练数据批次
            batch_idx: 批次索引
            optimizer_idx: 优化器索引（0=自编码器，1=判别器）
        
        Returns:
            loss: 当前优化器的损失值
        """
        # 步骤1: 更新温度（温度退火策略）
        # 必须在每次训练步骤开始时调用，确保使用正确的温度
        self.temperature_scheduling()
        
        # 步骤2: 提取输入数据
        x = self.get_input(batch, self.image_key)
        
        # 步骤3: 前向传播（使用更新后的温度）
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # ========== 自编码器优化分支 ==========
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            # 记录损失
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            
            # 记录当前温度（重要：用于监控温度退火过程）
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            
            return aeloss

        if optimizer_idx == 1:
            # ========== 判别器优化分支 ==========
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


class EMAVQ(VQModel):
    """
    VQGAN 的 EMA（指数移动平均）码本更新变体
    
    这个类使用 EMA 方式更新码本，而不是通过梯度下降。
    主要特点：
    - 码本向量通过 EMA 更新，不需要梯度
    - 训练更稳定，码本更新更平滑
    - 优化器不包含量化器参数（因为码本通过 EMA 更新）
    
    优势：
    - 训练更稳定
    - 码本更新更平滑，避免突然变化
    - 适合大规模训练
    """
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 num_quantizers: int = 1,
                 use_residual_rq: bool = True,
                 ):
        """
        初始化 EMAVQ 模型
        
        Args:
            ddconfig: 编解码器配置
            lossconfig: 损失函数配置
            n_embed: 码本大小
            embed_dim: 量化特征维度
            ckpt_path: 可选的 checkpoint 路径
            ignore_keys: 加载 checkpoint 时忽略的键
            image_key: 图像数据的键名
            colorize_nlabels: 颜色映射标签数
            monitor: 监控指标名称
            remap: 码本重映射配置
            sane_index_shape: 是否返回 bhw 形状的索引
        """
        # 调用父类初始化（注意：ckpt_path=None，因为量化器会被替换）
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,  # 先不加载
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        # 如果 num_quantizers > 1，则使用 RQ-VAE 风格的多层残差量化；
        # 否则退化为单层 EMA 量化，与原实现保持一致。
        if use_residual_rq and num_quantizers > 1:
            self.quantize = ResidualEMAVectorQuantizer(
                n_embed=n_embed,
                embedding_dim=embed_dim,
                beta=0.25,
                num_quantizers=num_quantizers,
                remap=remap,
            )
        else:
            # beta=0.25 是 commitment loss 的权重
            # EMA 量化器会在前向传播时自动更新码本（通过 EMA）
            self.quantize = EMAVectorQuantizer(
                n_embed=n_embed,
                embedding_dim=embed_dim,
                beta=0.25,
                remap=remap,
            )
    def configure_optimizers(self):
        """
        配置优化器（EMAVQ 版本）
        
        关键区别：量化器参数不在优化器列表中，因为码本通过 EMA 更新。
        EMA 更新在前向传播时自动进行，不需要梯度。
        
        Returns:
            optimizers: 优化器列表 [opt_ae, opt_disc]
            schedulers: 学习率调度器列表（空）
        """
        lr = self.learning_rate
        
        # ========== 自编码器优化器 ==========
        # 注意：不包含 self.quantize.parameters()
        # 因为 EMA 量化器的码本通过指数移动平均更新，不需要梯度优化
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        
        # ========== 判别器优化器 ==========
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        
        return [opt_ae, opt_disc], []                                           