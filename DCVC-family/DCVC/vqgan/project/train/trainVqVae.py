import os.path
import time
import math
from datetime import timedelta
from typing import  Dict, Any


import attr
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from train.train_utils import save_checkpoint, AverageMeter, ProgressMeter
from log_utils.log_utils import LogManager
from metric_utils.metric_utils import MetricsEvaluator
from train.lossconfig import LossCalculator, get_train_args_from_config

# =============================================================================
# trainVqVae.py
# -----------------------------------------------------------------------------
# 本文件封装了 "VQ-VAE" 及其相关 GAN 变体模型的视频训练主流程。
#
# 主要功能包含：
#   1. 训练主控类 TrainVqVae 的定义 —— 管理模型训练、验证、日志记录、断点续训、结果可视化等工作；
#   2. 实用工具及辅助模块的导入，如日志管理、指标评估、判别器损失等；
#   3. 支持灵活的训练参数、可视化频率、wandb集成、模型存档与恢复等。
#
# -----------------------------------------------------------------------------
# 文件结构详细说明
# -----------------------------------------------------------------------------
# 1. 主要依赖及工具导入
#    - 通用包：os、time、datetime、typing；
#    - 第三方库：numpy、torch 及相关深度学习/功能扩展库（einops、attr等）；
#    - 项目内自定义模块，如：训练工具、日志、判别器、LPIPS感知损失、GAN损失辅助等。
#
# 2. TrainVqVae 训练主类（核心）
#    - 属性参数以 attr 类方式声明，类型清晰，并支持灵活扩展；
#    - 步骤覆盖：数据加载 -> 前后标准化 -> 优化器/调度器配置 -> 主循环 -> 验证/记录 -> checkpoint存储；
#    - 丰富的日志信息方便定位与追踪训练过程问题；
#    - 兼容 wandb 远程监控 & 本地日志管理；
#    - 支持自动恢复(断点续训)、周期性保存、良好异常处理机制。
#
# 3. 相关工具说明
#    - 日志管理(LogManager)：统一控制tensorboard/wandb/本地日志等；
#    - save_checkpoint/AverageMeter/ProgressMeter：训练常用的实用函数，提升代码整洁与健壮性；
#    - MetricEvaluator/BadCaseDetector: 评估各项指标和检测训练异常样本；
#    - GAN/LPIPS模块: 支持GAN判别器训练与感知损失引入。
#
# -----------------------------------------------------------------------------
# 使用建议
# -----------------------------------------------------------------------------
# - 外部通过主脚本 (如 train_custom_videos.py) 实例化 TrainVqVae, 调用其 train() 方法发起全部训练流程；
# - 可通过自定义 logger/log_mgr、metrics 等参数灵活接入不同的训练/日志环境；
# - 强烈建议配合集成的配置管理（如 yaml/json/params）一同使用；
#
# -----------------------------------------------------------------------------
# 总结
# -----------------------------------------------------------------------------
# 本文件旨在提供高可扩展性、高内聚、便于调试的视频 VQ-VAE/GAN 训练解决方案。
# 所有核心方法与配置均围绕研究/生产需求进行精细设计，保障易用性和鲁棒性。
# =============================================================================



@attr.s(eq=False, repr=False)
class TrainVqVae:
    """
    VQ-VAE 训练类
    
    负责模型的训练过程，包括：
    - 数据加载和预处理
    - 前向传播和反向传播
    - 损失计算和优化
    - 模型保存和可视化
    - 详细的日志记录
    """
    # ========== 必填参数 ==========
    model: nn.Module = attr.ib()
    normalize: nn.Module = attr.ib()
    unnormalize: nn.Module = attr.ib()
    training_loader: torch.utils.data.DataLoader = attr.ib()
    log_mgr: LogManager = attr.ib()
    
    # 以下参数将从 param.json 配置文件中读取，不需要手动传递
    config_path: str = attr.ib(default='param.json')
    # 配置文件路径，默认为 'param.json'

    validation_loader: torch.utils.data.DataLoader = attr.ib(default=None)
    # （可选）验证集数据加载器，如果提供则定期进行验证

    logger: Any = attr.ib(default=None)
    # Python logger对象，由LogManager自动注入。若为None则自动获取log_mgr.logger

    metrics: MetricsEvaluator = attr.ib(default=None)
    # 指标评估器（如PSNR、感知损失等），默认自动创建。可自定义注入

    # 以下参数如果不在配置文件中，将使用默认值
    n_images_save: int = attr.ib(default=16)
    # 每次保存时采样的图像/视频序列数，默认16

    wandb_log_interval: int = attr.ib(default=50)
    # 每 wandb_log_interval 步向wandb记录一次标量（如loss，指标数据）

    wandb_image_interval: int = attr.ib(default=200)
    # 每 wandb_image_interval 步向wandb上传一次重建结果图片/视频

    wandb_max_images: int = attr.ib(default=8)
    # 每次wandb可视化最多上传多少帧（节省资源，避免占用空间）

    default_gpu: str = attr.ib(default='cuda:0')
    # 默认GPU



    def __attrs_post_init__(self):
        """
        初始化训练环境
        - 从配置文件读取参数
        - 创建必要的目录
        - 加载checkpoint（如果提供）
        - 初始化logger（如果未提供）
        """
        # 从配置文件读取参数
        train_args = get_train_args_from_config(self.config_path)
        
        # 从配置文件读取必填参数
        self.num_steps = train_args.get('num_steps', 100000)  # 默认100000步
        self.lr = train_args.get('lr', 0.001)
        self.lr_decay = train_args.get('lr_decay', 0.99)
        self.folder_name = train_args.get('folder_name', './checkpoints')+f"_{self.config_path.split('/')[-1].split('.')[0]}"
        self.checkpoint_path = train_args.get('checkpoint_path', None)
        self.save_interval = train_args.get('save_interval', 10000)  # 默认每10000步保存一次
        self.validation_interval = train_args.get('validation_interval', 10000)  # 默认每10000步验证一次        
        # 创建输出目录
        self.path_img_orig = os.path.join(self.folder_name, 'images_orig')+f"cuda_{self.default_gpu.split(':')[-1]}"
        self.path_img_recs = os.path.join(self.folder_name, 'images_recs')+f"cuda_{self.default_gpu.split(':')[-1]}"
        os.makedirs(self.folder_name, exist_ok=True)
        os.makedirs(self.path_img_orig, exist_ok=True)
        os.makedirs(self.path_img_recs, exist_ok=True)
        
        # 使用外部注入的 logger
        if self.logger is None:
            self.logger = self.log_mgr.logger

        # 初始化损失计算器（包含判别器和感知损失函数的初始化）
        # LossCalculator 会从 param.json 配置文件中读取相关参数
        device = next(self.model.parameters()).device
        self.loss_calculator = LossCalculator(
            decoder=self.model.decoder if hasattr(self.model, 'decoder') else None,
            num_steps=self.num_steps,  # 传入总步数
            output_channels=self.model.output_channels,
            device=device,
            base_lr=self.lr,
            config_path=self.config_path,
            model=self.model  # 传递模型以检测是否是 taming 模型
        )
        # 将 logger 传递给 loss_calculator（如果它支持）
        if hasattr(self.loss_calculator, 'logger'):
            self.loss_calculator.logger = self.logger
        
        # 检测是否是 taming 模型（使用 taming 损失函数）
        self.use_taming_loss = self.loss_calculator.use_taming_loss
        
        # 根据模型类型初始化优化器
        if self.use_taming_loss:
            # taming 模型使用双优化器（自编码器和判别器）
            opt_ae, opt_disc = self.loss_calculator.get_taming_optimizers()
            self.optimizer = opt_ae  # 自编码器优化器
            self.optimizer_disc = opt_disc  # 判别器优化器
            self.logger.info("使用 Taming VQGAN 双优化器训练模式")
        else:
            # 传统模型使用单优化器
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=False)
            self.optimizer_disc = None
            self.logger.info("使用传统单优化器训练模式")
        
        # 学习率调度器（只用于自编码器优化器）
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        
        # 默认从 0 开始
        self.start_step = 0  # 从 checkpoint 恢复的起始步数

        # 指标评估器
        if self.metrics is None:
            vocab_size = getattr(self.model, "vocab_size", None)
            device = next(self.model.parameters()).device
            self.metrics = MetricsEvaluator(vocab_size=vocab_size, device=device)
        
        # Best model 跟踪（PSNR, FID, LPIPS）
        self.best_psnr = float('-inf')
        self.best_fid = float('inf')
        self.best_lpips = float('inf')
        self.best_psnr_step = -1
        self.best_fid_step = -1
        self.best_lpips_step = -1
        
        # 加载checkpoint
        if self.checkpoint_path is not None:
            self.logger.info(f"正在从checkpoint加载模型: {self.checkpoint_path}")
            # 使用模型所在的设备加载checkpoint
            device = next(self.model.parameters()).device
            checkpoint = torch.load(self.checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            # 如果是 taming 模型，加载判别器优化器
            if self.use_taming_loss and self.optimizer_disc is not None:
                if 'optimizer_disc' in checkpoint:
                    self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
                    self.logger.info("已加载判别器优化器状态")
            
            # 通过 LossCalculator 加载判别器和优化器状态（传统模型）
            if not self.use_taming_loss:
                self.loss_calculator.load_checkpoint(checkpoint)
            
            # 从 checkpoint 恢复 step
            if 'steps' in checkpoint:
                self.start_step = checkpoint['steps']
                self.logger.info(f"成功加载checkpoint，从 step {self.start_step} 继续训练")
            else:
                self.start_step = 0
                self.logger.info("checkpoint 中没有 steps 信息，从头开始训练")
        else:
            self.logger.info("从头开始训练")
        
        # 训练时间统计
        self.step_times = []  # 记录每个step的时间
        self.start_time = None
        

    def _log_wandb_metrics(self, step: int, metrics: Dict[str, Any]):
        self.log_mgr.log_metrics(step, metrics)

    def _log_wandb_images(self, step: int, images: torch.Tensor, images_recon: torch.Tensor, b: int, d: int):
        self.log_mgr.log_images(
            step=step,
            images=images,
            images_recon=images_recon,
            b=b,
            d=d,
            n_images_save=self.n_images_save,
            wandb_max_images=self.wandb_max_images,
            unnormalize=self.unnormalize,
            save_dir_orig=self.path_img_orig,
            save_dir_recs=self.path_img_recs
        )
    
    def _check_and_save_best_models(self, global_step: int, val_metrics: Dict[str, Any]):
        """
        检查并保存 best model（基于 PSNR, FID, LPIPS）
        
        Args:
            global_step: 当前全局步数
            val_metrics: 验证指标字典
        """
        import math
        
        # 获取当前指标值（尝试多个可能的键）
        current_psnr = val_metrics.get('val/stats/psnr') or val_metrics.get('val/metrics/psnr', float('-inf'))
        current_fid = val_metrics.get('val/stats/fid') or val_metrics.get('val/metrics/fid', float('inf'))
        current_lpips = val_metrics.get('val/stats/lpips') or val_metrics.get('val/metrics/lpips', float('inf'))
        
        # 如果值无效，使用默认值
        if current_psnr is None or math.isnan(current_psnr):
            current_psnr = float('-inf')
        if current_fid is None or math.isnan(current_fid):
            current_fid = float('inf')
        if current_lpips is None or math.isnan(current_lpips):
            current_lpips = float('inf')
        
        # 检查 PSNR（越大越好）
        if not math.isnan(current_psnr) and current_psnr > self.best_psnr:
            self.best_psnr = current_psnr
            self.best_psnr_step = global_step
            self._save_best_model('PSNR', global_step, current_psnr, current_fid, current_lpips)
            self.logger.info(f"✓ 新的最佳 PSNR: {current_psnr:.4f} dB (Step {global_step})")
        
        # 检查 FID（越小越好）
        if not math.isnan(current_fid) and current_fid < self.best_fid:
            self.best_fid = current_fid
            self.best_fid_step = global_step
            self._save_best_model('FID', global_step, current_psnr, current_fid, current_lpips)
            self.logger.info(f"✓ 新的最佳 FID: {current_fid:.4f} (Step {global_step})")
        
        # 检查 LPIPS（越小越好）
        if not math.isnan(current_lpips) and current_lpips < self.best_lpips:
            self.best_lpips = current_lpips
            self.best_lpips_step = global_step
            self._save_best_model('LPIPS', global_step, current_psnr, current_fid, current_lpips)
            self.logger.info(f"✓ 新的最佳 LPIPS: {current_lpips:.4f} (Step {global_step})")
        
        # 记录到 wandb
        best_metrics = {
            'best/psnr': self.best_psnr,
            'best/fid': self.best_fid,
            'best/lpips': self.best_lpips,
            'best/psnr_step': self.best_psnr_step,
            'best/fid_step': self.best_fid_step,
            'best/lpips_step': self.best_lpips_step,
        }
        self._log_wandb_metrics(global_step, best_metrics)
    
    def _save_best_model(self, metric_name: str, global_step: int, 
                        psnr: float, fid: float, lpips: float):
        """
        保存 best model
        
        Args:
            metric_name: 指标名称 ('PSNR', 'FID', 'LPIPS')
            global_step: 当前全局步数
            psnr: 当前 PSNR 值
            fid: 当前 FID 值
            lpips: 当前 LPIPS 值
        """
        import math
        
        try:
            # 格式化指标值用于文件名
            psnr_str = f"{psnr:.2f}" if not math.isnan(psnr) else "nan"
            fid_str = f"{fid:.2f}" if not math.isnan(fid) else "nan"
            lpips_str = f"{lpips:.4f}" if not math.isnan(lpips) else "nan"
            
            # 构建文件名：bestmodel_PSNR_FID_IPIPS_best_PSNR
            # 例如：bestmodel_PSNR_28.50_FID_45.23_IPIPS_0.1234_best_PSNR
            # 注意：IPIPS 在文件名中使用 IPIPS，但在代码中使用 LPIPS
            filename = f"bestmodel_PSNR_{psnr_str}_FID_{fid_str}_IPIPS_{lpips_str}_best_{metric_name}_step{global_step}.pth.tar"
            filepath = os.path.join(self.folder_name, filename)
            
            # 准备 checkpoint 数据
            checkpoint_data = {
                'steps': global_step,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_metric': metric_name,
                'metrics': {
                    'psnr': psnr,
                    'fid': fid,
                    'lpips': lpips,
                }
            }
            
            # 如果是 taming 模型，保存判别器优化器状态
            if self.use_taming_loss and self.optimizer_disc is not None:
                checkpoint_data['optimizer_disc'] = self.optimizer_disc.state_dict()
            
            # 通过 LossCalculator 获取判别器和优化器状态（传统模型）
            if not self.use_taming_loss:
                checkpoint_data.update(self.loss_calculator.get_checkpoint_state())
            
            # 保存 checkpoint
            save_checkpoint(self.folder_name, checkpoint_data, filename)
            self.logger.info(f"Best {metric_name} model 已保存: {filepath}")
        except Exception as e:
            self.logger.error(f"保存最佳模型 ({metric_name}) 时出错: {e}", exc_info=True)
            raise  # 重新抛出异常，让调用者处理

    @torch.no_grad()
    def evaluate(self, max_images: int = 10000) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            max_images: 最多处理的图片数量，默认1000张
        """
        if self.validation_loader is None:
            return {}
        self.model.eval()
        metrics_accum = []
        loss_accum = []
        recon_accum = []
        vq_accum = []
        perp_accum = []

        # 重置 FID 指标（开始新的评估周期）
        self.metrics.reset_fid()
        
        # 跟踪已处理的图片数量
        total_images_processed = 0
        max_images_to_process = max_images
        
        if self.logger:
            self.logger.info(f"开始验证，最多处理 {max_images_to_process} 张图片...")
        
        for batch_idx, data_batch in enumerate(self.validation_loader):
            # 检查是否已经达到最大图片数量
            if total_images_processed >= max_images_to_process:
                if self.logger:
                    self.logger.info(f"已达到最大图片数量限制 ({max_images_to_process} 张)，停止验证")
                break
            # 处理可能包含路径的数据
            if isinstance(data_batch, tuple) and len(data_batch) == 2:
                images, _ = data_batch
            else:
                images = data_batch
            images = images.float().to(next(self.model.parameters()).device)
            b, d, h, w, c = images.size()
            
            # 计算当前批次包含的图片数量（batch_size * sequence_length）
            images_in_batch = b * d
            
            # 如果当前批次会超过限制，只处理部分图片
            remaining_images = max_images_to_process - total_images_processed
            if images_in_batch > remaining_images:
                # 只处理前 remaining_images 张图片
                # 计算需要保留的批次和序列
                if remaining_images >= d:
                    # 至少可以保留一个完整的序列
                    b_to_keep = remaining_images // d
                    d_to_keep = d
                    images = images[:b_to_keep, :, :, :, :]
                    b = b_to_keep
                else:
                    # 只能保留部分序列
                    b_to_keep = 1
                    d_to_keep = remaining_images
                    images = images[:1, :d_to_keep, :, :, :]
                    b = b_to_keep
                    d = d_to_keep
                
                if self.logger:
                    self.logger.info(f"批次 {batch_idx}: 只处理 {remaining_images} 张图片（共 {images_in_batch} 张）")
            
            images = rearrange(images, 'b d h w c -> (b d) c h w')
            images = self.normalize(images.float() / 255.)

            model_out = self.model(images)
            if isinstance(model_out, (list, tuple)) and len(model_out) == 4:
                vq_loss, images_recon, perplexity, encoding_indices = model_out
            else:
                vq_loss, images_recon, perplexity = model_out
                encoding_indices = None

            recon_error = F.mse_loss(images_recon, images)
            loss = recon_error + vq_loss

            # 计算所有指标（包括 FID 和 LPIPS）
            metric_vals = self.metrics.compute_all(
                orig=images,
                recon=images_recon,
                recon_mse=recon_error,
                perplexity=perplexity,
                encoding_indices=encoding_indices,
                b=b,
                d=d,
                compute_fid_lpips=True  # 启用 FID 和 LPIPS 计算
            )
            metrics_accum.append(metric_vals)
            loss_accum.append(loss.item())
            recon_accum.append(recon_error.item())
            vq_accum.append(vq_loss.item())
            perp_accum.append(perplexity.item())
            
            # 更新已处理的图片数量
            total_images_processed += b * d
            
            # 每100个批次输出一次进度
            if self.logger and (batch_idx % 100 == 0 or total_images_processed >= max_images_to_process):
                self.logger.info(f"验证进度: 已处理 {total_images_processed}/{max_images_to_process} 张图片 ({100 * total_images_processed / max_images_to_process:.1f}%)")

        # 汇总
        def _mean(lst):
            return sum(lst) / max(len(lst), 1)

        aggregated = {
            "val/loss/total": _mean(loss_accum),
            "val/loss/recon": _mean(recon_accum),
            "val/loss/vq": _mean(vq_accum),
            "val/stats/perplexity": _mean(perp_accum),
        }
        # 对字典类指标求均值
        if metrics_accum:
            keys = metrics_accum[0].keys()
            for k in keys:
                vals = [m[k] for m in metrics_accum if m.get(k) is not None and not math.isnan(m.get(k))]
                if vals:
                    aggregated[f"val/stats/{k}"] = _mean(vals)
        
        # 计算最终的 FID 值
        final_fid = self.metrics.compute_fid_final()
        if not math.isnan(final_fid):
            aggregated["val/stats/fid"] = final_fid
        
        # 确保所有指标都记录到 wandb（使用统一的命名格式）
        # 重命名指标以便在 wandb 中更清晰地显示
        if 'val/stats/psnr' in aggregated:
            aggregated['val/metrics/psnr'] = aggregated['val/stats/psnr']
        
        # 确保 FID 和 LPIPS 被记录到 wandb（使用 metrics 命名空间以便在 wandb 中更清晰地显示）
        if 'val/stats/fid' in aggregated:
            aggregated['val/metrics/fid'] = aggregated['val/stats/fid']
        else:
            # 如果没有 FID 值，记录 NaN（以便在 wandb 中显示）
            aggregated['val/metrics/fid'] = float('nan')
        
        if 'val/stats/lpips' in aggregated:
            aggregated['val/metrics/lpips'] = aggregated['val/stats/lpips']
        else:
            # 如果没有 LPIPS 值，记录 NaN（以便在 wandb 中显示）
            aggregated['val/metrics/lpips'] = float('nan')
        
        # 记录日志以便调试
        if self.logger:
            fid_val = aggregated.get('val/metrics/fid', 'N/A')
            lpips_val = aggregated.get('val/metrics/lpips', 'N/A')
            self.logger.info(f"验证完成，共处理 {total_images_processed} 张图片")
            self.logger.info(f"验证指标汇总: FID={fid_val}, LPIPS={lpips_val}")
        
        self.model.train()
        return aggregated

    def train(self):
        """
        主训练循环
        
        执行以下步骤：
        1. 数据加载和预处理
        2. 前向传播（Encoder -> VQ -> Decoder）
        3. 损失计算（重建损失 + VQ损失）
        4. 反向传播和优化
        5. 记录详细日志
        6. 定期保存checkpoint和可视化结果
        """
        self.model.train()
        self.start_time = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info("开始训练")
        self.logger.info("=" * 80)
        self.logger.info(f"训练总步数: {self.num_steps}")
        self.logger.info(f"起始步数: {self.start_step}")
        self.logger.info(f"剩余步数: {self.num_steps - self.start_step}")
        self.logger.info(f"保存间隔: {self.save_interval} 步")
        self.logger.info(f"验证间隔: {self.validation_interval} 步")
        self.logger.info(f"学习率: {self.lr}")
        self.logger.info(f"学习率衰减: {self.lr_decay}")
        self.logger.info("")

        # 初始化统计指标
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        meter_loss = AverageMeter('Loss', ':.4e')
        meter_loss_constr = AverageMeter('Constr', ':6.2f')
        meter_loss_perp = AverageMeter('Perplexity', ':6.2f')
        
        # 全局步数计数器
        global_step = self.start_step
        
        # 创建数据加载器的迭代器（用于无限循环）
        data_iter = iter(self.training_loader)

        # 训练循环：按 step 进行
        while global_step < self.num_steps:
            try:
                # 获取下一个批次（如果数据加载器耗尽，重新创建）
                data_batch = next(data_iter)
            except StopIteration:
                # 数据加载器耗尽，重新创建
                data_iter = iter(self.training_loader)
                data_batch = next(data_iter)
            
            step_iter_start = time.time()
            
            # ========== 数据加载阶段 ==========
            data_load_start = time.time()
            # DataLoader 返回 (tensor, paths) 元组
            if isinstance(data_batch, tuple) and len(data_batch) == 2:
                images, video_paths = data_batch
            else:
                images = data_batch
                video_paths = None
            
            data_load_time = time.time() - data_load_start
            data_time.update(data_load_time)

            # ========== 数据预处理阶段 ==========
            preprocess_start = time.time()
            # DataLoader 通过 collate_fn 返回 tensor，格式为 (batch, sequence_length, height, width, channels)
            images = images.float()
            
            # 移动到模型所在的设备
            device = next(self.model.parameters()).device
            images = images.to(device)
            b, d, h, w, c = images.size()
            
            # 重新排列维度: (batch, sequence, height, width, channels) -> (batch*sequence, channels, height, width)
            images = rearrange(images, 'b d h w c -> (b d) c h w')
            # 归一化: 将像素值从 [0, 255] 缩放到 [0, 1]，然后应用ImageNet标准化
            images = self.normalize(images.float() / 255.)
            # 注意: VQ-VAE2 不需要将序列维度合并到通道维度
            preprocess_time = time.time() - preprocess_start
            
            # ========== 前向传播阶段 ==========
            forward_start = time.time()
            
            # 模型前向传播: Encoder -> VQ -> Decoder
            model_out = self.model(images)
            vq_loss, images_recon, perplexity, encoding_indices = model_out
           
            # ========== 使用 LossCalculator 计算所有损失 ==========
            # 对于 taming 模型，需要分别计算自编码器和判别器的损失
            if self.use_taming_loss:
                    # taming 模型使用双优化器训练
                    # 先训练自编码器（optimizer_idx=0）
                    self.optimizer.zero_grad()
                    loss_dict_ae = self.loss_calculator.compute_total_loss(
                        images=images,
                        images_recon=images_recon,
                        vq_loss=vq_loss,
                        step=global_step,
                        logger=self.logger,
                        optimizer_idx=0
                    )
                    total_loss_ae = loss_dict_ae['total_loss']
                    total_loss_ae.backward()
                    self.optimizer.step()
                    
                    # 再训练判别器（optimizer_idx=1）
                    # 只有在 use_gan=True 时才训练判别器
                    if (self.use_taming_loss and 
                        self.loss_calculator.use_gan and 
                        self.optimizer_disc is not None):  # 每3步训练一次判别器
                        self.optimizer_disc.zero_grad()
                        loss_dict_disc = self.loss_calculator.compute_total_loss(
                            images=images,
                            images_recon=images_recon,
                            vq_loss=vq_loss,
                            step=global_step,
                            logger=self.logger,
                            optimizer_idx=1
                        )
                        total_loss_disc = loss_dict_disc['total_loss']
                        total_loss_disc.backward()
                        self.optimizer_disc.step()
                        
                        # 使用判别器损失作为 gan_loss（用于日志）
                        gan_loss = loss_dict_disc['gan_loss']
                    else:
                        gan_loss = torch.tensor(0.0, device=images.device)
                        loss_dict_disc = {}
                    
                    # 合并损失字典（用于日志）
                    loss_dict = {**loss_dict_ae, **loss_dict_disc}
                    total_loss = total_loss_ae  # 用于统计
                    loss_dict["total_loss"]=total_loss
            else:
                # 传统模型使用单优化器训练
                self.optimizer.zero_grad()
                loss_dict = self.loss_calculator.compute_total_loss(
                    images=images,
                    images_recon=images_recon,
                    vq_loss=vq_loss,
                    step=global_step,
                    logger=self.logger,
                    optimizer_idx=0
                )
                total_loss = loss_dict['total_loss']
                total_loss.backward()
                self.optimizer.step()
                # 通过 LossCalculator 执行判别器优化步骤
                self.loss_calculator.step_discriminator(loss_dict['gan_loss'], global_step)
                gan_loss = loss_dict['gan_loss']
            


            # 提取损失值（用于日志）
            recon_error = loss_dict.get('recon_error', torch.tensor(0.0, device=images.device))
            pixel_loss = loss_dict.get('pixel_loss', recon_error)
            perceptual_loss = loss_dict.get('perceptual_loss', torch.tensor(0.0, device=images.device))
            g_loss = loss_dict.get('g_loss', torch.tensor(0.0, device=images.device))
            disc_factor = loss_dict.get('disc_factor', 0.0)

            forward_time = time.time() - forward_start
            current_lr = self.scheduler.get_last_lr()[0]

         

            # ========== 更新统计指标 ==========
            meter_loss_constr.update(recon_error.item(), 1)
            meter_loss_perp.update(perplexity.item(), 1)
            meter_loss.update(total_loss.item(), 1)
            
            # ========== 时间统计 ==========
            step_total_time = time.time() - step_iter_start
            batch_time.update(step_total_time)
            self.step_times.append(step_total_time)
            
            # 只保留最近100个step的时间用于预测
            if len(self.step_times) > 100:
                self.step_times.pop(0)
            
            step_end_time = time.time()

            # ========== 详细日志记录（每个batch） ==========
            self.logger.debug(
                f"Step {global_step:6d}/{self.num_steps-1} | "
                f"Loss: {total_loss.item():.6f} (Recon: {recon_error.item():.6f}, VQ: {vq_loss.item():.6f}) | "
                f"Perplexity: {perplexity.item():.4f} | "
                f"Data: {data_load_time:.3f}s | "
                f"Preprocess: {preprocess_time:.3f}s | "
                f"Forward: {forward_time:.3f}s | "
                f"Total: {step_total_time:.3f}s | "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
            )

            # ========== 计算并记录指标 ==========
            metric_vals = self.metrics.compute_all(
                orig=images,
                recon=images_recon,
                recon_mse=recon_error,
                perplexity=perplexity,
                encoding_indices=encoding_indices,
                b=b,
                d=d
            )
            metrics_to_log = {
                'iter': global_step,
                'stats/perplexity': metric_vals.get('perplexity'),
                'lr': current_lr,
                'stats/psnr': metric_vals.get('psnr'),
                'stats/temporal_consistency': metric_vals.get('temporal_consistency'),
                'stats/codebook_usage': metric_vals.get('codebook_usage'),
                'stats/codebook_entropy': metric_vals.get('codebook_entropy'),
                'stats/bitrate': metric_vals.get('bitrate'),
                'stats/compression_ratio': metric_vals.get('compression_ratio'),
                'stats/codebook_usage_global': metric_vals.get('codebook_usage_global'),
                'stats/codebook_entropy_global': metric_vals.get('codebook_entropy_global'),
            }
            # 使用 LossCalculator 的 get_loss_metrics 方法获取损失指标
            metrics_to_log.update(self.loss_calculator.get_loss_metrics(loss_dict))

            if self.log_mgr.run_wandb is not None and global_step % self.wandb_log_interval == 0:
                self._log_wandb_metrics(step=global_step, metrics=metrics_to_log)

            # ========== 定期显示进度 ==========
            if global_step % 100 == 0:
                # 预测剩余训练时间
                if len(self.step_times) >= 10:
                    avg_step_time = np.mean(self.step_times[-50:])  # 使用最近50个step的平均时间
                    remaining_steps = self.num_steps - global_step - 1
                    estimated_time = timedelta(seconds=int(avg_step_time * remaining_steps))
                    elapsed_time = timedelta(seconds=int(time.time() - self.start_time))
                    self.logger.info(
                        f"进度: Step {global_step}/{self.num_steps - 1} ({100*global_step/self.num_steps:.1f}%) | "
                        f"已用时间: {elapsed_time} | "
                        f"预计剩余时间: {estimated_time} | "
                        f"平均每步: {avg_step_time:.3f}s | "
                        f"Loss: {meter_loss.avg:.6f} | "
                        f"Perplexity: {meter_loss_perp.avg:.4f}"
                    )

            # ========== 记录图片到 wandb ==========
            if global_step % self.wandb_image_interval == 0:
                if self.log_mgr.run_wandb is not None:
                    self.logger.info(f"Step {global_step}: 记录图片到 wandb...")
                    self._log_wandb_images(step=global_step, images=images, images_recon=images_recon, b=b, d=d)
                else:
                    self.logger.debug(f"Step {global_step}: wandb 未初始化，跳过图片记录")

            # ========== 更新学习率（按步数） ==========
            if global_step % 1000 == 0 and global_step > 0:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.logger.info(f"学习率更新为: {current_lr:.2e}")

            # ========== 定期保存checkpoint和可视化（按步数） ==========
            if global_step % self.save_interval == 0 and global_step > 0:
                try:
                    self.logger.info(f"Step {global_step}: 保存checkpoint...")
                    checkpoint_start_time = time.time()
                    
                    # 保存checkpoint
                    checkpoint_path = os.path.join(self.folder_name, f'checkpoint_step{global_step}.pth.tar')
                    checkpoint_data = {
                        'steps': global_step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                    }

                    # 如果是 taming 模型，保存判别器优化器状态
                    if self.use_taming_loss and self.optimizer_disc is not None:
                        checkpoint_data['optimizer_disc'] = self.optimizer_disc.state_dict()

                    # 通过 LossCalculator 获取判别器和优化器状态（传统模型）
                    if not self.use_taming_loss:
                        checkpoint_data.update(self.loss_calculator.get_checkpoint_state())

                    save_checkpoint(self.folder_name, checkpoint_data, f'checkpoint_step{global_step}.pth.tar')
                    checkpoint_time = time.time() - checkpoint_start_time
                    self.logger.info(f"Checkpoint已保存: {checkpoint_path} (耗时: {checkpoint_time:.2f}秒)")

                    # 输出当前统计信息
                    self.logger.info(
                        f"Step {global_step} 统计: "
                        f"Loss={meter_loss.avg:.6f}, "
                        f"Recon={meter_loss_constr.avg:.6f}, "
                        f"VQ={meter_loss.avg - meter_loss_constr.avg:.6f}, "
                        f"Perplexity={meter_loss_perp.avg:.4f}"
                    )
                except Exception as e:
                    self.logger.error(f"保存checkpoint时出错: {e}", exc_info=True)
                    self.logger.warning("继续训练，不中断...")

            # ========== 验证（按步数） ==========
            if global_step % self.validation_interval == 0 and global_step > 0:
                    # 验证并记录wandb指标
                    if self.validation_loader is not None:
                        try:
                            self.logger.info("=" * 80)
                            self.logger.info(f"开始验证 (Step {global_step})...")
                            self.logger.info("=" * 80)
                            
                            val_metrics = self.evaluate()
                            
                            if val_metrics:
                                self._log_wandb_metrics(step=global_step, metrics=val_metrics)
                                
                                # 检查并保存 best model（基于 PSNR, FID, LPIPS）
                                try:
                                    self._check_and_save_best_models(global_step, val_metrics)
                                except Exception as e:
                                    self.logger.error(f"保存最佳模型时出错: {e}", exc_info=True)
                                    self.logger.warning("继续训练，不中断...")
                            
                            self.logger.info("=" * 80)
                            self.logger.info("验证完成，继续训练...")
                            self.logger.info("=" * 80)
                        except Exception as e:
                            self.logger.error(f"验证过程中出错: {e}", exc_info=True)
                            self.logger.warning("验证失败，但继续训练...")
                            # 确保模型恢复为训练模式
                            self.model.train()

            # 更新全局步数
            global_step += 1

        # ========== 训练完成 ==========
        self.logger.info("=" * 80)
        self.logger.info("训练完成，保存最终checkpoint...")
        final_checkpoint_path = os.path.join(self.folder_name, f'checkpoint_step{global_step-1}_final.pth.tar')
        checkpoint_data = {
            'steps': global_step - 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        
        # 如果是 taming 模型，保存判别器优化器状态
        if self.use_taming_loss and self.optimizer_disc is not None:
            checkpoint_data['optimizer_disc'] = self.optimizer_disc.state_dict()
        
        # 通过 LossCalculator 获取判别器和优化器状态（传统模型）
        if not self.use_taming_loss:
            checkpoint_data.update(self.loss_calculator.get_checkpoint_state())
        
        save_checkpoint(self.folder_name, checkpoint_data, f'checkpoint_step{global_step-1}_final.pth.tar')
        
        total_time = timedelta(seconds=int(time.time() - self.start_time))
        self.logger.info(f"最终checkpoint已保存: {final_checkpoint_path}")
        self.logger.info(f"总训练时间: {total_time}")
        self.logger.info(f"总训练步数: {global_step - 1}")
        if len(self.step_times) > 0:
            self.logger.info(f"平均每步时间: {np.mean(self.step_times):.3f}s")
        self.logger.info("=" * 80)

