import logging
import os
from datetime import datetime
from typing import Optional, Dict

import torch
from einops import rearrange

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False
    wandb = None  # type: ignore


from train.train_utils import train_visualize, save_images


class LogManager:
    """
    统一管理本地日志和 wandb 日志，训练代码只关注训练逻辑。
    """

    def __init__(self,
                 log_dir: str,
                 use_wandb: bool = False,
                 project_name: Optional[str] = None,
                 config: Optional[Dict] = None,
                 resume: bool = False,
                 logger_name: str = "VqVaeTraining",
                 id_name: str = "VqVaeTraining"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logger = self._build_logger(logger_name, log_dir)
        self.id_name = id_name
        self.run_wandb = None
        
        # 初始化 wandb
        if use_wandb and _WANDB_AVAILABLE:
            self.logger.info("初始化 wandb...")
            self.run_wandb = wandb.init(
                project=project_name or "vq_vae_project",
                job_type="train_model",
                config=config,
                id=id_name,
                resume="allow",
                dir="/data/huyang/wandb"
            )
            self.logger.info("wandb 初始化完成")
        elif use_wandb and not _WANDB_AVAILABLE:
            self.logger.warning("请求使用 wandb，但未安装或不可用，已跳过 wandb 初始化")

    @staticmethod
    def _build_logger(name: str, log_dir: str) -> logging.Logger:
        log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if logger.handlers:
            return logger

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        console_handler.setFormatter(console_format)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info(f"日志系统已初始化，日志文件保存在: {log_file}")
        return logger

    def info(self, msg: str):
        self.logger.info(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def log_metrics(self, step: int, metrics: Dict):
        """
        记录指标到 wandb
        """
        # 记录到 wandb
        if self.run_wandb is not None:
            if hasattr(self.run_wandb, "log") and self.run_wandb.log is not None and callable(self.run_wandb.log):
                try:
                    self.run_wandb.log(metrics, step=step)
                except Exception as e:
                    self.logger.warning(f"wandb 指标记录失败 (step {step}): {e}")

    def log_images(self,
                   step: int,
                   images: torch.Tensor,
                   images_recon: torch.Tensor,
                   b: int,
                   d: int,
                   n_images_save: int,
                   wandb_max_images: int,
                   unnormalize: torch.nn.Module,
                   save_dir_orig: Optional[str] = None,
                   save_dir_recs: Optional[str] = None,
                   hand_cropper=None):
        """
        将原/重建图像保存到本地并（可选）记录到 wandb。
        images: (b*d, c, h, w)
        """
        # 本地保存
        images_vis = rearrange(images, '(b d) c h w -> b d c h w', b=b, d=d, c=3)
        images_recon_vis = rearrange(images_recon, '(b d) c h w -> b d c h w', b=b, d=d, c=3)
        n_frames = min(wandb_max_images, n_images_save, images_vis.shape[1])
        
        # 提取要可视化的帧，最多4张
        n_show = min(4, n_frames)
        images_frames = images_vis[0, :n_show]  # (n_show, c, h, w)
        images_recon_frames = images_recon_vis[0, :n_show]  # (n_show, c, h, w)
        
        # 反归一化
        images_orig = unnormalize(images_frames).detach().cpu()  # (n_show, c, h, w)
        images_recs = unnormalize(images_recon_frames).detach().cpu()  # (n_show, c, h, w)
        
        # 避免数值越界导致全黑，限制到 [0,1]
        images_orig = images_orig.clamp(0.0, 1.0)
        images_recs = images_recs.clamp(0.0, 1.0)
        
        # 将图像拼接成左右两列布局
        # 左边列：重建图像（垂直堆叠）
        # 右边列：原始图像（垂直堆叠）
        # 图像格式: (n_show, c, h, w) -> 需要转换为 (c, n_show*h, w) 进行垂直堆叠
        
        # 垂直堆叠重建图像: (n_show, c, h, w) -> (c, n_show*h, w)
        left_column = torch.cat([images_recs[i] for i in range(n_show)], dim=1)  # (c, n_show*h, w)
        
        # 垂直堆叠原始图像: (n_show, c, h, w) -> (c, n_show*h, w)
        right_column = torch.cat([images_orig[i] for i in range(n_show)], dim=1)  # (c, n_show*h, w)
        
        # 水平拼接两列: (c, n_show*h, w) + (c, n_show*h, w) -> (c, n_show*h, 2*w)
        combined_image = torch.cat([left_column, right_column], dim=2)  # (c, n_show*h, 2*w)
        
        # if save_dir_orig and save_dir_recs and step % 2500==0:
        #     os.makedirs(save_dir_orig, exist_ok=True)
        #     os.makedirs(save_dir_recs, exist_ok=True)
        #     orig_path = os.path.join(save_dir_orig, f'image_{step}.png')
        #     recs_path = os.path.join(save_dir_recs, f'image_{step}.png')
            # save_images(file_name=orig_path, image=images_orig)
            # save_images(file_name=recs_path, image=images_recs)
            # self.logger.info(f"可视化结果已保存: {orig_path}, {recs_path}")

        # 记录图像到 wandb
        # combined_image 格式: (c, n_show*h, 2*w)
        
        # 记录到 wandb
        if self.run_wandb is not None:
            if hasattr(self.run_wandb, "log") and self.run_wandb.log is not None and callable(self.run_wandb.log):
                try:
                    # 确保图像格式正确：转换为 (H, W, C) 格式用于 wandb
                    if combined_image.dim() == 3 and combined_image.shape[0] == 3:
                        # (C, H, W) -> (H, W, C)
                        combined_image_np = combined_image.permute(1, 2, 0).numpy()
                    else:
                        combined_image_np = combined_image.numpy()
                    
                    # 确保值在 [0, 1] 范围内
                    combined_image_np = combined_image_np.clip(0.0, 1.0)
                    
                    # 记录合并后的图像到 wandb
                    import wandb
                    self.run_wandb.log({
                        'viz/recon-original': wandb.Image(combined_image_np, caption=f'recon(left)_original(right)_step_{step}')
                    }, step=step)
                except Exception as e:
                    self.logger.warning(f"wandb 图片记录失败 (step {step}): {e}")

    def finish(self):
        """
        关闭所有日志记录器
        """
        # 关闭 wandb
        if self.run_wandb is not None and hasattr(self.run_wandb, "finish") and callable(self.run_wandb.finish):
            try:
                self.run_wandb.finish()
                self.logger.info("wandb 已关闭")
            except Exception as e:
                self.logger.warning(f"关闭 wandb 时出错: {e}")

