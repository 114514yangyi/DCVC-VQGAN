import math
from typing import Optional, Dict, Any

import torch
from einops import rearrange

# 尝试导入 FID 和 LPIPS 相关库
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    HAS_FID_LPIPS = True
except ImportError:
    HAS_FID_LPIPS = False
    print("警告: 未安装 torchmetrics，FID 和 LPIPS 指标将不可用")
    print("安装方法: pip install torchmetrics")


class MetricsEvaluator:
    """
    统一评估指标：
    - PSNR
    - Temporal Consistency（帧间差一致性）
    - Codebook 使用率与熵（需 encoding_indices）
    - Bitrate/压缩比估计（基于 code 序列）
    """

    def __init__(self, vocab_size: Optional[int] = None, device: Optional[torch.device] = None):
        self.vocab_size = vocab_size
        # 累积直方图，用于统计「全局」码本使用情况
        self._global_hist = None
        
        # 初始化 FID 和 LPIPS 指标（如果可用）
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if HAS_FID_LPIPS:
            self.fid_metric = FrechetInceptionDistance(feature=64).to(self.device)
            self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(self.device)
            self._fid_reset = False  # 标记是否需要重置 FID 指标
        else:
            self.fid_metric = None
            self.lpips_metric = None

    @staticmethod
    def psnr_from_mse(mse: torch.Tensor, max_val: float = 1.0) -> float:
        mse_val = mse.detach().item()
        if mse_val <= 0:
            return float("inf")
        return 10.0 * math.log10((max_val ** 2) / mse_val)

    @staticmethod
    def temporal_consistency(orig: torch.Tensor, recon: torch.Tensor, b: int, d: int) -> float:
        """
        计算帧间差的 MSE 差异，值越小代表时间一致性越好。
        orig/recon: (b*d, c, h, w) 归一化后的张量
        """
        try:
            orig_seq = rearrange(orig, '(b d) c h w -> b d c h w', b=b, d=d)
            recon_seq = rearrange(recon, '(b d) c h w -> b d c h w', b=b, d=d)
            if d < 2:
                return 0.0
            diff_o = orig_seq[:, 1:] - orig_seq[:, :-1]
            diff_r = recon_seq[:, 1:] - recon_seq[:, :-1]
            return torch.mean((diff_o - diff_r) ** 2).item()
        except Exception:
            return float("nan")

    def codebook_stats(self, encoding_indices: Optional[torch.Tensor]) -> Dict[str, Any]:
        """
        返回本 batch 使用率与熵；同时维护一个跨 batch 的全局使用率与熵。
        """
        if encoding_indices is None or self.vocab_size is None:
            return {}
        flat = encoding_indices.reshape(-1).detach()
        # 这行代码使用 PyTorch 的 torch.bincount() 函数来统计张量中每个值的出现次数。
        hist = torch.bincount(flat, minlength=self.vocab_size).float()
        total = hist.sum()
        if total <= 0:
            return {}
        probs = hist / total
        usage = (hist > 0).float().mean().item()
        entropy = -(probs[probs > 0] * torch.log2(probs[probs > 0])).sum().item()
        result = {
            "codebook_usage": usage,
            "codebook_entropy": entropy,
        }

        # ===== 更新全局统计（跨所有已见 batch）=====
        if self._global_hist is None:
            # 在第一次调用时初始化为当前直方图
            self._global_hist = hist.clone()
        else:
            # 累加当前 batch 的使用次数
            self._global_hist += hist

        global_total = self._global_hist.sum()
        if global_total > 0:
            global_probs = self._global_hist / global_total
            global_usage = (self._global_hist > 0).float().mean().item()
            global_entropy = -(global_probs[global_probs > 0] *
                               torch.log2(global_probs[global_probs > 0])).sum().item()
            result.update({
                "codebook_usage_global": global_usage,
                "codebook_entropy_global": global_entropy,
            })

        return result

    def bitrate_and_ratio(self, encoding_indices: Optional[torch.Tensor], vocab_size: Optional[int],
                          orig_shape: torch.Size) -> Dict[str, Any]:
        """
        估算比特率与压缩比：
        - 编码比特 = 序列长度 * log2(vocab_size)
        - 原始比特 = 像素数 * 8
        """
        if encoding_indices is None or vocab_size is None:
            return {}
        num_tokens = encoding_indices.numel()
        bits_per_token = math.log2(vocab_size)
        encoded_bits = num_tokens * bits_per_token
        raw_bits = 1
        for dim in orig_shape:
            raw_bits *= dim
        raw_bits *= 8  # uint8 假设
        bitrate = encoded_bits  # 未除时间长度，作为估计值
        compression_ratio = raw_bits / encoded_bits if encoded_bits > 0 else float("inf")
        return {
            "bitrate": bitrate,
            "compression_ratio": compression_ratio
        }

    def compute_fid_lpips(self, orig: torch.Tensor, recon: torch.Tensor) -> Dict[str, Any]:
        """
        计算 FID 和 LPIPS 指标
        
        Args:
            orig: 原始图像 (b*d, c, h, w)，值在 [0, 1] 范围内（归一化后）
            recon: 重建图像 (b*d, c, h, w)，值在 [0, 1] 范围内（归一化后）
        
        Returns:
            包含 FID 和 LPIPS 的字典
        """
        if not HAS_FID_LPIPS or self.fid_metric is None or self.lpips_metric is None:
            return {}
        
        metrics = {}
        
        try:
            # 确保张量在正确的设备上
            orig = orig.to(self.device)
            recon = recon.to(self.device)
            
            # 转换为 [0, 255] 范围用于 FID（需要 uint8 格式）
            # orig 和 recon 当前是归一化后的 [0, 1] 范围
            # 先反归一化到 [0, 1]，然后转换为 [0, 255]
            orig_denorm = orig.clamp(0.0, 1.0)  # 确保在 [0, 1] 范围内
            recon_denorm = recon.clamp(0.0, 1.0)  # 确保在 [0, 1] 范围内
            
            orig_uint8 = (orig_denorm * 255.0).clamp(0, 255).to(torch.uint8)
            recon_uint8 = (recon_denorm * 255.0).clamp(0, 255).to(torch.uint8)
            
            # FID 需要 RGB 格式，值在 [0, 255] 范围内
            # 假设输入已经是 RGB 格式 (b*d, c, h, w)，其中 c=3
            # 更新 FID 指标（逐帧更新）
            for i in range(orig_uint8.shape[0]):
                self.fid_metric.update(orig_uint8[i:i+1], real=True)
                self.fid_metric.update(recon_uint8[i:i+1], real=False)
            
            # 计算 LPIPS（需要归一化到 [-1, 1]）
            # 从 [0, 1] 转换为 [-1, 1]
            orig_lpips = orig_denorm * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            recon_lpips = recon_denorm * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            
            # 计算 LPIPS（逐帧计算然后平均）
            lpips_values = []
            for i in range(orig_lpips.shape[0]):
                lpips_val = self.lpips_metric(orig_lpips[i:i+1], recon_lpips[i:i+1])
                lpips_values.append(lpips_val.item())
            
            metrics["lpips"] = sum(lpips_values) / len(lpips_values) if lpips_values else 0.0
            
        except Exception as e:
            print(f"警告: 计算 FID/LPIPS 时出错: {e}")
            metrics["lpips"] = float("nan")
        
        return metrics
    
    def compute_fid_final(self) -> float:
        """
        计算最终的 FID 值（在所有数据更新后调用）
        
        Returns:
            FID 值，如果计算失败则返回 NaN
        """
        if not HAS_FID_LPIPS or self.fid_metric is None:
            return float("nan")
        
        try:
            fid_value = self.fid_metric.compute().item()
            return fid_value
        except Exception as e:
            print(f"警告: 计算最终 FID 时出错: {e}")
            return float("nan")
    
    def reset_fid(self):
        """重置 FID 指标（用于新的评估周期）"""
        if HAS_FID_LPIPS and self.fid_metric is not None:
            self.fid_metric.reset()
            self._fid_reset = True

    def compute_all(self,
                    orig: torch.Tensor,
                    recon: torch.Tensor,
                    recon_mse: torch.Tensor,
                    perplexity: Optional[torch.Tensor],
                    encoding_indices: Optional[torch.Tensor],
                    b: int,
                    d: int,
                    compute_fid_lpips: bool = False) -> Dict[str, Any]:
        """
        计算所有指标
        
        Args:
            compute_fid_lpips: 是否计算 FID 和 LPIPS（这些指标计算较慢）
        """
        metrics = {}
        metrics["psnr"] = self.psnr_from_mse(recon_mse)
        metrics["temporal_consistency"] = self.temporal_consistency(orig, recon, b, d)
        metrics.update(self.codebook_stats(encoding_indices))
        metrics.update(self.bitrate_and_ratio(encoding_indices, self.vocab_size, orig.shape))
        if perplexity is not None:
            metrics["perplexity"] = perplexity.detach().item()
        
        # 计算 FID 和 LPIPS（如果请求）
        if compute_fid_lpips:
            fid_lpips_metrics = self.compute_fid_lpips(orig, recon)
            metrics.update(fid_lpips_metrics)
        
        return metrics

