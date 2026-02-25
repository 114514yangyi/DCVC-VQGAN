"""
VQGAN - 指标评估（最小版）

提供：
- PSNR（基于 MSE）
- FID / LPIPS（可选，依赖 torchmetrics；未安装则跳过）

说明：
- 为了保持最小依赖，FID/LPIPS 使用 torchmetrics 的实现。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    HAS_FID_LPIPS = True
except Exception:
    HAS_FID_LPIPS = False
    FrechetInceptionDistance = None  # type: ignore
    LearnedPerceptualImagePatchSimilarity = None  # type: ignore


class MetricsEvaluator:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if HAS_FID_LPIPS:
            self.fid_metric = FrechetInceptionDistance(feature=64).to(self.device)
            self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(self.device)
        else:
            self.fid_metric = None
            self.lpips_metric = None

    @staticmethod
    def psnr_from_mse(mse: torch.Tensor, max_val: float = 1.0) -> float:
        mse_val = float(mse.detach().item())
        if mse_val <= 0:
            return float("inf")
        return 10.0 * math.log10((max_val**2) / mse_val)

    def reset_fid(self):
        if self.fid_metric is not None:
            self.fid_metric.reset()

    def update_fid_lpips(self, orig_01: torch.Tensor, recon_01: torch.Tensor) -> Dict[str, Any]:
        """
        orig_01 / recon_01: (N,3,H,W) in [0,1]
        """
        if self.fid_metric is None or self.lpips_metric is None:
            return {}

        orig_01 = orig_01.to(self.device)
        recon_01 = recon_01.to(self.device)

        orig_u8 = (orig_01.clamp(0, 1) * 255.0).to(torch.uint8)
        recon_u8 = (recon_01.clamp(0, 1) * 255.0).to(torch.uint8)

        # FID accumulate
        for i in range(orig_u8.shape[0]):
            self.fid_metric.update(orig_u8[i : i + 1], real=True)
            self.fid_metric.update(recon_u8[i : i + 1], real=False)

        # LPIPS expects [-1,1]
        orig_lp = orig_01 * 2.0 - 1.0
        recon_lp = recon_01 * 2.0 - 1.0
        lpips_vals = []
        for i in range(orig_lp.shape[0]):
            lp = self.lpips_metric(orig_lp[i : i + 1], recon_lp[i : i + 1])
            lpips_vals.append(float(lp.item()))

        return {"lpips": sum(lpips_vals) / len(lpips_vals) if lpips_vals else 0.0}

    def compute_fid_final(self) -> float:
        if self.fid_metric is None:
            return float("nan")
        return float(self.fid_metric.compute().item())

