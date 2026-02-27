"""
VQGAN - taming-vqgan 最小训练脚本

训练逻辑（最小闭环）：
- 从 config.json 读取参数
- 构建数据加载器（视频或图片序列）
- 直接实例化 taming-vqgan 模型（不使用 model_adapter 包装层）
- 每步：
  - 前向得到 vq_loss + recon
  - 使用 taming 的 loss（VQLPIPSWithDiscriminator）分别计算 ae / disc loss
  - 手动 step 两个优化器
  - 记录日志
  - 定期保存 checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import Normalize

# 让 VQGAN 目录成为 import 根，使得 `models/`, `data/`, `log_utils/` 可作为顶层包导入
_VQGAN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _VQGAN_ROOT not in sys.path:
    sys.path.insert(0, _VQGAN_ROOT)

from data.datasets import DataConfig, build_loaders  # noqa: E402
from log_utils.log_utils import build_logger  # noqa: E402
from metric_utils.metric_utils import MetricsEvaluator  # noqa: E402


@dataclass(frozen=True)
class TrainConfig:
    device: str
    num_steps: int
    lr: float
    disc_lr: float
    log_dir: str
    save_dir: str
    save_interval: int
    log_interval: int
    validation_interval: int
    best_max_images: int
    resume_ckpt: Optional[str]
    grad_accum_steps: int


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: str
    name: Optional[str]
    entity: Optional[str]
    log_interval: int
    image_interval: int


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_train(cfg: Dict[str, Any]) -> TrainConfig:
    t = cfg["train"]
    return TrainConfig(
        device=t.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        num_steps=int(t["num_steps"]),
        lr=float(t.get("lr", 1e-4)),
        disc_lr=float(t.get("disc_lr", t.get("lr", 1e-4))),
        log_dir=str(t["log_dir"]),
        save_dir=str(t["save_dir"]),
        save_interval=int(t.get("save_interval", 10000)),
        log_interval=int(t.get("log_interval", 50)),
        validation_interval=int(t.get("validation_interval", 0)),
        best_max_images=int(t.get("best_max_images", 2048)),
        resume_ckpt=t.get("resume_ckpt", None),
        grad_accum_steps=int(t.get("grad_accum_steps", 1)),
    )


def _parse_data(cfg: Dict[str, Any]) -> DataConfig:
    d = cfg["data"]
    return DataConfig(
        train_dir=str(d["train_dir"]),
        val_dir=str(d.get("val_dir", "")) or None,
        use_images=bool(d.get("use_images", False)),
        batch_size=int(d["batch_size"]),
        num_workers=int(d.get("num_workers", 4)),
        sequence_length=int(d["sequence_length"]),
        resolution=int(d.get("resolution", 256)),
        seed=int(d.get("seed", 2025)),
    )


def _parse_wandb(cfg: Dict[str, Any]) -> WandbConfig:
    w = cfg.get("wandb", {}) or {}
    return WandbConfig(
        enabled=bool(w.get("enabled", False)),
        project=str(w.get("project", "vqgan")),
        name=w.get("name", None),
        entity=w.get("entity", None),
        log_interval=int(w.get("log_interval", 50)),
        image_interval=int(w.get("image_interval", 500)),
    )


def _build_taming_model(cfg: Dict[str, Any], train_cfg: TrainConfig):
    """
    直接根据 config.json 的 model_args 构建 taming 的 VQGAN 模型。

    说明：
    - 不使用 model_adapter，避免包装层导致的 state_dict 保存/恢复异常。
    - 训练循环使用 encode/decode 获取 (x_rec, vq_loss, indices)。
    """
    from models.taming.models.vqgan import EMAVQ, VQModel  # noqa: E402

    model_args = dict(cfg.get("model_args", {}) or {})
    model_variant = str(model_args.get("model_variant", "VQModel"))

    ddconfig = model_args["ddconfig"]
    lossconfig = model_args.get("lossconfig", None)
    n_embed = int(model_args["n_embed"])
    embed_dim = int(model_args["embed_dim"])

    # RVQ 深度（Residual Quantization levels）
    num_quantizers = int(model_args.get("rvq_levels", model_args.get("num_quantizers", 1)))
    use_residual_rq = bool(model_args.get("use_residual_rq", True))

    if lossconfig is None:
        lossconfig = {
            "target": "models.taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
            "params": {
                "disc_conditional": False,
                "disc_in_channels": ddconfig.get("in_channels", 3),
                "disc_start": 10000,
                "disc_weight": 0.8,
                "codebook_weight": 1.0,
            },
        }

    if model_variant == "EMAVQ":
        taming_model = EMAVQ(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            n_embed=n_embed,
            embed_dim=embed_dim,
            ckpt_path=None,
            ignore_keys=[],
            image_key="image",
            colorize_nlabels=None,
            monitor=None,
            remap=None,
            sane_index_shape=False,
            num_quantizers=num_quantizers,
            use_residual_rq=use_residual_rq,
        )
    else:
        taming_model = VQModel(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            n_embed=n_embed,
            embed_dim=embed_dim,
            ckpt_path=None,
            ignore_keys=[],
            image_key="image",
            colorize_nlabels=None,
            monitor=None,
            remap=None,
            sane_index_shape=False,
        )

    taming_model.automatic_optimization = False
    taming_model.learning_rate = float(train_cfg.lr)
    return taming_model


def _save_checkpoint(
    save_dir: str,
    step: int,
    model,
    opt_ae,
    opt_disc,
    config: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename or f"checkpoint_step{step}.pt")
    payload = {
        "step": step,
        "model": model.state_dict(),
        "opt_ae": opt_ae.state_dict(),
        "opt_disc": opt_disc.state_dict() if opt_disc is not None else None,
        "config": config,
        "metrics": metrics or {},
    }
    torch.save(payload, path)
    return path


@torch.no_grad()
def evaluate(
    taming_model,
    val_loader,
    device: torch.device,
    normalize: Normalize,
    max_images: int,
) -> Dict[str, Any]:
    if val_loader is None:
        return {}

    taming_model.eval()
    metrics_eval = MetricsEvaluator(device=device)
    metrics_eval.reset_fid()

    # 尝试探测是否为多层 RVQ（ResidualEMAVectorQuantizer）
    quantizer = getattr(taming_model, "quantize", None)
    rvq_depth = int(getattr(quantizer, "num_quantizers", 1)) if quantizer is not None else 1
    # 每两层记录一次，如 D=4 -> 记录 d=2,4；D=6 -> 2,4,6
    rvq_depths = [d for d in range(2, rvq_depth + 1, 2)] if rvq_depth > 1 else []

    n_seen = 0
    mse_sum = 0.0
    n_mse = 0
    lpips_vals = []

    # 按 depth 聚合 RVQ 的指标
    rvq_mse_sum = {d: 0.0 for d in rvq_depths}
    rvq_n_mse = {d: 0 for d in rvq_depths}
    rvq_lpips_vals = {d: [] for d in rvq_depths}

    for batch in val_loader:
        x_u8 = batch.to(device=device, dtype=torch.float32)  # (B,D,H,W,C)
        b, d, h, w, c = x_u8.shape
        n_batch = b * d
        if n_seen + n_batch > max_images:
            keep = max_images - n_seen
            if keep <= 0:
                break
            # 仅截断 D（保持最小实现）
            d_keep = min(d, keep)
            x_u8 = x_u8[:, :d_keep]
            b, d, h, w, c = x_u8.shape
            n_batch = b * d

        x_01 = rearrange(x_u8, "b d h w c -> (b d) c h w") / 255.0  # [0,1]
        x_in = normalize(x_01)  # [-1,1]

        quant, vq_loss, _info = taming_model.encode(x_in)
        x_rec = taming_model.decode(quant)

        # recon/orig 都用 [0,1] 计算指标
        x_rec_01 = (x_rec * 0.5 + 0.5).clamp(0.0, 1.0)
        x_01 = x_01.clamp(0.0, 1.0)

        mse = torch.mean((x_rec_01 - x_01) ** 2)
        mse_sum += float(mse.item()) * n_batch
        n_mse += n_batch

        fid_lpips = metrics_eval.update_fid_lpips(x_01, x_rec_01)
        if "lpips" in fid_lpips:
            lpips_vals.append(float(fid_lpips["lpips"]))

        # ====== 额外：针对 RVQ 每两层的指标（例如只用前 2 层、前 4 层） ======
        if rvq_depths:
            # 使用底层 taming 模型的 encode 拿到多层 code indices
            try:
                quant_full, _emb_loss, info = taming_model.encode(x_in)
            except Exception:
                info = None

            if info is not None and len(info) >= 3:
                indices = info[2]
                # 只在 indices 为 [B, D, H, W] 时计算分层指标
                if isinstance(indices, torch.Tensor) and indices.ndim == 4:
                    for d_depth in rvq_depths:
                        if d_depth > indices.shape[1]:
                            continue
                        partial_idx = indices[:, :d_depth, :, :]  # [B, d, H, W]
                        try:
                            quant_d = taming_model.quantize.embed_code(partial_idx)
                            x_rec_d = taming_model.decode(quant_d)
                        except Exception:
                            continue

                        x_rec_d_01 = (x_rec_d * 0.5 + 0.5).clamp(0.0, 1.0)
                        # MSE / PSNR
                        mse_d = torch.mean((x_rec_d_01 - x_01) ** 2)
                        rvq_mse_sum[d_depth] += float(mse_d.item()) * n_batch
                        rvq_n_mse[d_depth] += n_batch

                        # LPIPS（若可用）
                        if metrics_eval.lpips_metric is not None:
                            orig_lp = x_01 * 2.0 - 1.0
                            rec_lp = x_rec_d_01 * 2.0 - 1.0
                            lp_batch = []
                            for i in range(orig_lp.shape[0]):
                                lp = metrics_eval.lpips_metric(
                                    orig_lp[i : i + 1], rec_lp[i : i + 1]
                                )
                                lp_batch.append(float(lp.item()))
                            if lp_batch:
                                rvq_lpips_vals[d_depth].append(
                                    sum(lp_batch) / len(lp_batch)
                                )

        n_seen += n_batch

    avg_mse = mse_sum / max(n_mse, 1)
    psnr = MetricsEvaluator.psnr_from_mse(torch.tensor(avg_mse, device=device))
    fid = metrics_eval.compute_fid_final()
    lpips = sum(lpips_vals) / len(lpips_vals) if lpips_vals else float("nan")

    taming_model.train()
    results: Dict[str, Any] = {
        "val/psnr": psnr,
        "val/fid": fid,
        "val/lpips": lpips,
        "val/mse": avg_mse,
        "val/n_images": n_seen,
    }

    # 汇总并写入每两层 RVQ 的指标
    for d_depth in rvq_depths:
        if rvq_n_mse[d_depth] <= 0:
            continue
        avg_mse_d = rvq_mse_sum[d_depth] / rvq_n_mse[d_depth]
        psnr_d = MetricsEvaluator.psnr_from_mse(torch.tensor(avg_mse_d, device=device))
        lpips_d = (
            sum(rvq_lpips_vals[d_depth]) / len(rvq_lpips_vals[d_depth])
            if rvq_lpips_vals[d_depth]
            else float("nan")
        )
        results[f"val/psnr_rvq_d{d_depth}"] = psnr_d
        results[f"val/lpips_rvq_d{d_depth}"] = lpips_d
        results[f"val/mse_rvq_d{d_depth}"] = avg_mse_d

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    train_cfg = _parse_train(cfg)
    data_cfg = _parse_data(cfg)
    wandb_cfg = _parse_wandb(cfg)

    os.makedirs(train_cfg.log_dir, exist_ok=True)
    os.makedirs(train_cfg.save_dir, exist_ok=True)
    logger = build_logger(train_cfg.log_dir, name="VQGAN")
    logger.info(f"config: {os.path.abspath(args.config)}")

    train_loader, val_loader, n_train_files = build_loaders(data_cfg)
    logger.info(f"train files: {n_train_files}")
    logger.info(
        f"use_images={data_cfg.use_images}, batch_size={data_cfg.batch_size}, seq={data_cfg.sequence_length}, res={data_cfg.resolution}"
    )

    # 直接创建 taming 模型（不使用适配器）
    taming_model = _build_taming_model(cfg, train_cfg)
    device = torch.device(train_cfg.device)
    taming_model = taming_model.to(device)

    # taming 优化器（来自 configure_optimizers）
    optimizers, _schedulers = taming_model.configure_optimizers()
    opt_ae = optimizers[0]
    opt_disc = optimizers[1] if isinstance(optimizers, (list, tuple)) and len(optimizers) > 1 else None
    # 允许在 config 中单独设置判别器学习率
    if opt_disc is not None:
        for pg in opt_disc.param_groups:
            pg["lr"] = float(train_cfg.disc_lr)

    # normalize: [0,1] -> [-1,1]
    normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    wb_run = None
    if wandb_cfg.enabled:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb 已在 config.json 中启用，但当前环境未安装。请先执行: pip install -r VQGAN/requirements.txt"
            ) from e

        wb_run = wandb.init(
            project=wandb_cfg.project,
            name=wandb_cfg.name,
            entity=wandb_cfg.entity,
            config=cfg,
        )

    start_step = 0
    if train_cfg.resume_ckpt:
        ckpt = torch.load(train_cfg.resume_ckpt, map_location=device)
        if isinstance(ckpt.get("model", None), dict) and len(ckpt["model"]) == 0:
            logger.warning(
                "checkpoint 的 ckpt['model'] 为空 dict：这通常意味着保存时 model.state_dict() 为空（适配器未注册子模块）。"
                "这种 ckpt 无法恢复模型权重，只能从头训练。建议使用修复后的代码重新训练并保存新 ckpt。"
            )
        taming_model.load_state_dict(ckpt["model"])
        opt_ae.load_state_dict(ckpt["opt_ae"])
        if opt_disc is not None and ckpt.get("opt_disc") is not None:
            opt_disc.load_state_dict(ckpt["opt_disc"])
        start_step = int(ckpt.get("step", 0))
        logger.info(f"resumed from {train_cfg.resume_ckpt} @ step={start_step}")

    best_psnr = float("-inf")
    best_fid = float("inf")
    best_lpips = float("inf")

    data_iter = iter(train_loader)

    # 累计梯度相关设置
    grad_accum_steps = max(1, int(train_cfg.grad_accum_steps))
    step_offset = start_step

    for step in range(start_step, train_cfg.num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # batch: (B,D,H,W,C) uint8
        x = batch.to(device=device, dtype=torch.float32)
        b, d, h, w, c = x.shape
        x = rearrange(x, "b d h w c -> (b d) c h w")
        x = normalize(x / 255.0)

        quant, vq_loss, _info = taming_model.encode(x)
        x_rec = taming_model.decode(quant)

        # taming loss (ae)
        last_layer = taming_model.get_last_layer()
        loss_ae, log_ae = taming_model.loss(
            codebook_loss=vq_loss.mean(),
            inputs=x,
            reconstructions=x_rec,
            optimizer_idx=0,
            global_step=step,
            last_layer=last_layer,
            cond=None,
            split="train",
        )
        # 累计梯度：按 grad_accum_steps 归一化 loss 并延迟 step
        loss_ae_scaled = loss_ae / grad_accum_steps
        if (step - step_offset) % grad_accum_steps == 0:
            opt_ae.zero_grad(set_to_none=True)
        loss_ae_scaled.backward()

        loss_disc = torch.tensor(0.0, device=device)
        if opt_disc is not None:
            loss_disc, log_disc = taming_model.loss(
                codebook_loss=vq_loss.mean(),
                inputs=x,
                reconstructions=x_rec.detach(),
                optimizer_idx=1,
                global_step=step,
                last_layer=last_layer,
                cond=None,
                split="train",
            )
            loss_disc_scaled = loss_disc / grad_accum_steps
            if (step - step_offset) % grad_accum_steps == 0:
                opt_disc.zero_grad(set_to_none=True)
            loss_disc_scaled.backward()
        else:
            log_disc = {}

        # 在累计到指定步数时再执行优化器 step
        if (step - step_offset + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(taming_model.parameters(), max_norm=1.0)
            opt_ae.step()
            if opt_disc is not None:
                opt_disc.step()

        if step % train_cfg.log_interval == 0:
            rec_loss = log_ae.get("train/rec_loss", None)
            g_loss = log_ae.get("train/g_loss", None)
            d_loss = log_disc.get("train/disc_loss", None)
            disc_factor = log_ae.get("train/disc_factor", None)
            logger.info(
                f"step {step}/{train_cfg.num_steps} | "
                f"loss_ae={float(loss_ae):.6f} loss_disc={float(loss_disc):.6f} | "
                f"rec={float(rec_loss) if rec_loss is not None else float('nan'):.6f} | "
                f"vq={float(vq_loss.mean()):.6f} | "
                f"g={float(g_loss) if g_loss is not None else float('nan'):.6f} "
                f"d={float(d_loss) if d_loss is not None else float('nan'):.6f} "
                f"disc_factor={float(disc_factor) if disc_factor is not None else float('nan'):.4f}"
            )

        if wb_run is not None and step % wandb_cfg.log_interval == 0:
            wb_run.log(
                {
                    "train/loss_ae": float(loss_ae.item()),
                    "train/loss_disc": float(loss_disc.item()),
                    "train/vq_loss": float(vq_loss.mean().item()),
                },
                step=step,
            )

        if wb_run is not None and step % wandb_cfg.image_interval == 0:
            # x/x_rec 当前在 [-1,1]，转回 [0,1]
            x_01 = (x * 0.5 + 0.5).clamp(0, 1)
            xrec_01 = (x_rec * 0.5 + 0.5).clamp(0, 1)
            n_show = min(4, x_01.shape[0])
            grid = make_grid(torch.cat([xrec_01[:n_show], x_01[:n_show]], dim=0), nrow=n_show)
            # Run 对象没有 Image 属性，这里应使用 wandb.Image
            import wandb as _wandb  # 局部引用以避免未启用 wandb 时出错
            img = _wandb.Image(grid.permute(1, 2, 0).cpu().numpy())
            wb_run.log({"viz/recon_top_orig_bottom": img}, step=step)

        if train_cfg.save_interval > 0 and step > 0 and step % train_cfg.save_interval == 0:
            ckpt_path = _save_checkpoint(train_cfg.save_dir, step, taming_model, opt_ae, opt_disc, cfg)
            logger.info(f"saved: {ckpt_path}")

        if (
            train_cfg.validation_interval > 0
            and val_loader is not None
            and step > 0
            and step % train_cfg.validation_interval == 0
        ):
            val_metrics = evaluate(
                taming_model=taming_model,
                val_loader=val_loader,
                device=device,
                normalize=normalize,
                max_images=train_cfg.best_max_images,
            )
            if val_metrics:
                logger.info(
                    f"val @ step {step} | psnr={val_metrics['val/psnr']:.4f} "
                    f"fid={val_metrics['val/fid']:.4f} lpips={val_metrics['val/lpips']:.6f}"
                )
                # 如存在 RVQ 分层指标，一并打印（例如：只用前 2/4 层量化时的 PSNR / LPIPS）
                rvq_keys = sorted(
                    [k for k in val_metrics.keys() if k.startswith("val/psnr_rvq_d")]
                )
                for k in rvq_keys:
                    d_str = k.split("d")[-1]
                    psnr_d = val_metrics.get(k, float("nan"))
                    lpips_k = f"val/lpips_rvq_d{d_str}"
                    lpips_d = val_metrics.get(lpips_k, float("nan"))
                    logger.info(
                        f"  rvq depth {d_str}: psnr={psnr_d:.4f}, lpips={lpips_d:.6f}"
                    )
                if wb_run is not None:
                    wb_run.log(val_metrics, step=step)

                if val_metrics["val/psnr"] > best_psnr:
                    best_psnr = val_metrics["val/psnr"]
                    path = _save_checkpoint(
                        train_cfg.save_dir,
                        step,
                        taming_model,
                        opt_ae,
                        opt_disc,
                        cfg,
                        metrics=val_metrics,
                        filename=f"best_psnr_step{step}.pt",
                    )
                    logger.info(f"saved best psnr: {path}")

                if not math.isnan(val_metrics["val/fid"]) and val_metrics["val/fid"] < best_fid:
                    best_fid = val_metrics["val/fid"]
                    path = _save_checkpoint(
                        train_cfg.save_dir,
                        step,
                        taming_model,
                        opt_ae,
                        opt_disc,
                        cfg,
                        metrics=val_metrics,
                        filename=f"best_fid_step{step}.pt",
                    )
                    logger.info(f"saved best fid: {path}")

                if not math.isnan(val_metrics["val/lpips"]) and val_metrics["val/lpips"] < best_lpips:
                    best_lpips = val_metrics["val/lpips"]
                    path = _save_checkpoint(
                        train_cfg.save_dir,
                        step,
                        taming_model,
                        opt_ae,
                        opt_disc,
                        cfg,
                        metrics=val_metrics,
                        filename=f"best_lpips_step{step}.pt",
                    )
                    logger.info(f"saved best lpips: {path}")

    ckpt_path = _save_checkpoint(train_cfg.save_dir, train_cfg.num_steps, taming_model, opt_ae, opt_disc, cfg)
    logger.info(f"done. saved final: {ckpt_path}")

    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    main()

