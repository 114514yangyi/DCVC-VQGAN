"""
VQGAN 数据准备模块

- 使用 VQGAN 将训练/验证视频编码为 codebook index，按 sequence_length 切分为 transformer 训练样本。
- 验证集：同时生成重构视频并保存，得到 (index, 原视频路径, 重构视频路径) 用于评估阶段生成 CSV。
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np
import torch
from einops import rearrange
from torchvision import transforms
from tqdm import tqdm

_VQGAN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _VQGAN_ROOT not in sys.path:
    sys.path.insert(0, _VQGAN_ROOT)

from data.datasets import list_files
from models.model_adapter import create_model

logger = logging.getLogger(__name__)


def list_videos(video_dir: str, exts: Tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".webm")) -> List[str]:
    """列出目录下所有视频文件（绝对路径），排序后返回。"""
    paths = list_files(video_dir, exts)
    paths.sort()
    return paths


def load_vqgan_model(vq_config: str, vq_ckpt: str, device: str):
    """
    加载 VQGAN 模型（适配 VQGAN 项目 config 与 checkpoint 格式）。
    - config 中使用 data / model_args；checkpoint 可能为 state_dict 或 model。
    """
    with open(vq_config, "r", encoding="utf-8") as f:
        config = json.load(f)
    data_cfg = config.get("data", config.get("data_args", {}))
    model_args = dict(config.get("model_args", {}))
    sequence_length = data_cfg.get("sequence_length", 8)
    if "sequence_length" not in model_args:
        model_args = {**model_args, "sequence_length": sequence_length}

    model = create_model(config_path=vq_config, model_args=model_args)
    model = model.to(device)

    ckpt = torch.load(vq_ckpt, map_location=device)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, config


def _get_indices_from_encode(vqgan_model, x: torch.Tensor, device: str) -> torch.Tensor:
    """调用 VQGAN 内部 encode，返回 codebook indices (B, H, W)。"""
    inner = vqgan_model._get_model()
    quant, emb_loss, info = inner.encode(x)
    if info is None or len(info) < 3:
        raise ValueError("VQGAN encode 未返回 indices (info[2])")
    indices = info[2]
    if indices.ndim == 1:
        b, _, h, w = quant.shape
        indices = indices.view(b, h, w)
    return indices


def encode_video_to_indices(
    vqgan_model,
    video: torch.Tensor,
    device: str,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    使用 VQGAN 将视频编码为 codebook indices。
    video: (num_frames, H, W, C) uint8 → 返回 (num_frames, h, w) int indices。
    """
    num_frames, h, w, c = video.shape
    video = video.float().to(device)
    video = rearrange(video, "d h w c -> d c h w")
    video = video / 255.0
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    video = normalize(video)

    all_indices = []
    with torch.no_grad():
        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            batch_video = video[i:end_idx]
            batch_indices = _get_indices_from_encode(vqgan_model, batch_video, device)
            all_indices.append(batch_indices)
    return torch.cat(all_indices, dim=0)


def load_video_frames(video_path: str, image_size: int = 256) -> np.ndarray:
    """加载视频全部帧，返回 (num_frames, H, W, C) uint8。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (image_size, image_size):
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"视频无有效帧: {video_path}")
    return np.stack(frames, axis=0).astype(np.uint8)


def decode_indices_to_video(
    vqgan_model,
    indices: torch.Tensor,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    将 codebook indices 用 VQGAN 解码为视频帧。
    indices: (num_frames, h, w) → 返回 (num_frames, H, W, C) uint8。
    """
    num_frames = indices.shape[0]
    indices = indices.to(device)
    inner = vqgan_model._get_model()
    unnorm = _get_unnormalize()

    all_frames = []
    with torch.no_grad():
        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            batch_indices = indices[i:end_idx]
            quant_b = inner.quantize.embed_code(batch_indices)
            dec = inner.decode(quant_b)
            dec = unnorm(dec)
            dec = dec.clamp(0, 1)
            dec = (dec * 255).cpu().numpy().astype(np.uint8)
            dec = rearrange(dec, "d c h w -> d h w c")
            all_frames.append(dec)
    return np.concatenate(all_frames, axis=0)


def _get_unnormalize():
    """与 data 归一化 (mean=0.5, std=0.5) 对应的反归一化: x * 0.5 + 0.5。"""
    def unnorm(x):
        return x * 0.5 + 0.5
    return unnorm


def process_videos_to_indices(
    video_dir: str,
    vq_config: str,
    vq_ckpt: str,
    sequence_length: int,
    image_size: int = 256,
    device: str = "cuda:0",
    num_samples: Optional[int] = None,
    max_size: int = 100000,
) -> np.ndarray:
    """
    从视频目录用 VQGAN 编码并切分为训练样本。
    返回 [num_samples, seq_len]，seq_len = sequence_length * h * w。
    """
    vqgan_model, config = load_vqgan_model(vq_config, vq_ckpt, device)
    model_args = config.get("model_args", {})
    vocab_size = model_args.get("n_embed", model_args.get("vocab_size", 1024))

    video_files = list_videos(video_dir)
    if not video_files:
        raise ValueError(f"目录下无视频: {video_dir}")

    all_samples = []
    device_torch = torch.device(device)
    for video_path in tqdm(video_files, desc="训练集编码"):
        try:
            if len(all_samples) >= max_size:
                break
            video_np = load_video_frames(video_path, image_size)
            video_tensor = torch.from_numpy(video_np)
            indices = encode_video_to_indices(vqgan_model, video_tensor, device_torch)
            del video_np, video_tensor
            if device_torch.type == "cuda":
                torch.cuda.empty_cache()

            num_frames, ih, iw = indices.shape
            indices_per_frame = ih * iw
            num_segments = num_frames // sequence_length
            for i in range(num_segments):
                start = i * sequence_length
                end = start + sequence_length
                seg = indices[start:end].flatten().cpu().numpy().astype(np.int64)
                all_samples.append(seg)
                if num_samples is not None and len(all_samples) >= num_samples:
                    break
            del indices
            if device_torch.type == "cuda":
                torch.cuda.empty_cache()
            if num_samples is not None and len(all_samples) >= num_samples:
                break
        except Exception as e:
            logger.warning("处理视频失败 %s: %s", video_path, e)
            if device_torch.type == "cuda":
                torch.cuda.empty_cache()
            continue

    if not all_samples:
        raise ValueError("未生成任何训练样本")
    data = np.array(all_samples, dtype=np.int64)
    del vqgan_model
    if device_torch.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("训练集: %d 样本, 每样本长度 %d", len(data), data.shape[1])
    return data


def process_val_videos_to_indices_and_recon(
    video_dir: str,
    vq_config: str,
    vq_ckpt: str,
    sequence_length: int,
    image_size: int,
    device: str,
    recon_output_dir: str,
    original_output_dir: str,
    num_samples: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    处理验证集视频：编码为 indices 并生成重构视频保存到磁盘。
    返回:
        val_data: [N, seq_len] 的 index 数组（与训练集相同的切分方式，用于 DataLoader）
        val_meta: 每条对应一个验证样本（按视频或按段），用于写 CSV：
                  [{"original_path", "recon_path", "indices": 该段/视频的 indices 或 None, "num_indices"}]
                  其中 size 在评估时由 transformer 概率估算，这里只提供路径与索引信息。
    """
    os.makedirs(recon_output_dir, exist_ok=True)
    os.makedirs(original_output_dir, exist_ok=True)
    vqgan_model, config = load_vqgan_model(vq_config, vq_ckpt, device)
    data_cfg = config.get("data", {})
    fps = max(1, float(data_cfg.get("fps", 30)))
    device_torch = torch.device(device)
    unnorm = _get_unnormalize()
    inner = vqgan_model._get_model()

    video_files = list_videos(video_dir)
    if not video_files:
        raise ValueError(f"验证目录下无视频: {video_dir}")

    all_samples = []
    val_meta = []
    for video_path in tqdm(video_files, desc="验证集编码与重构"):
        try:
            if num_samples is not None and len(all_samples) >= num_samples:
                break
            video_np = load_video_frames(video_path, image_size)
            video_tensor = torch.from_numpy(video_np)
            indices = encode_video_to_indices(vqgan_model, video_tensor, device_torch)
            num_frames, ih, iw = indices.shape
            indices_per_frame = ih * iw
            seq_len = sequence_length * indices_per_frame
            num_segments = num_frames // sequence_length
            processed_frames = num_segments * sequence_length

            # 重构视频（仅处理完整段对应的帧）
            indices_recon = indices[:processed_frames].to(device_torch)
            with torch.no_grad():
                quant_b = inner.quantize.embed_code(indices_recon)
                dec = inner.decode(quant_b)
                dec = unnorm(dec)
                dec = dec.clamp(0, 1)
                recon_np = (dec * 255).cpu().numpy().astype(np.uint8)
                recon_np = rearrange(recon_np, "d c h w -> d h w c")

            video_name = Path(video_path).stem
            recon_path = os.path.join(recon_output_dir, f"{video_name}_recon.mp4")
            orig_crop_path = os.path.join(original_output_dir, f"{video_name}_original.mp4")

            def _write_mp4(arr: np.ndarray, out_path: str):
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                tensor = torch.from_numpy(arr).float() / 255.0
                tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0)
                try:
                    from torchvision.io import write_video
                    t_np = (tensor[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                    write_video(out_path, torch.from_numpy(t_np), fps=fps)
                except Exception:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    h, w = arr.shape[1], arr.shape[2]
                    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                    for f in arr:
                        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                    out.release()

            _write_mp4(recon_np, recon_path)
            _write_mp4(video_np[:processed_frames], orig_crop_path)

            for i in range(num_segments):
                start = i * sequence_length
                end = start + sequence_length
                seg = indices[start:end].flatten().cpu().numpy().astype(np.int64)
                all_samples.append(seg)
                val_meta.append({
                    "original_path": os.path.abspath(orig_crop_path),
                    "recon_path": os.path.abspath(recon_path),
                    "num_indices": len(seg),
                    "segment_idx": i,
                    "video_path": video_path,
                })
            del video_np, video_tensor, indices, indices_recon, recon_np
            if device_torch.type == "cuda":
                torch.cuda.empty_cache()
            if num_samples is not None and len(all_samples) >= num_samples:
                break
        except Exception as e:
            logger.warning("验证视频处理失败 %s: %s", video_path, e)
            if device_torch.type == "cuda":
                torch.cuda.empty_cache()
            continue

    if not all_samples:
        raise ValueError("未生成任何验证样本")
    val_data = np.array(all_samples, dtype=np.int64)
    del vqgan_model
    if device_torch.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("验证集: %d 样本, 每样本长度 %d, meta %d 条", len(val_data), val_data.shape[1], len(val_meta))
    return val_data, val_meta


def data_injection(
    data_dir: str,
    vq_config: Optional[str] = None,
    vq_ckpt: Optional[str] = None,
    sequence_length: int = 8,
    image_size: int = 256,
    device: str = "cuda:0",
    num_samples: Optional[int] = None,
    seq_len: Optional[int] = None,
    max_size: int = 100000,
) -> np.ndarray:
    """从视频目录生成训练用 index 数组 [N, seq_len]。"""
    if vq_config is None:
        vq_config = os.getenv("VQ_CONFIG", "")
    if vq_ckpt is None:
        vq_ckpt = os.getenv("VQ_CKPT", "")
    if not vq_config or not vq_ckpt:
        raise ValueError("请提供 vq_config 与 vq_ckpt 或设置 VQ_CONFIG / VQ_CKPT")
    data = process_videos_to_indices(
        video_dir=data_dir,
        vq_config=vq_config,
        vq_ckpt=vq_ckpt,
        sequence_length=sequence_length,
        image_size=image_size,
        device=device,
        num_samples=num_samples,
        max_size=max_size,
    )
    if seq_len is not None and data.shape[1] != seq_len:
        logger.warning("数据长度 %d 与 seq_len %d 不一致", data.shape[1], seq_len)
    return data


def get_data_injection(
    data_dir: str,
    seq_len: Optional[int] = None,
    vq_config: Optional[str] = None,
    vq_ckpt: Optional[str] = None,
    sequence_length: int = 8,
    image_size: int = 256,
    device: str = "cuda:0",
    num_samples: Optional[int] = None,
    max_size: int = 100000,
) -> np.ndarray:
    """便捷接口：返回训练 index 数组。"""
    return data_injection(
        data_dir=data_dir,
        vq_config=vq_config,
        vq_ckpt=vq_ckpt,
        sequence_length=sequence_length,
        image_size=image_size,
        device=device,
        num_samples=num_samples,
        seq_len=seq_len,
        max_size=max_size,
    )
