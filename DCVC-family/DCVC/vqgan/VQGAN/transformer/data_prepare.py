"""
VQVAE2 Transformer 数据准备

- 使用 VQVAE2 将训练/验证视频编码为 codebook index 序列
- VQVAE2 每帧有多个层级（top/bottom 等）的 index，将同一帧各层 index 拼接为一帧的 index 序列
- 训练集：生成 [num_samples, seq_len] 的 index 数组供 Transformer 学习
- 验证集：除 index 外，生成重构视频，并记录 原视频路径、重构视频路径、每视频的 index，用于验证阶段写 CSV
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
import cv2
from einops import rearrange
from torchvision import transforms
from torchvision.io import write_video
from tqdm import tqdm

# VQVAE2 项目根
_VQVAE2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _VQVAE2_ROOT not in sys.path:
    sys.path.insert(0, _VQVAE2_ROOT)

from models.model_adapter import create_model

logger = logging.getLogger(__name__)

# 视频扩展名
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def list_videos(video_dir: str) -> List[str]:
    """列出目录下所有视频文件路径（排序）"""
    paths = []
    for name in os.listdir(video_dir):
        if not any(name.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
            continue
        path = os.path.join(video_dir, name)
        if os.path.isfile(path):
            paths.append(path)
    paths.sort()
    return paths


def load_vqvae2_model(vq_config: str, vq_ckpt: str, device: str):
    """加载 VQVAE2 模型（适配器）及配置"""
    with open(vq_config, "r") as f:
        config = json.load(f)
    data_args = config.get("data", {})
    model_args = config.get("model_args", {})
    if "sequence_length" in data_args and "sequence_length" not in model_args:
        model_args = {**model_args, "sequence_length": data_args["sequence_length"]}
    model = create_model(model_args=model_args, config_path=vq_config)
    model = model.to(device)
    ckpt = torch.load(vq_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, config


def encode_video_to_indices_vqvae2(
    model,
    video: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
) -> List[torch.Tensor]:
    """
    使用 VQVAE2 将视频编码为各层 codebook indices。
    VQVAE2 的 forward 返回 id_outputs: list of (B, H_l, W_l)，从 top 到 bottom。

    Args:
        model: VQVAE2 适配器（内部需能拿到 id_outputs，这里用 _get_model().forward）
        video: (num_frames, height, width, channels), uint8
        device: 设备
        batch_size: 批大小

    Returns:
        id_outputs: list of (num_frames, H_l, W_l)，每层一维
    """
    num_frames, h, w, c = video.shape
    video = video.float().to(device)
    video = rearrange(video, "d h w c -> d c h w")
    video = video / 255.0
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    video = normalize(video)

    inner = model._get_model()
    all_id_outputs = None

    with torch.no_grad():
        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            batch_video = video[i:end_idx]
            # inner.forward 返回 (decoder_outputs[-1], diffs, enc_out, dec_out, id_outputs)
            _, _, _, _, id_outputs = inner(batch_video)
            if all_id_outputs is None:
                all_id_outputs = [x.cpu() for x in id_outputs]
            else:
                for l, x in enumerate(id_outputs):
                    all_id_outputs[l] = torch.cat([all_id_outputs[l], x.cpu()], dim=0)
    return all_id_outputs


def id_outputs_to_flat_per_frame(id_outputs: List[torch.Tensor]) -> torch.Tensor:
    """
    将 VQVAE2 的 id_outputs（每层 (T, H_l, W_l)）按「每帧」拼接为一条长序列：
    对每一帧，按 level 顺序拼接 [top_flatten, ..., bottom_flatten]，再按帧顺序拼接。

    Returns:
        flat: (T, total_per_frame) 或 (T * total_per_frame,) — 这里返回 (T, total_per_frame) 便于按帧切段
    """
    # id_outputs[l]: (T, H_l, W_l)
    T = id_outputs[0].shape[0]
    per_frame = []
    for t in range(T):
        parts = []
        for level_tensor in id_outputs:
            # level_tensor: (T, H, W) -> 取第 t 帧
            parts.append(level_tensor[t].flatten())
        per_frame.append(torch.cat(parts, dim=0))
    # (T, sum_l H_l*W_l)
    out = torch.stack(per_frame, dim=0)
    return out


def load_video_frames(video_path: str, image_size: int = 256) -> np.ndarray:
    """加载视频所有帧，返回 (num_frames, H, W, C) uint8"""
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
    if len(frames) == 0:
        raise ValueError(f"视频无帧: {video_path}")
    return np.stack(frames, axis=0).astype(np.uint8)


def process_videos_to_indices_vqvae2(
    video_dir: str,
    vq_config: str,
    vq_ckpt: str,
    sequence_length: int,
    image_size: int = 256,
    device: str = "cuda:0",
    num_samples: Optional[int] = None,
    max_size: int = 10000,
) -> np.ndarray:
    """
    从视频目录生成 Transformer 训练用的 index 数组。
    每段为 sequence_length 帧，每帧的 index 为各层 (top...bottom) 展平后拼接。

    Returns:
        data: (num_samples, seq_len), int64. seq_len = sequence_length * indices_per_frame.
    """
    model, config = load_vqvae2_model(vq_config, vq_ckpt, device)
    device_t = torch.device(device)
    video_files = list_videos(video_dir)
    if len(video_files) == 0:
        raise ValueError(f"目录下无视频: {video_dir}")

    all_samples = []
    for video_path in tqdm(video_files, desc="VQVAE2 encode train"):
        if len(all_samples) >= max_size:
            break
        try:
            video_np = load_video_frames(video_path, image_size)
            video_t = torch.from_numpy(video_np)
            id_outputs = encode_video_to_indices_vqvae2(model, video_t, device_t)
            # (T, indices_per_frame)
            per_frame = id_outputs_to_flat_per_frame(id_outputs)
            T, indices_per_frame = per_frame.shape
            num_segments = T // sequence_length
            for i in range(num_segments):
                start = i * sequence_length
                end = start + sequence_length
                seg = per_frame[start:end]  # (sequence_length, indices_per_frame)
                seg_flat = seg.flatten().numpy().astype(np.int64)
                all_samples.append(seg_flat)
                if num_samples is not None and len(all_samples) >= num_samples:
                    break
            del video_np, video_t, id_outputs, per_frame
            if device_t.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"处理视频失败 {video_path}: {e}")
            continue
        if num_samples is not None and len(all_samples) >= num_samples:
            break

    if len(all_samples) == 0:
        raise ValueError("未生成任何训练样本")
    data = np.array(all_samples, dtype=np.int64)
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    logger.info(f"生成训练样本: {data.shape[0]}, seq_len={data.shape[1]}")
    return data


def prepare_validation_with_recon(
    val_video_dir: str,
    vq_config: str,
    vq_ckpt: str,
    sequence_length: int,
    image_size: int,
    device: str,
    recon_output_dir: str,
    original_output_dir: str,
    max_val_videos: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    为验证集生成：
    1) Transformer 用的 index 数组（与训练相同格式，可来自多段）；
    2) 验证元信息列表，每项包含：original_path, recon_path, indices（该视频所有段的 index 或整视频 flat），
       用于评估时按视频写 CSV（orig, recon, size）。

    会对每个验证视频：编码 -> 解码得到重构视频 -> 保存到 recon_output_dir；
    裁剪后的原视频保存到 original_output_dir（与 project recon_judge 一致）。
    """
    model, config = load_vqvae2_model(vq_config, vq_ckpt, device)
    device_t = torch.device(device)
    inner = model._get_model()
    video_files = list_videos(val_video_dir)
    if max_val_videos is not None:
        video_files = video_files[: max_val_videos]

    # 反归一化：训练时用 (x-0.5)/0.5，故解码后 (out+1)/2 -> [0,1]
    unnorm = transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])
    os.makedirs(recon_output_dir, exist_ok=True)
    os.makedirs(original_output_dir, exist_ok=True)

    all_val_indices = []  # 用于 Transformer 验证的 index 列表（按段）
    val_meta_list = []    # 每视频一条：original_path, recon_path, indices_per_video (list of segments or one array)

    for video_path in tqdm(video_files, desc="VQVAE2 val encode+recon"):
        try:
            video_np = load_video_frames(video_path, image_size)
            video_t = torch.from_numpy(video_np).float().to(device_t)
            video_t = rearrange(video_t, "d h w c -> d c h w") / 255.0
            video_t = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(video_t)
            T = video_t.shape[0]

            with torch.no_grad():
                _, _, _, _, id_outputs = inner(video_t)
            per_frame = id_outputs_to_flat_per_frame([x.cpu() for x in id_outputs])
            indices_per_frame = per_frame.shape[1]
            num_segments = T // sequence_length
            segments_for_transformer = []
            for i in range(num_segments):
                start = i * sequence_length
                end = start + sequence_length
                seg = per_frame[start:end].flatten().numpy().astype(np.int64)
                segments_for_transformer.append(seg)
                all_val_indices.append(seg)

            # 解码重建：decode_codes(*cs) 每层 (B, H_l, W_l)，按帧解码
            with torch.no_grad():
                recon_list = []
                for t in range(T):
                    level_codes = [id_outputs[l][t : t + 1] for l in range(len(id_outputs))]
                    out = inner.decode_codes(*level_codes)
                    out = unnorm(out.clamp(-1.0, 1.0))
                    out = out.clamp(0, 1)
                    recon_list.append(out)
                recon = torch.cat(recon_list, dim=0)
            recon_np = (recon.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

            # 保存重构视频
            video_name = Path(video_path).stem
            recon_path = os.path.join(recon_output_dir, f"{video_name}_recon.mp4")
            processed_frames = num_segments * sequence_length
            recon_actual = recon_np[:processed_frames]
            fps = 30.0
            # write_video 需要 (T, C, H, W), 0-255
            recon_t = torch.from_numpy(recon_actual).permute(0, 3, 1, 2).byte().cpu()
            write_video(recon_path, recon_t, fps=fps)

            # 裁剪原视频并保存
            orig_path = os.path.join(original_output_dir, f"{video_name}_original.mp4")
            orig_actual = video_np[:processed_frames]
            orig_t = torch.from_numpy(orig_actual).permute(0, 3, 1, 2).byte().cpu()
            write_video(orig_path, orig_t, fps=fps)

            val_meta_list.append({
                "original_path": orig_path,
                "recon_path": recon_path,
                "indices_segments": segments_for_transformer,
                "video_path": video_path,
            })
        except Exception as e:
            logger.warning(f"验证视频处理失败 {video_path}: {e}")
            continue

    if len(all_val_indices) == 0:
        raise ValueError("未生成任何验证样本")
    val_data = np.array(all_val_indices, dtype=np.int64)
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return val_data, val_meta_list


def get_data_injection(
    data_dir: str,
    seq_len: Optional[int] = None,
    vq_config: Optional[str] = None,
    vq_ckpt: Optional[str] = None,
    sequence_length: int = 8,
    image_size: int = 256,
    device: str = "cuda:0",
    num_samples: Optional[int] = None,
    max_size: int = 10000,
) -> np.ndarray:
    """训练集：返回 (N, seq_len) 的 index 数组。vq_config/vq_ckpt 若为 None 则从环境变量或默认路径读取。"""
    if vq_config is None:
        vq_config = os.getenv("VQ_CONFIG", os.path.join(_VQVAE2_ROOT, "config.json"))
    if vq_ckpt is None:
        vq_ckpt = os.getenv("VQ_CKPT", "")
    if not vq_ckpt or not os.path.isfile(vq_ckpt):
        raise FileNotFoundError(f"请设置 VQ_CKPT 或传入 vq_ckpt: {vq_ckpt}")
    return process_videos_to_indices_vqvae2(
        video_dir=data_dir,
        vq_config=vq_config,
        vq_ckpt=vq_ckpt,
        sequence_length=sequence_length,
        image_size=image_size,
        device=device,
        num_samples=num_samples,
        max_size=max_size,
    )
