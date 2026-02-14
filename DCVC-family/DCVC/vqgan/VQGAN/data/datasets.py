"""
VQGAN - 最小数据加载

支持两种模式：
- use_images=false: 读取视频文件（mp4 等），每次采样 sequence_length 帧
- use_images=true : 读取图片文件（jpg/png），随机采样 sequence_length 张组成序列

输出统一为 uint8 tensor: (B, D, H, W, C)
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def list_files(root: str, exts: Sequence[str]) -> List[str]:
    files: List[str] = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isfile(path) and any(name.lower().endswith(e) for e in exts):
            files.append(path)
    files.sort()
    return files


class ImageSequenceDataset(Dataset):
    def __init__(self, image_files: List[str], sequence_length: int, size: Tuple[int, int], seed: int):
        self.files = image_files
        self.sequence_length = int(sequence_length)
        self.size = size  # (H, W)
        self.rng = random.Random(int(seed))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        frames = []
        for _ in range(self.sequence_length):
            path = self.files[self.rng.randrange(0, len(self.files))]
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape[:2] != self.size:
                    img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
            frames.append(img)
        video = np.stack(frames, axis=0).astype(np.uint8)  # (D,H,W,C)
        return video


class VideoClipDataset(Dataset):
    def __init__(self, video_files: List[str], sequence_length: int, size: Tuple[int, int], seed: int):
        self.files = video_files
        self.sequence_length = int(sequence_length)
        self.size = size
        self.rng = random.Random(int(seed))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > self.sequence_length:
            start = self.rng.randint(0, total - self.sequence_length)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frames = []
        for _ in range(self.sequence_length):
            ok, frame = cap.read()
            if not ok:
                if frames:
                    frame = frames[-1].copy()
                else:
                    frame = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
            if frame.shape[:2] != self.size:
                frame = cv2.resize(frame, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        video = np.stack(frames, axis=0).astype(np.uint8)
        return video


@dataclass(frozen=True)
class DataConfig:
    train_dir: str
    val_dir: str | None
    use_images: bool
    batch_size: int
    num_workers: int
    sequence_length: int
    resolution: int
    seed: int


def build_loaders(cfg: DataConfig):
    size = (cfg.resolution, cfg.resolution)
    if cfg.use_images:
        train_files = list_files(cfg.train_dir, exts=(".jpg", ".jpeg", ".png", ".webp"))
        train_ds = ImageSequenceDataset(train_files, cfg.sequence_length, size=size, seed=cfg.seed)
    else:
        train_files = list_files(cfg.train_dir, exts=(".mp4", ".mov", ".mkv", ".avi", ".webm"))
        train_ds = VideoClipDataset(train_files, cfg.sequence_length, size=size, seed=cfg.seed)

    def collate(batch):
        arr = np.stack(batch, axis=0)  # (B,D,H,W,C) uint8
        return torch.from_numpy(arr)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=False,
        collate_fn=collate,
    )

    val_loader = None
    if cfg.val_dir:
        if cfg.use_images:
            val_files = list_files(cfg.val_dir, exts=(".jpg", ".jpeg", ".png", ".webp"))
            val_ds = ImageSequenceDataset(val_files, cfg.sequence_length, size=size, seed=cfg.seed + 1)
        else:
            val_files = list_files(cfg.val_dir, exts=(".mp4", ".mov", ".mkv", ".avi", ".webm"))
            val_ds = VideoClipDataset(val_files, cfg.sequence_length, size=size, seed=cfg.seed + 1)
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False,
            pin_memory=False,
            collate_fn=collate,
        )

    return train_loader, val_loader, len(train_files)

