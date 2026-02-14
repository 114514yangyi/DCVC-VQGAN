import os
import sys
import json
import argparse
import math
import heapq
from collections import Counter
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import torch
from torchvision.transforms import Normalize
from torchvision.io import write_video


# 将项目根目录加入 sys.path，便于直接运行该脚本
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.vq_vae.vq_vae import VqVae2 as VqVae  # noqa: E402


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_SIZE = (256, 256)  # 训练时 video_pipe 默认下采样到 256x256


def build_model_from_checkpoint(checkpoint_path: str,
                                config_path: Optional[str],
                                device: str = "cuda") -> VqVae:
    """
    加载用于压缩/解压的 VqVae2 模型。

    关键点：严格复用训练时的 model_args 配置（来自项目根目录的 param.json），
    避免靠 state_dict 反推结构导致的 shape / groups 不匹配问题。
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["state_dict"]

    # 从配置文件读取与训练一致的 model_args
    if config_path is None:
        config_path = os.path.join(_project_root, "param.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件 {config_path}，请确认配置中包含 model_args")
    with open(config_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    model_args = params["model_args"]

    model = VqVae(**model_args).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ==================== Huffman 编码工具 ==================== #

class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    # heapq 需要可比较对象，这里按频率比较
    def __lt__(self, other: "HuffmanNode"):
        return self.freq < other.freq


def build_huffman_tree(freqs: Dict[int, int]) -> HuffmanNode:
    heap = [HuffmanNode(symbol=s, freq=f) for s, f in freqs.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        # 只有一个 symbol 的退化情况
        node = heapq.heappop(heap)
        return HuffmanNode(left=node, right=None, freq=node.freq)
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        parent = HuffmanNode(freq=n1.freq + n2.freq, left=n1, right=n2)
        heapq.heappush(heap, parent)
    return heap[0]


def build_codebook(node: HuffmanNode, prefix: str = "", codebook: Dict[int, str] = None) -> Dict[int, str]:
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix or "0"
        return codebook
    if node.left is not None:
        build_codebook(node.left, prefix + "0", codebook)
    if node.right is not None:
        build_codebook(node.right, prefix + "1", codebook)
    return codebook


def huffman_encode(indices: torch.Tensor) -> Tuple[bytes, Dict[int, str], int]:
    """对一维整型 tensor 做 Huffman 编码，返回 (字节流, 码表, 有效比特数)。"""
    flat = indices.reshape(-1).tolist()
    freqs = Counter(flat)
    root = build_huffman_tree(freqs)
    codebook = build_codebook(root)
    bit_str = "".join(codebook[v] for v in flat)
    # 记录有效比特长度，用于解码时裁剪
    bit_len = len(bit_str)
    # 补齐到 8 的倍数
    pad = (8 - bit_len % 8) % 8
    bit_str_padded = bit_str + "0" * pad
    # 转成 bytes
    b = int(bit_str_padded, 2).to_bytes(len(bit_str_padded) // 8, byteorder="big")
    return b, codebook, bit_len


def huffman_decode(b: bytes, codebook: Dict[int, str], bit_len: int, shape: Tuple[int, ...]) -> torch.Tensor:
    """根据码表与比特长度解码为指定形状的 long tensor。"""
    # 还原 bit 字符串
    total_bits = len(b) * 8
    bit_str = bin(int.from_bytes(b, byteorder="big"))[2:].zfill(total_bits)
    bit_str = bit_str[:bit_len]

    # 反向码表：bit串 -> 符号
    inv = {v: k for k, v in codebook.items()}
    decoded = []
    cur = ""
    for ch in bit_str:
        cur += ch
        if cur in inv:
            decoded.append(inv[cur])
            cur = ""
    return torch.tensor(decoded, dtype=torch.long).reshape(*shape)


def load_video_frames(path: str, device: str) -> Tuple[torch.Tensor, float]:
    """
    读取视频并返回 (T, C, H, W) 的 float32 张量以及 fps。
    处理流程尽量与训练时的 VideoDataset 保持一致：
    - 使用 OpenCV 读取（BGR 通道顺序）
    - 下采样到 TARGET_SIZE
    - 保持帧顺序，不做随机采样
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 调整分辨率到训练时使用的尺寸
        if frame.shape[:2] != TARGET_SIZE:
            frame = cv2.resize(
                frame,
                (TARGET_SIZE[1], TARGET_SIZE[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"视频中未读取到任何帧: {path}")

    # (T, H, W, C) uint8 BGR
    video_np = np.stack(frames, axis=0).astype(np.uint8)
    # 转为 tensor，并标准化到 [0,1]
    video = torch.from_numpy(video_np).float() / 255.0
    # (T, C, H, W)
    video = video.permute(0, 3, 1, 2)
    return video.to(device), float(fps)


# ==================== 视频压缩 / 解压 ==================== #


def save_video_frames(path: str, frames: torch.Tensor, fps: float):
    """将 (T, C, H, W) 的张量写回视频。"""
    # 训练和压缩过程中我们一直在使用 OpenCV 的 BGR 通道约定，
    # 而 torchvision.write_video 期望的是 RGB。
    # 因此在输出前需要做一次 BGR -> RGB 的通道翻转。
    frames = frames.clamp(0.0, 1.0)
    # 当前 frames 通道顺序等价于 BGR
    frames = frames[:, [2, 1, 0], :, :]  # 转为真正的 RGB
    frames = (frames * 255.0).to(torch.uint8)
    frames = frames.permute(0, 2, 3, 1).cpu()  # (T, H, W, C) RGB
    write_video(path, frames, fps=fps)


def _resolve_device(device_str: str, gpu_index: Optional[int]) -> str:
    """根据字符串和 GPU 序号解析最终 device，例如 cuda:0 / cpu。"""
    if device_str == "auto":
        if torch.cuda.is_available():
            return f"cuda:{gpu_index}" if gpu_index is not None else "cuda"
        return "cpu"
    if device_str == "cuda":
        return f"cuda:{gpu_index}" if gpu_index is not None else "cuda"
    # 明确指定 cpu
    return "cpu"


def _resolve_model_paths(checkpoint: Optional[str],
                         pattern_path: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    根据用户输入解析模型权重路径和配置文件路径。
    - 如果提供 pattern_path（一个目录），则在其中查找 param.json 和最新的 .pth/.pt/.pth.tar 文件。
    - 否则使用 checkpoint 路径，并默认使用项目根目录下的 param.json。
    """
    if pattern_path:
        cfg = os.path.join(pattern_path, "param.json")
        if not os.path.exists(cfg):
            raise ValueError(f"在 pattern_path={pattern_path} 下未找到 param.json")
        candidates = []
        for root, _, files in os.walk(pattern_path):
            for fn in files:
                if fn.endswith((".pth", ".pt", ".pth.tar")):
                    candidates.append(os.path.join(root, fn))
        if not candidates:
            raise ValueError(f"在 pattern_path={pattern_path} 中未找到任何 .pth/.pt/.pth.tar 模型文件")
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        ckpt = candidates[0]
        cfg_path = cfg
    else:
        if checkpoint is None:
            raise ValueError("必须提供 --checkpoint 或 --pattern_path 之一")
        ckpt = checkpoint
        cfg_path = None  # 使用默认的项目根目录 param.json
    return ckpt, cfg_path


def compress_video(args: argparse.Namespace):
    device = _resolve_device(args.device, getattr(args, "gpu_index", None))

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 解析模型路径（支持直接传入 checkpoint 或 pattern_path）
    checkpoint_path, config_path = _resolve_model_paths(
        getattr(args, "checkpoint", None),
        getattr(args, "pattern_path", None),
    )
    # 2) 加载模型
    model = build_model_from_checkpoint(checkpoint_path, config_path, device=device)
    seq_len = model.sequence_length

    # 3) 读取视频
    frames, fps = load_video_frames(args.input, device=device)  # (T, C, H, W)
    T, C, H, W = frames.shape

    # 和训练时保持一致的归一化
    normalize = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    frames_norm = normalize(frames)

    # 3) 拆分为多个 sequence_length 的片段，必要时重复最后一帧进行 padding
    n_seq = math.ceil(T / seq_len)
    total = n_seq * seq_len
    if total > T:
        pad_frames = frames_norm[-1:].repeat(total - T, 1, 1, 1)
        frames_norm = torch.cat([frames_norm, pad_frames], dim=0)
    else:
        pad_frames = None

    # 4) 组织成 (batch * seq_len, C, H, W) 形式；这里 batch = n_seq
    frames_in = frames_norm  # (total, C, H, W)

    with torch.no_grad():
        # 模型内部根据 sequence_length 还原 (b, d, ...) 结构
        vq_codes = []
        for i in range(n_seq):
            start = i * seq_len
            end = (i + 1) * seq_len
            x_chunk = frames_in[start:end]  # (seq_len, C, H, W)
            x_chunk = x_chunk  # 直接视作 batch*seq_len
            codes = model.encode(x_chunk)
            # codes: (1, h', w') 或 (batch, h', w'); 这里 batch==1
            vq_codes.append(codes)
        codes_all = torch.cat(vq_codes, dim=0)  # (n_seq, h', w')

    print(f"codes_all: {codes_all.shape}")
    # 5) Huffman 编码
    bitstream, codebook, bit_len = huffman_encode(codes_all)

    # 6) 保存压缩文件（包括元信息）
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    compressed_path = os.path.join(args.output_dir, base_name + ".vqcmp")
    meta = {
        "bitstream": bitstream,
        "codebook": codebook,
        "bit_len": bit_len,
        "codes_shape": tuple(codes_all.shape),
        "fps": fps,
        "height": H,
        "width": W,
        "num_frames": T,
        "sequence_length": int(seq_len),
        "checkpoint": os.path.abspath(checkpoint_path),
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
    }

    print(f"压缩信息：")
    print(f"  视频总帧数: {T}")
    print(f"  输入分辨率: {H}x{W}")
    print(f"  帧率: {fps}")
    print(f"  sequence_length: {seq_len}")
    print(f"  分段数 n_seq: {n_seq}")
    print(f"  VQ 码本索引形状: {codes_all.shape}")
    print(f"  Huffman 编码后 bitstream 长度（比特）: {bit_len}")
    print(f"  码本大小: {len(codebook)}")
    torch.save(meta, compressed_path)

    # 7) 同时从 codes 解码一份重建视频，方便直接查看效果
    with torch.no_grad():
        recon = model.decode(codes_all.to(device))  # (n_seq*seq_len, C, H, W)
    # 反归一化
    std = torch.tensor(IMAGENET_STD, device=device).view(1, C, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, C, 1, 1)
    recon = recon * std + mean
    recon = recon[:T]  # 去掉 padding 帧

    recon_path = os.path.join(args.output_dir, base_name + "_recon.mp4")
    save_video_frames(recon_path, recon, fps=fps)

    print(f"压缩完成：\n  压缩文件: {compressed_path}\n  重建视频: {recon_path}")


def decompress_video(args: argparse.Namespace):
    device = _resolve_device(args.device, getattr(args, "gpu_index", None))

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 读取压缩文件
    meta = torch.load(args.input, map_location="cpu")
    bitstream = meta["bitstream"]
    codebook = meta["codebook"]
    bit_len = meta["bit_len"]
    codes_shape = tuple(meta["codes_shape"])
    fps = float(meta["fps"])
    H = int(meta["height"])
    W = int(meta["width"])
    T = int(meta["num_frames"])
    seq_len = int(meta["sequence_length"])
    checkpoint_path = meta["checkpoint"]

    # 2) Huffman 解码得到 codebook indices
    codes_all = huffman_decode(bitstream, codebook, bit_len, codes_shape).to(device)

    # 3) 加载模型（可选使用 pattern_path 覆盖默认 checkpoint/config）
    pattern_path = getattr(args, "pattern_path", None)
    checkpoint_path, config_path = _resolve_model_paths(
        checkpoint_path if pattern_path is None else None,
        pattern_path,
    )
    model = build_model_from_checkpoint(checkpoint_path, config_path, device=device)

    # 4) 从 indices 解码视频
    with torch.no_grad():
        recon = model.decode(codes_all)  # (n_seq*seq_len, C, H, W)

    C = recon.shape[1]
    std = torch.tensor(IMAGENET_STD, device=device).view(1, C, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, C, 1, 1)
    recon = recon * std + mean
    recon = recon[:T]  # 去掉 padding 帧

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    out_video = os.path.join(args.output_dir, base_name + "_decoded.mp4")
    save_video_frames(out_video, recon, fps=fps)

    print(f"解压完成：\n  输出视频: {out_video}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VQ-VAE2 视频压缩 / 解压 (Huffman on codebook indices)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 压缩
    p_comp = subparsers.add_parser("compress", help="压缩视频为 Huffman 编码的 codebook 索引")
    p_comp.add_argument("--input", type=str, required=True, help="输入视频路径")
    group_model = p_comp.add_mutually_exclusive_group(required=True)
    group_model.add_argument("--checkpoint", type=str, help="训练好的 VQ-VAE2 checkpoint 路径")
    group_model.add_argument(
        "--pattern_path",
        type=str,
        help="包含模型权重文件（.pth/.pt/.pth.tar）和 param.json 的目录路径"
    )
    p_comp.add_argument("--output-dir", type=str, default=os.path.join(_project_root, "temp"),
                        help="压缩文件与重建视频的输出目录（默认: 项目根目录下 temp）")
    p_comp.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"],
                        help="运行设备 (默认: auto)")
    p_comp.add_argument("--gpu-index", type=int, default=None,
                        help="当 device=cuda/auto 且存在多块 GPU 时，指定使用的 GPU 序号，如 0 或 1")
    p_comp.set_defaults(func=compress_video)

    # 解压
    p_decomp = subparsers.add_parser("decompress", help="从 Huffman 编码文件还原视频")
    p_decomp.add_argument("--input", type=str, required=True, help="压缩文件 (.vqcmp)")
    p_decomp.add_argument("--output-dir", type=str, default=os.path.join(_project_root, "temp"),
                          help="输出视频目录（默认: 项目根目录下 temp）")
    p_decomp.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"],
                          help="运行设备 (默认: auto)")
    p_decomp.add_argument("--gpu-index", type=int, default=None,
                          help="当 device=cuda/auto 且存在多块 GPU 时，指定使用的 GPU 序号，如 0 或 1")
    p_decomp.add_argument(
        "--pattern_path",
        type=str,
        default=None,
        help="可选：包含模型权重和 param.json 的目录，覆盖压缩文件内记录的 checkpoint"
    )
    p_decomp.set_defaults(func=decompress_video)

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


