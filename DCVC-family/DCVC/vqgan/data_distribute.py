import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


def list_images(root: Path) -> List[Path]:
    """递归列出目录下的所有图片文件。"""
    files: List[Path] = []
    if not root.exists():
        raise FileNotFoundError(f"源目录不存在: {root}")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return files


def split_indices(n: int, ratio: float) -> Tuple[List[int], List[int]]:
    """根据比例将 0..n-1 划分成两部分，前一部分大小约为 ratio*n。"""
    idx = list(range(n))
    random.shuffle(idx)
    n_first = int(round(n * ratio))
    first = idx[:n_first]
    second = idx[n_first:]
    return first, second


def copy_images(
    all_images: List[Path],
    idx_first: List[int],
    idx_second: List[int],
    dst1: Path,
    dst2: Path,
) -> None:
    """
    按索引将图片拷贝到两个目标目录。
    不会修改/移动原始文件，只进行 copy。
    """
    dst1.mkdir(parents=True, exist_ok=True)
    dst2.mkdir(parents=True, exist_ok=True)

    for dst, indices in ((dst1, idx_first), (dst2, idx_second)):
        for i in indices:
            src = all_images[i]
            # 为了避免重名冲突，将原始相对路径平铺到目标目录下
            rel_name = f"{src.parent.name}__{src.name}"
            out_path = dst / rel_name
            # 若已存在则略过
            if out_path.exists():
                continue
            shutil.copy2(src, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "从两个源目录收集全部图片，打乱后按比例划分到两个新目录中；"
            "原始目录内容不会被修改，只做复制。"
        )
    )
    parser.add_argument("--src1", type=str, required=False, default="/data/huyang/data/train",help="源图片目录1")
    parser.add_argument("--src2", type=str, required=False, default="/data/huyang/data/vaild", help="源图片目录2")
    parser.add_argument("--dst1", type=str, required=False, default="/data/huyang/data/train_distribute", help="目标目录1")
    parser.add_argument("--dst2", type=str, required=False, default="/data/huyang/data/vaild_distribute", help="目标目录2")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.9,
        help="分配到目标目录1的比例 (0~1)，剩余给目标目录2，默认 0.5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="随机种子，保证可复现的划分",
    )

    args = parser.parse_args()

    if not (0.0 < args.ratio < 1.0):
        raise ValueError(f"ratio 必须在 (0,1) 之间，当前为 {args.ratio}")

    random.seed(args.seed)

    src1 = Path(args.src1).expanduser().resolve()
    src2 = Path(args.src2).expanduser().resolve()
    dst1 = Path(args.dst1).expanduser().resolve()
    dst2 = Path(args.dst2).expanduser().resolve()

    imgs1 = list_images(src1)
    imgs2 = list_images(src2)
    all_images = imgs1 + imgs2
    if not all_images:
        raise RuntimeError(f"在 {src1} 和 {src2} 中没有找到任何图片文件")

    idx_first, idx_second = split_indices(len(all_images), args.ratio)
    copy_images(all_images, idx_first, idx_second, dst1, dst2)


if __name__ == "__main__":
    main()

