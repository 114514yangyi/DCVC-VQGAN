import json
import os
import sys
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import List, Tuple

import cv2

logger = logging.getLogger("data_process")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

# 添加项目根目录到 Python 路径，以便可以直接运行脚本
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class VideoDataProcessor:
    """
    负责:
    1) 读取原始视频目录，中心裁剪为正方形，保存到 clipdata_dir
    2) 将 clipdata_dir 中的视频缩放至 256x256，保存到 processed_dir
    3) 将 processed_dir 中视频按比例划分训练/验证集,将划分后的视频拆分成图像帧，保存到 train 和 vaild 目录
    4) 对 raw/clipdata/processed/train/vaild 做可用性检测（ffprobe + cv2），输出坏文件报告
    """

    def __init__(self, raw_dir: str, clipdata_dir: str, processed_dir: str, val_ratio: float = 0.1, 
                 seed: int = 2025, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe"):
        self.raw_dir = Path(raw_dir)
        self.clipdata_dir = Path(clipdata_dir)
        self.processed_dir = Path(processed_dir)
        self.val_ratio = max(0.0, min(1.0, val_ratio))
        self.seed = seed
        self.ffmpeg = ffmpeg_bin
        self.ffprobe = ffprobe_bin
        self.clipdata_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _run_cmd(self, cmd: List[str]):
        try:
            return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            sys.stderr.write(f"[ffmpeg] 命令失败: {' '.join(cmd)}\n")
            sys.stderr.write(e.stderr.decode(errors="ignore") if e.stderr else "")
            raise

    def _clean_dir(self, path: Path):
        """
        清空目录下所有文件/子目录，但保留目录本身。
        """
        if not path.exists():
            return
        for child in path.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                shutil.rmtree(child, ignore_errors=True)

    def _is_video_ok_ffprobe(self, video_path: Path) -> bool:
        """检查视频能否被 ffprobe 正常解析且时长 > 0。"""
        if not video_path.exists() or video_path.stat().st_size == 0:
            return False
        cmd = [
            self.ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
            dur = float(out)
            return dur > 0
        except Exception:
            return False

    @staticmethod
    def _is_video_ok_cv2(video_path: Path) -> bool:
        """使用 cv2.VideoCapture 检测是否能打开视频。"""
        if not video_path.exists() or video_path.stat().st_size == 0:
            return False
        cap = cv2.VideoCapture(str(video_path))
        ok = cap.isOpened()
        cap.release()
        return ok

    def _scan_videos(self, paths: List[Path], label: str) -> List[Path]:
        """双重检测（ffprobe + cv2），返回坏文件列表并记录日志。"""
        bad = []
        for p in paths:
            if not (self._is_video_ok_ffprobe(p) and self._is_video_ok_cv2(p)):
                bad.append(p)
                # 直接删除不可用视频
                try:
                    p.unlink(missing_ok=True)
                    logger.warning(f"[{label}] 删除不可用视频: {p}")
                except Exception as e:
                    logger.error(f"[{label}] 删除失败 {p}: {e}")
        if not bad:
            logger.info(f"[{label}] 视频检测通过，共 {len(paths)} 个。")
        return bad

    def _get_center_crop_bbox(self, video_path: Path) -> Tuple[int, int, int, int]:
        """
        获取视频的中心正方形裁剪区域 (x, y, w, h)。
        从两边裁剪直到中间部分是正方形。
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        try:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        
        # 计算中心正方形裁剪区域
        if frame_width > frame_height:
            # 宽度大于高度，从左右两边裁剪
            size = frame_height
            x = (frame_width - size) // 2
            y = 0
        else:
            # 高度大于宽度，从上下两边裁剪
            size = frame_width
            x = 0
            y = (frame_height - size) // 2
        
        w = h = size
        return (x, y, w, h)

    def _crop_video_center(self, src: Path, dst: Path) -> bool:
        """
        使用 ffmpeg 将视频中心裁剪为正方形。
        注意：ffmpeg 的 crop 滤镜会对整个视频的所有帧应用相同的裁剪坐标，
        确保整个视频使用一致的裁剪，避免裁剪抖动。
        返回是否成功
        """
        if not src.exists():
            return False
        
        try:
            bbox = self._get_center_crop_bbox(src)
            x, y, w, h = bbox
        except Exception as e:
            logger.error(f"获取裁剪区域失败 {src}: {e}")
            return False
        
        # 使用 ffmpeg 的 crop 滤镜裁剪视频
        # crop=w:h:x:y 会对整个视频的所有帧应用相同的裁剪
        cmd = [
            self.ffmpeg, "-y", "-i", str(src),
            "-vf", f"crop={w}:{h}:{x}:{y}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-an",  # 移除音频
            str(dst)
        ]
        try:
            self._run_cmd(cmd)
            logger.debug(f"裁剪完成: {src} -> {dst} (区域: {x}, {y}, {w}, {h})")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"裁剪视频失败 {src}: {e}")
            return False

    def _scale_video(self, src: Path, dst: Path) -> bool:
        """
        将视频缩放至 256x256。
        返回是否成功
        """
        if not src.exists():
            return False
        
        cmd = [
            self.ffmpeg, "-y", "-i", str(src),
            "-vf", "scale=256:256,format=yuv444p",
            "-an",  # 移除音频
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            str(dst)
        ]
        try:
            self._run_cmd(cmd)
            logger.debug(f"缩放完成: {src} -> {dst}")
            return True
        except subprocess.CalledProcessError:
            # 若因尺度/像素格式报错，尝试退化处理
            fallback_cmd = [
                self.ffmpeg, "-y", "-i", str(src),
                "-vf", "scale=256:256",
                "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                str(dst)
            ]
            try:
                self._run_cmd(fallback_cmd)
                logger.debug(f"缩放完成(降级): {src} -> {dst}")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"缩放视频失败 {src}: {e}")
                return False

    def crop_all(self, valid_videos: List[Path]):
        """裁剪所有视频为中心正方形，保存到 clipdata_dir"""
        logger.info(f"开始裁剪 {len(valid_videos)} 个视频...")
        for i, v in enumerate(valid_videos, 1):
            base = v.stem
            dst = self.clipdata_dir / f"{base}.mp4"
            if self._crop_video_center(v, dst):
                if i % 100 == 0:
                    logger.info(f"已裁剪 {i}/{len(valid_videos)} 个视频")
            else:
                logger.warning(f"裁剪失败: {v}")
        logger.info(f"裁剪完成，共处理 {len(valid_videos)} 个视频")

    def scale_all(self):
        """缩放 clipdata_dir 中的所有视频到 256x256，保存到 processed_dir"""
        cropped_videos = sorted(self.clipdata_dir.glob("*.mp4"))
        if not cropped_videos:
            logger.warning("clipdata_dir 中没有视频文件")
            return
        
        logger.info(f"开始缩放 {len(cropped_videos)} 个视频...")
        for i, v in enumerate(cropped_videos, 1):
            base = v.stem
            dst = self.processed_dir / f"{base}.mp4"
            if self._scale_video(v, dst):
                if i % 100 == 0:
                    logger.info(f"已缩放 {i}/{len(cropped_videos)} 个视频")
            else:
                logger.warning(f"缩放失败: {v}")
        logger.info(f"缩放完成，共处理 {len(cropped_videos)} 个视频")

    def split_train_val(self) -> Tuple[List[str], List[str]]:
        files = sorted(self.processed_dir.glob("*.mp4"))
        files = [str(p) for p in files]
        if not files:
            return [], []
        rng = __import__("random").Random(self.seed)
        rng.shuffle(files)
        val_n = int(len(files) * self.val_ratio)
        val_n = min(max(val_n, 0), len(files))
        val_files = files[:val_n]
        train_files = files[val_n:]
        return train_files, val_files

    def update_param(self, param_path: Path, train_files: List[str], val_files: List[str]):
        with open(param_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        params.setdefault("data_args", {})
        params["data_args"]["training_data_files"] = train_files
        params["data_args"]["validation_data_files"] = val_files
        params["data_args"]["validation_ratio"] = self.val_ratio
        with open(param_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)

    def copy_split_to_dirs(self, train_files: List[str], val_files: List[str]) -> Tuple[List[str], List[str]]:
        """
        将划分结果复制到 ./data/train 与 ./data/vaild 下，返回新的路径列表。
        """
        train_dir = Path("/data/huyang/data/train")
        val_dir = Path("/data/huyang/data/vaild")
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        new_train, new_val = [], []
        for src in train_files:
            dst = train_dir / Path(src).name
            shutil.copy2(src, dst)
            new_train.append(str(dst))
        for src in val_files:
            dst = val_dir / Path(src).name
            shutil.copy2(src, dst)
            new_val.append(str(dst))
        return new_train, new_val

    def run(self, param_path: str = "param.json"):
        report_lines = []

        # 1) 清理输出目录（clipdata/processed/train/vaild），避免旧文件干扰
        for p in [self.clipdata_dir, self.processed_dir, Path("/data/huyang/data/train"), Path("/data/huyang/data/vaild")]:
            self._clean_dir(p)

        # 2) 检查 raw 视频可用性（ffprobe + cv2）
        raw_videos = sorted(self.raw_dir.glob("*.mp4"))
        bad_raw = self._scan_videos(raw_videos, label="raw")
        valid_raw = [v for v in raw_videos if v not in bad_raw]
        report_lines.append(f"RAW: total={len(raw_videos)}, bad={len(bad_raw)}")

        # 3) 裁剪所有视频为中心正方形，保存到 clipdata_dir
        self.crop_all(valid_videos=valid_raw)

        # 4) 检查裁剪后的视频可用性
        cropped_videos = sorted(self.clipdata_dir.glob("*.mp4"))
        bad_cropped = self._scan_videos(cropped_videos, label="clipdata")
        valid_cropped = [v for v in cropped_videos if v not in bad_cropped]
        report_lines.append(f"CLIPDATA: total={len(cropped_videos)}, bad={len(bad_cropped)}")

        # 5) 缩放所有裁剪后的视频到 256x256，保存到 processed_dir
        self.scale_all()

        # 6) 划分并复制
        train_files, val_files = self.split_train_val()
        train_files, val_files = self.copy_split_to_dirs(train_files, val_files)
        self.update_param(Path(param_path), train_files, val_files)

        # 7) 对 processed/train/vaild 做 cv2+ffprobe 检测
        processed_videos = sorted(self.processed_dir.glob("*.mp4"))
        bad_processed = self._scan_videos(processed_videos, label="processed")
        train_videos = [Path(p) for p in train_files]
        val_videos = [Path(p) for p in val_files]
        bad_train = self._scan_videos(train_videos, label="train")
        bad_val = self._scan_videos(val_videos, label="vaild")

        report_lines.extend([
            f"PROCESSED: total={len(processed_videos)}, bad={len(bad_processed)}",
            f"TRAIN: total={len(train_videos)}, bad={len(bad_train)}",
            f"VAILD: total={len(val_videos)}, bad={len(bad_val)}",
        ])

        # 8) 输出报告文件
        report_path = Path(self.processed_dir).parent / "data_check_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")
            if bad_raw:
                f.write("Bad raw videos:\n")
                f.write("\n".join(str(p) for p in bad_raw) + "\n")
            if bad_cropped:
                f.write("Bad cropped videos:\n")
                f.write("\n".join(str(p) for p in bad_cropped) + "\n")
            if bad_processed:
                f.write("Bad processed videos:\n")
                f.write("\n".join(str(p) for p in bad_processed) + "\n")
            if bad_train:
                f.write("Bad train videos:\n")
                f.write("\n".join(str(p) for p in bad_train) + "\n")
            if bad_val:
                f.write("Bad vaild videos:\n")
                f.write("\n".join(str(p) for p in bad_val) + "\n")

        logger.info("数据检查报告已生成: %s", report_path)
        return train_files, val_files


if __name__ == "__main__":
    with open("param.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    dp_cfg = cfg.get("data_process", {})
    raw_dir = dp_cfg.get("raw_video_dir", "/data/huyang/video")
    clipdata_dir = dp_cfg.get("clipdata_dir", "/data/huyang/data/clipdata")
    processed_dir = dp_cfg.get("processed_dir", "/data/huyang/data/processedVideo")
    val_ratio = dp_cfg.get("val_ratio", 0.1)
    seed = cfg.get("data_args", {}).get("seed", 2025)

    processor = VideoDataProcessor(
        raw_dir=raw_dir, 
        clipdata_dir=clipdata_dir,
        processed_dir=processed_dir, 
        val_ratio=val_ratio, 
        seed=seed
    )
    train_list, val_list = processor.run("param.json")
    print(f"处理完成，训练集 {len(train_list)} 个，验证集 {len(val_list)} 个")

