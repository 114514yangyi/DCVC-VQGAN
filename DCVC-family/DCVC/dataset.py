import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2  # 用于读取视频文件


class VideoGOPDataset(Dataset):
    """
    基于 MP4 视频文件的 GOP 数据集
    支持直接从视频文件中读取指定 GOP 大小的帧序列
    参考 video_utils.py 中的 VideoDataset 实现
    """
    def __init__(self, video_dir, gop_size=7, transform=None, crop_size=256, 
                 random_shuffle=True, seed=None, max_frames_to_check=1000):
        """
        参数:
            video_dir: 视频文件目录路径（包含 .mp4 文件）
            gop_size: GOP 大小（帧数）
            transform: 图像变换（PIL Image -> Tensor）
            crop_size: 随机裁剪大小
            random_shuffle: 是否随机选择起始帧
            seed: 随机种子
            max_frames_to_check: 检查视频总帧数时的最大限制（避免大视频卡住）
        """
        super().__init__()
        self.video_dir = video_dir
        self.gop_size = gop_size
        self.transform = transform
        self.crop_size = crop_size
        self.random_shuffle = random_shuffle
        self.max_frames_to_check = max_frames_to_check
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 扫描目录，找到所有 .mp4 文件
        self.video_files = []
        if os.path.isdir(video_dir):
            for filename in sorted(os.listdir(video_dir)):
                if filename.lower().endswith('.mp4'):
                    video_path = os.path.join(video_dir, filename)
                    self.video_files.append(video_path)
        
        if len(self.video_files) == 0:
            raise ValueError(f"No MP4 files found in {video_dir}")
        
        print(f"Found {len(self.video_files)} video files in {video_dir}")
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        
        try:
            # 获取视频总帧数（限制检查范围以避免大视频卡住）
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                # 如果无法获取总帧数，尝试读取几帧来估算
                frame_count = 0
                while frame_count < self.max_frames_to_check:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                total_frames = frame_count
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开头
            
            # 检查视频帧数是否足够
            if total_frames < self.gop_size:
                cap.release()
                raise ValueError(f"视频 {video_path} 的帧数 ({total_frames}) 少于 GOP 大小 ({self.gop_size})")
            
            # 如果视频帧数足够，随机选择起始位置
            if total_frames > self.gop_size and self.random_shuffle:
                start_frame = random.randint(0, max(0, total_frames - self.gop_size))
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # 读取帧
            frames = []
            frames_read = 0
            
            while frames_read < self.gop_size:
                ret, frame = cap.read()
                if not ret:
                    # 如果视频结束但帧数不够，报错
                    cap.release()
                    raise RuntimeError(f"视频 {video_path} 在读取第 {frames_read + 1} 帧时失败，已读取 {len(frames)} 帧，需要 {self.gop_size} 帧")
                
                # 将 BGR 转换为 RGB（OpenCV 默认 BGR）
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    cap.release()
                    raise RuntimeError(f"视频 {video_path} 第 {frames_read + 1} 帧颜色转换失败: {e}")
                
                frames.append(frame)
                frames_read += 1
            
            cap.release()
            del cap  # 立即释放资源
            
            # 转换为 PIL Image 列表
            pil_frames = []
            for i, frame in enumerate(frames):
                try:
                    pil_frame = Image.fromarray(frame)
                    pil_frames.append(pil_frame)
                except Exception as e:
                    raise RuntimeError(f"视频 {video_path} 第 {i + 1} 帧转换为 PIL Image 失败: {e}")
            
            del frames  # 释放内存
            
            # 应用随机裁剪（如果指定）
            if self.crop_size:
                width, height = pil_frames[0].size
                if width < self.crop_size or height < self.crop_size:
                    raise ValueError(f"视频 {video_path} 的分辨率 ({width}x{height}) 小于裁剪大小 ({self.crop_size}x{self.crop_size})")
                
                x = random.randint(0, width - self.crop_size)
                y = random.randint(0, height - self.crop_size)
                pil_frames = [img.crop((x, y, x + self.crop_size, y + self.crop_size)) 
                             for img in pil_frames]
            
            # 应用 transform（转换为 Tensor）
            if self.transform:
                tensor_frames = []
                for i, img in enumerate(pil_frames):
                    try:
                        tensor_frames.append(self.transform(img))
                    except Exception as e:
                        raise RuntimeError(f"视频 {video_path} 第 {i + 1} 帧应用 transform 失败: {e}")
            else:
                # 默认：转换为 Tensor
                to_tensor = transforms.ToTensor()
                tensor_frames = []
                for i, img in enumerate(pil_frames):
                    try:
                        tensor_frames.append(to_tensor(img))
                    except Exception as e:
                        raise RuntimeError(f"视频 {video_path} 第 {i + 1} 帧转换为 Tensor 失败: {e}")
            
            # 返回形状为 (gop_size, C, H, W) 的 Tensor
            return torch.stack(tensor_frames)
            
        except (RuntimeError, ValueError) as e:
            # 重新抛出这些异常
            raise
        except Exception as e:
            if cap.isOpened():
                cap.release()
            raise RuntimeError(f"读取视频 {video_path} 时发生未知错误: {e}") from e


class VideoValidationDataset(Dataset):
    """
    用于验证的 MP4 视频数据集
    读取完整视频序列用于评估
    """
    def __init__(self, video_dir, transform=None, max_frames=None):
        """
        参数:
            video_dir: 视频文件目录路径
            transform: 图像变换
            max_frames: 最大读取帧数（None 表示读取全部）
        """
        super().__init__()
        self.video_dir = video_dir
        self.transform = transform
        self.max_frames = max_frames
        self.video_files = []
        
        # 扫描目录，找到所有 .mp4 文件
        if os.path.isdir(video_dir):
            for filename in sorted(os.listdir(video_dir)):
                if filename.lower().endswith('.mp4'):
                    video_path = os.path.join(video_dir, filename)
                    self.video_files.append(video_path)
        
        if len(self.video_files) == 0:
            raise ValueError(f"No MP4 files found in {video_dir}")
        
        print(f"Found {len(self.video_files)} validation video files in {video_dir}")
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"无法打开验证视频文件: {video_path}")
        
        try:
            frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 将 BGR 转换为 RGB
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    cap.release()
                    raise RuntimeError(f"验证视频 {video_path} 第 {frame_count + 1} 帧颜色转换失败: {e}")
                
                # 转换为 PIL Image
                try:
                    pil_frame = Image.fromarray(frame)
                except Exception as e:
                    cap.release()
                    raise RuntimeError(f"验证视频 {video_path} 第 {frame_count + 1} 帧转换为 PIL Image 失败: {e}")
                
                # 应用 transform
                try:
                    if self.transform:
                        tensor_frame = self.transform(pil_frame)
                    else:
                        to_tensor = transforms.ToTensor()
                        tensor_frame = to_tensor(pil_frame)
                except Exception as e:
                    cap.release()
                    raise RuntimeError(f"验证视频 {video_path} 第 {frame_count + 1} 帧应用 transform 失败: {e}")
                
                frames.append(tensor_frame)
                frame_count += 1
                
                # 如果达到最大帧数，停止读取
                if self.max_frames and frame_count >= self.max_frames:
                    break
            
            cap.release()
            del cap
            
            if len(frames) == 0:
                raise RuntimeError(f"验证视频 {video_path} 没有读取到任何帧")
            
            return {
                'frames': torch.stack(frames),
                'name': video_name,
                'num_frames': len(frames)
            }
            
        except (RuntimeError, ValueError) as e:
            # 重新抛出这些异常
            raise
        except Exception as e:
            if cap.isOpened():
                cap.release()
            raise RuntimeError(f"读取验证视频 {video_path} 时发生未知错误: {e}") from e
