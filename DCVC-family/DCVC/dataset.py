import os
import re
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2  # 新增：用于读取视频文件

class VimeoGOPDataset(Dataset):
    def __init__(self, root_dir, septuplet_list_path, gop_size=7, transform=None, crop_size=256):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop_size
        self.gop_size = gop_size
        self.total_frames = 7  # Vimeo dataset has 7 frames
        self.septuplet_list = []
        with open(septuplet_list_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.septuplet_list.append(line.strip())
    
    def __len__(self):
        return len(self.septuplet_list)
    
    def __getitem__(self, idx):
        septuplet_path = os.path.join(self.root_dir, self.septuplet_list[idx])
        
        # Calculate random starting frame
        max_start_frame = self.total_frames - self.gop_size + 1
        start_frame = random.randint(1, max_start_frame)
        
        frames = []
        for i in range(start_frame, start_frame + self.gop_size):
            try:
                img_path = os.path.join(septuplet_path, f'im{i}.png')
                image = Image.open(img_path).convert('RGB')
                frames.append(image)
            except FileNotFoundError:
                return self.__getitem__((idx + 1) % len(self))
        
        if self.crop_size:
            width, height = frames[0].size
            if width >= self.crop_size and height >= self.crop_size:
                x = random.randint(0, width - self.crop_size)
                y = random.randint(0, height - self.crop_size)
                frames = [img.crop((x, y, x + self.crop_size, y + self.crop_size)) for img in frames]
        
        if self.transform:
            frames = [self.transform(img) for img in frames]
        
        return torch.stack(frames)

class BVI_AOM_Dataset(Dataset):
    """
    BVI-AOM Dataset - RGB version
    Reads RGB image sequences (PNG/JPG) from directories
    Expected structure: root_dir/sequence_name/frame_*.png
    """
    def __init__(self, root_dir, gop_size=32, crop_size=256, max_sequences=None, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.gop_size = gop_size
        self.transform = transform
        self.sequences = []

        # Find all sequence directories
        for seq_name in sorted(os.listdir(root_dir)):
            seq_path = os.path.join(root_dir, seq_name)
            if os.path.isdir(seq_path):
                # Get all image frames in the sequence
                frame_files = sorted(glob.glob(os.path.join(seq_path, "*.png")))
                if not frame_files:
                    frame_files = sorted(glob.glob(os.path.join(seq_path, "*.jpg")))

                # BVI-AOM typically has 64 frames, but accept sequences with at least gop_size frames
                if len(frame_files) >= self.gop_size:
                    self.sequences.append({
                        'name': seq_name,
                        'path': seq_path,
                        'frames': frame_files[:64]  # BVI-AOM has max 64 frames
                    })

                    if max_sequences and len(self.sequences) >= max_sequences:
                        break

        print(f"Found {len(self.sequences)} BVI-AOM sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        try:
            frames = []

            # Randomly select a segment of gop_size frames
            if len(sequence['frames']) > self.gop_size:
                start_idx = random.randint(0, len(sequence['frames']) - self.gop_size)
                selected_frames = sequence['frames'][start_idx:start_idx + self.gop_size]
            else:
                selected_frames = sequence['frames'][:self.gop_size]

            # Read frames
            for frame_path in selected_frames:
                image = Image.open(frame_path).convert('RGB')
                frames.append(image)

            # Apply cropping
            if self.crop_size:
                width, height = frames[0].size
                if width >= self.crop_size and height >= self.crop_size:
                    x = random.randint(0, width - self.crop_size)
                    y = random.randint(0, height - self.crop_size)
                    frames = [img.crop((x, y, x + self.crop_size, y + self.crop_size)) for img in frames]
                else:
                    raise ValueError(
                        f"Frame size {width}x{height} is smaller than crop size {self.crop_size}x{self.crop_size}"
                    )

            # Apply transform if provided
            if self.transform:
                frames = [self.transform(img) for img in frames]
            else:
                # Default: convert to tensor and normalize to [0, 1]
                import torchvision.transforms as transforms
                to_tensor = transforms.ToTensor()
                frames = [to_tensor(img) for img in frames]

            return torch.stack(frames)

        except Exception as e:
            print(f"Error reading sequence {sequence['name']}: {e}")
            return self.__getitem__((idx + 1) % len(self))


class HEVCB_Dataset(Dataset):
    def __init__(self, root_dir, transform=transforms.ToTensor(), max_frames=96):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.max_frames = max_frames
        self.sequences = []
        
        # Find all sequence directories
        for seq_dir in sorted(os.listdir(root_dir)):
            seq_path = os.path.join(root_dir, seq_dir)
            if os.path.isdir(seq_path):
                # Get all PNG frames in the sequence
                frame_files = sorted(glob.glob(os.path.join(seq_path, "*.png")))
                if len(frame_files) >= max_frames:
                    self.sequences.append({
                        'name': seq_dir,
                        'path': seq_path,
                        'frames': frame_files[:max_frames]
                    })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        frames = []
        
        for frame_path in sequence['frames']:
            try:
                image = Image.open(frame_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
            except Exception as e:
                print(f"Error loading frame {frame_path}: {e}")
                return None
                
        return {
            'frames': torch.stack(frames),
            'name': sequence['name'],
            'num_frames': len(frames)
        }


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
