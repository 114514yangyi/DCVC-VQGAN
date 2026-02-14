import datetime
import json
import logging
import os
import random
from typing import List, Optional, Dict
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)



# =============================================================================
# video_utils.py 文件功能详细注释
# =============================================================================
#
# 本文件是视频任务训练管道中的辅助工具模块，主要用于视频数据集的管理、
# GPU 资源自动选择、参数读取、以及日志系统的配置。
#
# 结构和功能解释如下：
#
# 1. 引入标准库与第三方库
#    - os, sys: 用于路径管理、环境变量和系统操作。
#    - random, datetime, logging: 用于日志、随机数生成和时间管理。
#    - typing: 类型提示。
#    - cv2, numpy: 视频解码与数组操作，实现高效处理原始视频帧。
#    - torch, torch.utils.data: 支持 PyTorch 数据加载、Dataset 定义等。
#
# 2. GPU 可用性检测
#    - pynvml（NVIDIA 管理库）尝试导入，用于自动化检测 GPU 空闲显存、选择合适的 GPU 加速设备。
#    - 通过 _PYNVML_AVAILABLE 标志判断 pynvml 库是否可用，决定是否启用自动 GPU 选择功能。
#
# 3. 项目根路径配置
#    - 保证模块能够直接运行时正确 import 项目内其它文件。
#    - 将根目录插入 sys.path（如未存在），避免因模块导入路径问题导致的错误。
#
# 4. 参数与数据路径配置
#    - 读取 param.json，作为训练各阶段的参数配置来源（包括 data_args、train_args、model_args）。
#    - 用 list_videos() 列出指定训练/验证数据集文件夹下所有 mp4 视频，生成训练与验证文件路径。
#
# 5. 数据集文件辅助函数
#    - list_videos(video_folder):
#      列出传入目录下所有以 mp4 结尾的视频文件，绝对路径形式返回，用于构建训练集或验证集。
#
# 6. 日志系统配置
#    - setup_logging(log_dir): 设定日志文件输出格式、日志级别，并支持日志同时输出到文件和控制台（handler 池重复检测）。
#
# 7. 其它
#    - 预留数据集类、自定义 Dataset/DataLoader、视频预处理流水线（如 video_pipe）、
#      以及自动 GPU 分配（如 choose_gpu）等函数/类的实现（未粘贴部分）。
#
# 8. 文件作用总结
#    - 此工具脚本是训练主流程的“配套助手”，为主脚本提供数据、设备、配置和日志等基础能力，
#      提高了训练主循环代码的整洁性与可维护性，支持实验配置文件化与自动化实验管理。
# =============================================================================



def list_videos(video_folder):
    """
    folder structure
    parent_folder
        - video1
        - video2
    :return:
    """

    files = [os.path.join(video_folder, f2) for f2 in os.listdir(video_folder) ]
    return files


# import json
# with open('param.json', 'r') as f:
#     params = json.load(f)
#     print(params)






def setup_logging(log_dir: str) -> logging.Logger:
    """
    设置日志系统，同时输出到控制台和文件
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logger = logging.getLogger('VqVaeTraining')
    logger.setLevel(logging.DEBUG)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"日志系统已初始化，日志文件保存在: {log_file}")
    return logger


def list_gpu_info() -> List[dict]:
    """
    列出可用 GPU 的基本信息与利用率。
    优先使用 pynvml 获取实时利用率，若不可用则返回名称和显存容量。
    """
    if not torch.cuda.is_available():
        return []

    gpu_info_list = []
    if _PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for idx in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_info_list.append({
                    "index": idx,
                    "name": name,
                    "memory_total_GB": mem.total / 1024 ** 3,
                    "memory_used_GB": mem.used / 1024 ** 3,
                    "utilization_gpu": util.gpu,
                    "utilization_mem": util.memory
                })
            pynvml.nvmlShutdown()
            return gpu_info_list
        except Exception:
            pass

    # Fallback: 仅提供名称，无法获取利用率
    device_count = torch.cuda.device_count()
    for idx in range(device_count):
        props = torch.cuda.get_device_properties(idx)
        gpu_info_list.append({
            "index": idx,
            "name": props.name,
            "memory_total_GB": props.total_memory / 1024 ** 3,
            "memory_used_GB": None,
            "utilization_gpu": None,
            "utilization_mem": None
        })
    return gpu_info_list


# def choose_gpu(logger: logging.Logger, preferred: Optional[int] = None) -> str:
#     """
#     根据可用 GPU 列表选择设备。
#     - preferred: 用户指定的 GPU 序号（int），若合法则直接返回。
#     - 否则列出所有 GPU 的利用率并提示输入，默认为 0。
#     返回形如 "cuda:0" 的设备字符串。
#     """
#     gpu_list = list_gpu_info()
#     if not gpu_list:
#         logger.info("未检测到可用 GPU，回退到 CPU")
#         return "cpu"

#     if preferred is not None and 0 <= preferred < len(gpu_list):
#         logger.info(f"使用用户指定的 GPU: cuda:{preferred} ({gpu_list[preferred]['name']})")
#         return f"cuda:{preferred}"

#     logger.info("检测到以下 GPU，可选择使用的 GPU 序号：")
#     for g in gpu_list:
#         logger.info(
#             f"  [{g['index']}] {g['name']} | "
#             f"显存: {g['memory_used_GB']:.2f} / {g['memory_total_GB']:.2f} GB"
#             if g['memory_used_GB'] is not None else
#             f"  [{g['index']}] {g['name']} | 显存: {g['memory_total_GB']:.2f} GB"
#         )
#         if g['utilization_gpu'] is not None:
#             logger.info(f"      利用率: GPU {g['utilization_gpu']}% | 显存 {g['utilization_mem']}%")

#     try:
#         choice = input(f"请选择要使用的 GPU 序号 [0-{len(gpu_list)-1}]，直接回车默认0：").strip()
#         if choice == "":
#             idx = 0
#         else:
#             idx = int(choice)
#         if 0 <= idx < len(gpu_list):
#             logger.info(f"选择 GPU: cuda:{idx} ({gpu_list[idx]['name']})")
#             return f"cuda:{idx}"
#     except Exception:
#         pass

#     logger.info("输入无效，默认使用 cuda:2")
#     return params['train_args']['default_gpu']


# ============================================================================
# 训练配置参数说明
# ============================================================================
# 
# 【model_args - 模型架构参数】
#   - group_count: 编码/解码组的数量，决定模型深度和空间下采样层级（典型值：3-5）
#   - n_hid: 初始隐藏特征维度，基础特征维度（典型值：64-256）
#   - n_blk_per_group: 每个组中的块数量，每个块是残差块（典型值：1-3）
#   - vocab_size: 码本大小，离散向量的数量（典型值：512-8192）
#   - n_init: 编码器输出维度，也是codebook嵌入维度（典型值：128-1024）
#   - input_channels: 输入通道数，3=RGB，1=灰度
#   - output_channels: 输出通道数，应与input_channels相同
#   - commitment_cost: VQ损失中的commitment cost，控制对齐强度（典型值：0.1-0.5）
#   - decay: EMA衰减率，控制码本更新速度（典型值：0.99-0.999）
#
# 【data_args - 数据加载参数】
#   - batch_size: 批次大小，每个批次包含的视频数量（受GPU内存限制，典型值：2-16）
#   - num_threads: 数据加载线程数，用于并行加载（典型值：CPU核心数的一半到全部）
#   - device_id: 设备ID，多GPU时使用（单GPU设为0）
#   - training_data_files: 训练数据文件列表，使用list_videos2()函数获取
#   - seed: 随机种子，保证实验可重复性
#   - sequence_length: 视频序列长度，一次处理的帧数（典型值：8-32）
#   - shard_id: 分片ID，多GPU训练时使用（0表示第一个分片）
#   - num_shards: 分片总数，1表示不分片（多GPU时等于GPU数量）
#   - initial_prefetch_size: 初始预取大小（已废弃，保留用于兼容性）
#
# 【train_args - 训练参数】
#   - num_steps: 训练总步数，总迭代次数（典型值：10000-100000）
#   - lr: 初始学习率，参数更新步长（典型值：1e-5到1e-3）
#   - lr_decay: 学习率衰减率，指数衰减（典型值：0.95-0.99）
#   - folder_name: 输出文件夹路径，保存checkpoint、日志、可视化结果等
#   - checkpoint_path: 检查点路径，用于恢复训练（None表示从头开始）
#   - device: 设备选择，'cpu'/'cuda'/'auto'（推荐'auto'自动检测）
#
# 【use_wandb - 其他参数】
#   - use_wandb: 是否使用Weights & Biases进行实验跟踪（需要安装wandb）
#
# ============================================================================

# params = {
#     'model_args': {
#         'group_count': 4,              # 编码/解码组数量，决定模型深度
#         'n_hid': 64,                   # 初始隐藏特征维度
#         'n_blk_per_group': 1,          # 每个组中的块数量
#         'vocab_size': 8192,            # 码本大小，离散向量数量
#         'n_init': 1024,                # 编码器输出维度，codebook嵌入维度
#         'input_channels': 3,           # 输入通道数（3=RGB）
#         'output_channels': 3,          # 输出通道数
#         'commitment_cost': 0.2,       # VQ损失中的commitment cost
#         'decay': 0.99                  # EMA衰减率，控制码本更新速度
#     },
#     'data_args': {
#         'batch_size': 1,               # 批次大小，每个批次包含的视频数量
#         'num_threads': 6,               # 数据加载线程数
#         'device_id': 0,                 # 设备ID（多GPU时使用）
#         'training_data_files': list_videos2('./data/processedVideo'),  # 训练数据文件列表
#         'seed': 2025,                  # 随机种子
#         'sequence_length': 16,        # 视频序列长度，一次处理的帧数
#         'shard_id': 0,                 # 分片ID（多GPU训练时使用）
#         'num_shards': 1,               # 分片总数（1表示不分片）
#         'initial_prefetch_size': 1024  # 初始预取大小（已废弃）
#     },
#     'train_args': {
#         'num_steps': 100000,             # 训练总步数
#         'lr': 5e-4,                    # 初始学习率
#         'lr_decay': 0.98,              # 学习率衰减率
#         # 'folder_name': '/opt/project/data/trained_video2/' + datetime.datetime.today().strftime('%Y-%m-%d'),
#         'folder_name': './clipdata/model',  # 输出文件夹路径
#         # 'checkpoint_path': '/E:/work/sign/VQ-VAE/VqVaeVideo-master/clips/videos/checkpoint66000.pth.tar',
#         'checkpoint_path': 'clipdata/history_data/18MP-1e-3/model/checkpoint22500.pth.tar',        # 检查点路径（None表示从头开始训练）
#         # 设备选择: 'cpu' - 强制使用CPU, 'cuda' - 强制使用GPU, 'auto' - 自动检测(默认)
#         'device': 'auto'                # 设备选择（推荐'auto'自动检测）
#     },
#     'use_wandb': True                 # 是否使用wandb进行实验跟踪
# }




class ImageDataset(Dataset):
    """
    基于图片文件的数据集，用于加载随机图片序列
    所有图片都是独立的，随机选择组成序列
    """
    def __init__(self, filenames, sequence_length=16, random_shuffle=True, seed=None, 
                 target_size=(256, 256)):
        """
        参数:
            filenames: 图片文件路径列表（所有图片都是随机的，不需要分组）
            sequence_length: 要加载的帧数（默认16）
            random_shuffle: 是否随机选择图片
            seed: 随机种子
            target_size: 目标分辨率 (height, width)
        """
        self.filenames = filenames
        self.sequence_length = sequence_length
        self.random_shuffle = random_shuffle
        self.target_size = target_size
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 数据集大小：每个样本随机选择 sequence_length 张图片
        # 数据集大小设为图片数量，每个样本都随机选择
        self.num_samples = len(filenames) if len(filenames) > 0 else 0
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        随机选择 sequence_length 张图片组成序列
        """
        frames = []
        frames_read = 0
        first_image_path = None
        
        # 随机选择 sequence_length 张图片
        while frames_read < self.sequence_length:
            # 随机选择一张图片
            if self.random_shuffle:
                image_path = random.choice(self.filenames)
            else:
                # 如果不随机，按顺序循环选择
                image_idx = (idx * self.sequence_length + frames_read) % len(self.filenames)
                image_path = self.filenames[image_idx]
            
            # 记录第一张图片路径
            if frames_read == 0:
                first_image_path = image_path
            
            # 读取图片
            frame = cv2.imread(image_path)
            if frame is None:
                # 如果读取失败，使用黑帧或重复上一帧
                if len(frames) > 0:
                    frame = frames[-1].copy()
                else:
                    frame = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            else:
                # 转换 BGR 到 RGB（OpenCV 默认读取为 BGR）
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 调整大小
                if frame.shape[:2] != self.target_size:
                    frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]), 
                                     interpolation=cv2.INTER_LINEAR)
            
            frames.append(frame)
            frames_read += 1
        
        # 转换为 numpy 数组: (sequence_length, height, width, channels)
        video = np.stack(frames[:self.sequence_length], axis=0)
        del frames
        
        # 确保是 uint8 类型
        video = video.astype(np.uint8)
        
        # 返回视频数据和路径（使用第一张图片路径作为代表）
        return video, first_image_path if first_image_path else ""


class VideoDataset(Dataset):
    """
    基于 OpenCV 的视频数据集，优化显存使用
    只加载指定数量的帧，不加载整个视频
    """
    def __init__(self, filenames, sequence_length=16, random_shuffle=True, seed=None, 
                 target_size=(256, 256), max_frames_to_check=1000):
        """
        参数:
            filenames: 视频文件路径列表
            sequence_length: 要加载的帧数（默认16）
            random_shuffle: 是否随机选择起始帧
            seed: 随机种子
            target_size: 目标分辨率 (height, width)，用于下采样减少显存
            max_frames_to_check: 检查视频总帧数时的最大限制（避免大视频卡住）
        """
        self.filenames = filenames
        self.sequence_length = sequence_length
        self.random_shuffle = random_shuffle
        self.target_size = target_size
        self.max_frames_to_check = max_frames_to_check
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        video_path = self.filenames[idx]
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
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
        
        # 如果视频帧数足够，随机选择起始位置
        if total_frames > self.sequence_length and self.random_shuffle:
            start_frame = random.randint(0, max(0, total_frames - self.sequence_length))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 预分配数组，避免动态扩展
        frames = []
        frames_read = 0
        
        # 只读取需要的帧数，不多读
        while frames_read < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                # 如果视频结束但帧数不够，填充
                if len(frames) == 0:
                    # 如果一帧都没读到，返回黑帧
                    frame = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
                else:
                    # 重复最后一帧
                    frame = frames[-1].copy()
            else:
                # 立即调整大小以减少内存占用
                if frame.shape[:2] != self.target_size:
                    frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]), 
                                     interpolation=cv2.INTER_LINEAR)
            
            frames.append(frame)
            frames_read += 1
        
        cap.release()
        del cap  # 立即释放资源
        
        # 转换为 numpy 数组: (sequence_length, height, width, channels)
        # 只保留需要的帧数
        video = np.stack(frames[:self.sequence_length], axis=0)
        
        # 清理frames列表释放内存
        del frames
        
        # 确保是 uint8 类型，值在 0-255 范围（最小内存占用）
        video = video.astype(np.uint8)
        
        # 返回视频数据和路径，用于 bad case 检测
        # 注意：保持为uint8，在collate_fn中再转换为float32（如果需要）
        return video, video_path


def video_pipe(filenames, config_path:str,batch_size=4, num_threads=6, device_id=0, 
               sequence_length=16, shard_id=0, num_shards=1, 
               initial_prefetch_size=1024, seed=None, random_shuffle=False,
               target_size=(256, 256)):
    """
    创建视频或图片数据加载器，优化显存使用
    根据配置文件中的 use_images 参数决定加载视频还是图片
    
    显存优化措施:
    1. 只读取需要的帧数（sequence_length），不读取整个视频
    2. 立即调整视频分辨率到target_size，减少内存占用
    3. 使用uint8类型存储，直到需要时再转换为float32
    4. 及时释放视频捕获对象和临时数据
    5. 减少DataLoader的预取数量
    
    参数:
        filenames: 视频文件路径列表（视频模式）或图片文件路径列表（图片模式）
        config_path: 配置文件路径，用于读取 use_images 参数
        batch_size: 批次大小（减少此值可降低显存占用）
        num_threads: 数据加载线程数
        device_id: 设备ID
        sequence_length: 要加载的帧数（默认16，减少此值可降低显存占用）
        shard_id: 分片ID
        num_shards: 分片总数
        initial_prefetch_size: 初始预取大小（已废弃）
        seed: 随机种子
        random_shuffle: 是否随机选择起始帧
        target_size: 目标分辨率 (height, width)，用于下采样减少显存
                    例如 (128, 128) 比 (256, 256) 显存占用减少4倍
    
    显存占用估算（单个视频）:
        - uint8: sequence_length * height * width * 3 字节
        - 例如: 16 * 256 * 256 * 3 = 3MB (uint8)
        - float32: 16 * 256 * 256 * 3 * 4 = 12MB (float32)
        - batch显存 = 单个视频显存 * batch_size
    
    进一步减少显存的建议:
        1. 减小 batch_size（例如从4改为1或2）
        2. 减小 sequence_length（例如从16改为8）
        3. 减小 target_size（例如从256x256改为128x128）
        4. 在训练代码中使用混合精度训练（torch.cuda.amp）
    """
    # 读取配置文件，检查是否使用图片模式
    use_images = False
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            use_images = config.get('use_images', False)
            if use_images:
                logging.info(f"检测到 use_images=True，使用图片数据集模式")
            else:
                logging.info(f"检测到 use_images=False，使用视频数据集模式")
        except Exception as e:
            logging.warning(f"无法读取配置文件 {config_path}，使用默认视频模式: {e}")
    else:
        logging.info(f"配置文件 {config_path} 不存在，使用默认视频模式")
    
    # 处理分片逻辑
    if num_shards > 1:
        # 将文件列表分片
        total_files = len(filenames)
        files_per_shard = total_files // num_shards
        start_idx = shard_id * files_per_shard
        end_idx = start_idx + files_per_shard if shard_id < num_shards - 1 else total_files
        filenames = filenames[start_idx:end_idx]
    
    # 根据 use_images 选择数据集类型
    if use_images:
        dataset = ImageDataset(filenames, sequence_length=sequence_length, 
                              random_shuffle=random_shuffle, seed=seed,
                              target_size=target_size)
        logging.info(f"创建图片数据集，共 {len(filenames)} 个图片文件，数据集大小: {len(dataset)}")
    else:
        dataset = VideoDataset(filenames, sequence_length=sequence_length, 
                          random_shuffle=random_shuffle, seed=seed,
                          target_size=target_size)
        logging.info(f"创建视频数据集，共 {len(filenames)} 个视频文件，数据集大小: {len(dataset)}")
    
    # 自定义 collate_fn 确保数据格式正确，优化显存使用
    def collate_fn(batch):
        # batch 是一个列表，每个元素是 (video, video_path) 的元组
        # 分离视频数据和路径
        videos = [item[0] for item in batch]
        paths = [item[1] for item in batch]
        
        # 堆叠成 (batch_size, sequence_length, height, width, channels) 的 numpy 数组
        # 使用 uint8 类型堆叠，减少内存占用
        batch_array = np.stack(videos, axis=0)  # (batch_size, sequence_length, height, width, channels)
        
        # 清理原始列表释放内存
        del videos
        
        # 转换为 tensor，但保持为 uint8 类型以节省显存
        # 注意：如果模型需要 float32，应该在模型输入时转换，而不是在这里
        # 这样可以避免在CPU上占用过多内存
        batch_tensor = torch.from_numpy(batch_array)  # 仍然是 uint8
        
        # 返回 tensor 和路径列表
        return batch_tensor, paths
    
    # 创建 DataLoader，优化显存使用
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=random_shuffle,
        num_workers=num_threads,
        pin_memory=False, #CPU 模式下不需要 pin_memory，避免额外内存占用
        drop_last=True,  # 确保批次大小一致
        collate_fn=collate_fn,  # 使用自定义的 collate_fn
        persistent_workers=True, #不保持worker进程，减少内存占用
        prefetch_factor=2 if num_threads > 0 else None  # 减少预取数量，降低内存占用
    )
    
    # 为了兼容原有代码，添加 epoch_size 方法
    class DataLoaderWrapper:
        def __init__(self, loader):
            self.loader = loader
            self.dataset = loader.dataset
        
        def __iter__(self):
            return iter(self.loader)
        
        def epoch_size(self):
            # 返回每个 epoch 的批次数量
            return {'__Video_0': len(self.loader)}
    
    return DataLoaderWrapper(dataloader)
