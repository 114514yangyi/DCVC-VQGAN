# DCVC 模型训练详细指南

本文档详细说明如何训练 DCVC (Deep Contextual Video Compression) 模型。DCVC 是一个基于深度学习的视频压缩模型，使用多阶段渐进式训练策略。

## 目录

1. [环境要求](#环境要求)
2. [训练概述](#训练概述)
3. [训练阶段详解](#训练阶段详解)
4. [GOP 大小渐进式训练协议](#gop-大小渐进式训练协议)
5. [数据集准备](#数据集准备)
6. [训练命令详解](#训练命令详解)
7. [模型架构与损失函数](#模型架构与损失函数)
8. [训练参数说明](#训练参数说明)
9. [评估与验证](#评估与验证)
10. [检查点管理](#检查点管理)
11. [常见问题与解决方案](#常见问题与解决方案)

---

## 环境要求

### Python 版本

**推荐 Python 版本：Python 3.10 或 3.11**

- **最低要求**：Python 3.10
- **推荐版本**：Python 3.10 或 3.11
- **支持版本**：Python 3.10, 3.11, 3.12

**原因**：
- PyTorch 2.6.0 需要 Python 3.10+
- NumPy 2.1.2 需要 Python 3.10+
- Python 3.10/3.11 具有最佳的稳定性和兼容性

### 系统要求

- **操作系统**：Linux (推荐) 或 macOS
- **CUDA**：CUDA 12.6 (用于 GPU 训练，Linux) 或 MPS (Apple Silicon Mac)
- **GPU**：NVIDIA GPU (Linux) 或 Apple Silicon (M1/M2/M3, macOS)
- **内存**：建议 32GB+ RAM
- **存储**：足够的空间用于数据集和检查点（建议 100GB+）

**Mac 用户注意**：
- Apple Silicon (M1/M2/M3) 可以使用 MPS 加速
- Intel Mac 使用 CPU 训练（速度较慢）
- 已移除 CUDA 相关依赖，适配 Mac 环境

### 主要依赖

根据 `requirements.txt`，主要依赖包括：

- **PyTorch**: 2.6.0+cu126
- **NumPy**: 2.1.2
- **CompressAI**: 用于熵编码和 I-frame 压缩
- **Accelerate**: 1.6.0 (分布式训练)
- **torch-ema**: 0.3 (指数移动平均)
- **timm**: 1.0.15 (模型工具)
- **pytorch-msssim**: 1.0.0 (MS-SSIM 指标)

### 安装步骤

#### Linux (CUDA)

```bash
# 1. 创建虚拟环境（推荐使用 Python 3.10 或 3.11）
conda create -n dcvc python=3.10
conda activate dcvc

# 2. 安装 PyTorch (CUDA 12.6)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
python -c "import compressai; print(f'CompressAI {compressai.__version__}')"
```

#### macOS (Apple Silicon 或 Intel)

```bash
# 1. 创建虚拟环境（推荐使用 Python 3.10 或 3.11）
conda create -n dcvc python=3.10
conda activate dcvc

# 或者使用 venv
python3.10 -m venv dcvc_env
source dcvc_env/bin/activate

# 2. 安装 PyTorch (Mac 版本，支持 MPS/CPU)
pip install torch torchvision torchaudio

# 3. 安装其他依赖（已适配 Mac，移除了 CUDA 相关包）
pip install -r requirements.txt

# 4. 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}' if hasattr(torch.backends, 'mps') else 'CPU only')"
python -c "import compressai; print(f'CompressAI {compressai.__version__}')"
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"
```

### 环境检查

在开始训练前，建议运行以下检查：

```bash
# 检查 Python 版本
python --version  # 应该显示 Python 3.10.x 或 3.11.x

# 检查 GPU 是否可用（Linux）
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# 检查 MPS 是否可用（macOS Apple Silicon）
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}' if hasattr(torch.backends, 'mps') else 'MPS not supported')"

# 检查关键依赖
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
python -c "import compressai; print('CompressAI OK')"
python -c "import accelerate; print('Accelerate OK')"
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"  # 视频数据集需要
```

---

## 训练概述

DCVC 模型采用**多阶段渐进式训练策略**，通过分阶段训练不同的模块，逐步优化整个视频压缩系统。训练过程分为 4 个主要阶段，每个阶段专注于不同的组件和优化目标。

### 训练流程总览

```
Stage 1 (运动估计) → Stage 2 (残差编码) → Stage 3 (带比特成本的残差编码) 
→ Stage 4 (端到端优化)
```

### 关键特性

- **多阶段训练**：分阶段训练不同模块，避免训练不稳定
- **GOP 渐进式训练**：在每个阶段内，GOP 大小从 2 → 3 → 5 → 7 逐步增加
- **学习率重置**：每次增加 GOP 大小时，需要重置学习率到 1e-4
- **I-frame 压缩**：支持 CompressAI 预训练的 I-frame 模型或 VQGAN 模型
- **真实熵编码**：基于 CompressAI 的真实熵编码支持

---

## 训练阶段详解

### Stage 1: 运动估计与运动编码训练

**目标**：训练运动估计（光流）和运动向量编码模块

**训练内容**：
- 运动估计模块 (`opticFlow`: ME_Spynet)
- 运动向量编码器 (`mvEncoder`)
- 运动向量解码器 (`mvDecoder_part1`, `mvDecoder_part2`)
- 运动向量先验编码器/解码器 (`mvpriorEncoder`, `mvpriorDecoder`)
- 运动向量熵模型 (`bitEstimator_z_mv`, `gaussian_conditional_mv`)

**损失函数**：
```
L_me = λ * MSE(pixel_rec, input_image) + BPP_mv_y + BPP_mv_z
```
其中：
- `pixel_rec` 是运动补偿后的像素级预测
- `BPP_mv_y` 是运动向量特征的比特率
- `BPP_mv_z` 是运动向量先验的比特率

**冻结模块**：无（所有运动相关模块都参与训练）

**训练策略**：
- 使用运动补偿后的像素预测作为重建结果
- 只优化运动相关的比特率成本
- 为后续阶段提供良好的运动估计基础

### Stage 2: 残差编码训练（运动模块冻结）

**目标**：在固定运动估计的基础上，训练残差编码模块

**训练内容**：
- 上下文编码器 (`contextualEncoder`)
- 上下文解码器 (`contextualDecoder_part1`, `contextualDecoder_part2`)
- 先验编码器/解码器 (`priorEncoder`, `priorDecoder`)
- 残差熵模型 (`bitEstimator_z`, `gaussian_conditional`)
- 时间先验编码器 (`temporalPriorEncoder`)

**损失函数**：
```
L_rec = λ * MSE(recon_image, input_image)
```
注意：此阶段**不包含比特率项**，只优化重建质量

**冻结模块**：
- `opticFlow` (运动估计)
- `mvEncoder`, `mvDecoder_part1`, `mvDecoder_part2` (运动编码/解码)
- `mvpriorEncoder`, `mvpriorDecoder` (运动先验)
- `auto_regressive_mv`, `entropy_parameters_mv` (运动熵参数)
- `bitEstimator_z_mv` (运动熵瓶颈)

**训练策略**：
- 使用 Stage 1 训练好的运动估计结果
- 专注于学习如何编码残差信息
- 为 Stage 3 引入比特率约束做准备

### Stage 3: 带比特率约束的残差编码训练

**目标**：在残差编码中引入比特率约束，学习率失真权衡

**训练内容**：
- 与 Stage 2 相同的模块（残差编码相关）
- 运动模块仍然冻结

**损失函数**：
```
L_con = λ * MSE(recon_image, input_image) + BPP_y + BPP_z
```
其中：
- `BPP_y` 是残差特征的比特率
- `BPP_z` 是残差先验的比特率

**冻结模块**：与 Stage 2 相同（运动模块全部冻结）

**训练策略**：
- 在重建质量和比特率之间找到平衡
- 学习如何高效编码残差信息
- 为端到端优化做准备

### Stage 4: 端到端联合优化

**目标**：解冻所有模块，进行端到端联合训练

**训练内容**：
- **所有模块**都参与训练（包括运动模块和残差模块）

**损失函数**：
```
L_all = λ * MSE(recon_image, input_image) + BPP_total
```
其中：
```
BPP_total = BPP_y + BPP_z + BPP_mv_y + BPP_mv_z
```

**冻结模块**：无（所有模块都参与训练）

**训练策略**：
- 联合优化所有模块
- 可以启用 GOP 级别优化 (`--stage4_gop_opt`)
- 使用学习率调度器 (`--use_scheduler`)

**GOP 优化模式** (`--stage4_gop_opt`)：
- 在 GOP 级别累积梯度
- 考虑整个 GOP 的误差传播
- 适合长 GOP 序列的微调

**注意**：当前版本的 `train_dcvc.py` 仅支持 Stage 1-4，不支持 Stage 5 微调。如需进行长序列微调，请参考代码库的其他实现或等待后续更新。

---

## GOP 大小渐进式训练协议

### 为什么需要渐进式 GOP 训练？

DCVC 模型需要学习视频帧之间的时间依赖关系。直接使用长 GOP（如 7 帧）训练可能导致：
- 训练不稳定
- 梯度消失/爆炸
- 难以收敛

渐进式增加 GOP 大小可以帮助模型：
- 逐步学习更长的时序依赖
- 稳定训练过程
- 提高最终性能

### GOP 渐进式训练流程

**在每个训练阶段（Stage 1-4）内，必须遵循以下 GOP 大小递增顺序：**

```
GOP 2 → GOP 3 → GOP 5 → GOP 7
```

### 详细步骤

#### 步骤 1: 从 GOP 2 开始

**Linux (CUDA)**:
```bash
# 训练几个 epoch（例如 5-10 个）
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /root/DCVC-VQGAN/DCVC-family/data \
  --val_video_dir /root/DCVC-VQGAN/DCVC-family/data \
  --checkpoint_dir ./checkpoints \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 512 \
  --quality_index 4 \
  --stage 1 \
  --gop_size 2 \
  --epochs 10 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 100 \
  --num_workers 4
```

**macOS (Apple Silicon/Intel)**:
```bash
# 训练几个 epoch（例如 5-10 个）
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --gop_size 2 \
  --epochs 10 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4
```

#### 步骤 2: 增加到 GOP 3，重置学习率

**Linux (CUDA)**:
```bash
# 重要：必须使用 --force_learning_rate 1e-4 重置学习率
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /root/DCVC-VQGAN/DCVC-family/data \
  --val_video_dir /root/DCVC-VQGAN/DCVC-family/data \
  --checkpoint_dir ./checkpoints \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 512 \
  --quality_index 4 \
  --stage 1 \
  --gop_size 3 \
  --epochs 20 \
  --batch_size 4 \
  --force_learning_rate 1e-4 \
  --resume checkpoints/model_dcvc_lambda_512.0_quality_4_stage_1_best.pth \
  --crop_size 256 \
  --eval_freq 100 \
  --num_workers 4
```

**macOS (Apple Silicon/Intel)**:
```bash
# 重要：必须使用 --force_learning_rate 1e-4 重置学习率
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --gop_size 3 \
  --epochs 10 \
  --batch_size 4 \
  --force_learning_rate 1e-4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_latest.pth \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4
```

#### 步骤 3: 增加到 GOP 5，重置学习率

**Linux (CUDA)**:
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --gop_size 5 \
  --epochs 10 \
  --batch_size 4 \
  --force_learning_rate 1e-4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_latest.pth \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4
```

**macOS (Apple Silicon/Intel)**:
```bash
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --gop_size 5 \
  --epochs 10 \
  --batch_size 4 \
  --force_learning_rate 1e-4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_latest.pth \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4
```

#### 步骤 4: 增加到 GOP 7，重置学习率

**Linux (CUDA)**:
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --gop_size 7 \
  --epochs 10 \
  --batch_size 4 \
  --force_learning_rate 1e-4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_latest.pth \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4
```

**macOS (Apple Silicon/Intel)**:
```bash
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --gop_size 7 \
  --epochs 10 \
  --batch_size 4 \
  --force_learning_rate 1e-4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_latest.pth \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4
```

### 关键注意事项

1. **必须重置学习率**：每次增加 GOP 大小时，使用 `--force_learning_rate 1e-4` 重置学习率
2. **使用 --resume**：从上一个 GOP 大小的最新检查点继续训练
3. **不要跳过 GOP 大小**：必须按照 2 → 3 → 5 → 7 的顺序
4. **每个阶段都要重复**：Stage 1-4 的每个阶段都需要遵循这个协议

### GOP 大小选择建议

- **GOP 2**: 训练 5-10 个 epoch
- **GOP 3**: 训练 5-10 个 epoch
- **GOP 5**: 训练 5-10 个 epoch
- **GOP 7**: 训练 10-20 个 epoch（最终 GOP 大小）

---

## 数据集准备

### 数据集类型选择

DCVC 训练支持两种数据集格式：

1. **图片序列数据集**（传统方式）：Vimeo-90k、BVI-AOM、HEVC-B
2. **MP4 视频数据集**（新增）：直接使用 MP4 视频文件

通过 `--use_video_dataset` 参数选择使用视频数据集。

### MP4 视频数据集（推荐，新增功能）

**用途**：Stage 1-4 的训练数据集和验证数据集

**优势**：
- 无需预处理，直接使用 MP4 文件
- 支持任意长度的视频序列
- 内存优化：只读取需要的帧数，不加载整个视频

**目录结构**：
```
train_videos/
├── video_001.mp4
├── video_002.mp4
├── video_003.mp4
└── ...

val_videos/
├── test_video_001.mp4
├── test_video_002.mp4
└── ...
```

**使用方法**：
```bash
--train_video_dir /path/to/train/videos \
--val_video_dir /path/to/val/videos
```

**数据增强**：
- 随机裁剪：`--crop_size 256`（默认）
- 随机选择 GOP 起始帧（从视频中随机选择连续 GOP 大小的帧）
- 自动调整视频分辨率（如果需要）

**注意事项**：
- 视频文件必须是 MP4 格式
- 建议视频分辨率至少为 256x256
- 视频长度应至少包含 GOP 大小的帧数
- 验证视频目录用于训练过程中的评估

---

## 训练命令详解

### 完整训练流程示例

以下是一个完整的训练流程，展示如何从 Stage 1 训练到 Stage 4。

#### Stage 1: 运动估计训练（GOP 渐进式）

**使用 CompressAI I-frame 模型（默认）**：

**Linux (CUDA)**:
```bash
# ========== Stage 1, GOP 2 ==========
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 2 \
  --learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4

# ========== Stage 1, GOP 3 ==========
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 3 \
  --force_learning_rate 1e-4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_latest.pth \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4

# ========== Stage 1, GOP 5 ==========
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 5 \
  --force_learning_rate 1e-4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_latest.pth \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4

# ========== Stage 1, GOP 7 ==========
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 7 \
  --force_learning_rate 1e-4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_latest.pth \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4
```

**macOS (Apple Silicon/Intel)**:
```bash
# ========== Stage 1, GOP 2 ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 2 \
  --learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4

# ========== Stage 1, GOP 2 (使用 VQGAN I-frame) ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type vqgan \
  --vqgan_config /path/to/vqgan_config.json \
  --vqgan_checkpoint /path/to/vqgan_checkpoint.pth \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 2 \
  --learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4

# ========== Stage 1, GOP 3 ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 3 \
  --force_learning_rate 1e-4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_latest.pth \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4

# ========== Stage 1, GOP 5 ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 5 \
  --force_learning_rate 1e-4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_latest.pth \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4

# ========== Stage 1, GOP 7 ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 7 \
  --force_learning_rate 1e-4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_latest.pth \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4
```

#### Stage 2: 残差编码训练（运动模块冻结）

**Linux (CUDA)**:
```bash
# ========== Stage 2, GOP 2 ==========
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 2 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 2 \
  --learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_best.pth

# ========== Stage 2, GOP 3 ==========
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 2 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 3 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_2_latest.pth

# ========== Stage 2, GOP 5 ==========
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 2 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 5 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_2_latest.pth

# ========== Stage 2, GOP 7 ==========
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 2 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 7 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_2_latest.pth
```

**macOS (Apple Silicon/Intel)**:
```bash
# ========== Stage 2, GOP 2 ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 2 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 2 \
  --learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_best.pth

# ========== Stage 2, GOP 3 ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 2 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 3 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_2_latest.pth

# ========== Stage 2, GOP 5 ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 2 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 5 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_2_latest.pth

# ========== Stage 2, GOP 7 ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 2 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 7 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_2_latest.pth
```

#### Stage 3: 带比特率约束的残差编码

**Linux (CUDA)**:
```bash
# ========== Stage 3, GOP 2 ==========
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 3 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 2 \
  --learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_2_best.pth

# ========== Stage 3, GOP 3 ==========
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 3 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 3 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_3_latest.pth

# ========== Stage 3, GOP 5 ==========
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 3 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 5 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_3_latest.pth

# ========== Stage 3, GOP 7 ==========
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 3 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 7 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_3_latest.pth
```

**macOS (Apple Silicon/Intel)**:
```bash
# ========== Stage 3, GOP 2 ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 3 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 2 \
  --learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_2_best.pth

# ========== Stage 3, GOP 3 ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 3 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 3 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_3_latest.pth

# ========== Stage 3, GOP 5 ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 3 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 5 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_3_latest.pth

# ========== Stage 3, GOP 7 ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 3 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 7 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_3_latest.pth
```

#### Stage 4: 端到端优化

**Linux (CUDA)**:
```bash
# ========== Stage 4, GOP 2 ==========
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 4 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 2 \
  --learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --use_scheduler \
  --scheduler_patience 3 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_3_best.pth

# ========== Stage 4, GOP 3 ==========
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 4 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 3 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --use_scheduler \
  --scheduler_patience 3 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_4_latest.pth

# ========== Stage 4, GOP 5 ==========
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 4 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 5 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --use_scheduler \
  --scheduler_patience 3 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_4_latest.pth

# ========== Stage 4, GOP 7 ==========
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 4 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 7 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --use_scheduler \
  --scheduler_patience 3 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_4_latest.pth

# ========== Stage 4, GOP 7 + GOP 优化（可选）==========
# 在完成 Stage 4 GOP 7 训练后，可以启用 GOP 级别优化进行微调
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 4 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 7 \
  --stage4_gop_opt \
  --learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --use_scheduler \
  --scheduler_patience 3 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_4_latest.pth
```

**macOS (Apple Silicon/Intel)**:
```bash
# ========== Stage 4, GOP 2 ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 4 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 2 \
  --learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --use_scheduler \
  --scheduler_patience 3 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_3_best.pth

# ========== Stage 4, GOP 3 ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 4 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 3 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --use_scheduler \
  --scheduler_patience 3 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_4_latest.pth

# ========== Stage 4, GOP 5 ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 4 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 5 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --use_scheduler \
  --scheduler_patience 3 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_4_latest.pth

# ========== Stage 4, GOP 7 ==========
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 4 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 7 \
  --force_learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --use_scheduler \
  --scheduler_patience 3 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_4_latest.pth

# ========== Stage 4, GOP 7 + GOP 优化（可选）==========
# 在完成 Stage 4 GOP 7 训练后，可以启用 GOP 级别优化进行微调
python DCVC-family/DCVC/train_dcvc.py \
  --train_video_dir /path/to/train/videos \
  --val_video_dir /path/to/val/videos \
  --checkpoint_dir ./checkpoints \
  --i_frame_type compressai \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 4 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 7 \
  --stage4_gop_opt \
  --learning_rate 1e-4 \
  --crop_size 256 \
  --eval_freq 1 \
  --num_workers 4 \
  --use_scheduler \
  --scheduler_patience 3 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_4_latest.pth
```

**注意**：当前版本的 `train_dcvc.py` 仅支持 Stage 1-4，不支持 Stage 5 微调。

---

## 模型架构与损失函数

### DCVC 模型架构

DCVC 模型主要包含以下组件：

#### 1. 运动估计与编码

- **运动估计** (`opticFlow`): ME_Spynet，用于估计光流
- **运动编码器** (`mvEncoder`): 编码运动向量特征
- **运动解码器** (`mvDecoder_part1`, `mvDecoder_part2`): 解码并细化运动向量
- **运动先验编码器/解码器** (`mvpriorEncoder`, `mvpriorDecoder`): 编码运动向量的先验信息
- **运动熵模型**:
  - `bitEstimator_z_mv`: EntropyBottleneck，用于运动先验
  - `gaussian_conditional_mv`: GaussianConditional，用于运动特征

#### 2. 残差编码

- **特征提取** (`feature_extract`): 从参考帧提取特征
- **上下文细化** (`context_refine`): 细化运动补偿后的上下文
- **上下文编码器** (`contextualEncoder`): 编码残差特征
- **上下文解码器** (`contextualDecoder_part1`, `contextualDecoder_part2`): 解码残差特征
- **先验编码器/解码器** (`priorEncoder`, `priorDecoder`): 编码残差的先验信息
- **时间先验编码器** (`temporalPriorEncoder`): 编码时间上下文先验
- **残差熵模型**:
  - `bitEstimator_z`: EntropyBottleneck，用于残差先验
  - `gaussian_conditional`: GaussianConditional，用于残差特征

#### 3. 熵建模

- **自回归模型** (`auto_regressive`, `auto_regressive_mv`): MaskedConv2d，用于上下文建模
- **熵参数网络** (`entropy_parameters`, `entropy_parameters_mv`): 预测高斯分布的参数

### 损失函数详解

#### Stage 1 损失

```python
L_me = λ * MSE(pixel_rec, input_image) + BPP_mv_y + BPP_mv_z
```

- **目标**：训练运动估计和编码
- **重建**：使用像素级运动补偿 (`pixel_rec`)
- **比特率**：只考虑运动相关的比特率

#### Stage 2 损失

```python
L_rec = λ * MSE(recon_image, input_image)
```

- **目标**：训练残差编码（无比特率约束）
- **重建**：使用完整的残差解码结果 (`recon_image`)
- **比特率**：不包含比特率项

#### Stage 3 损失

```python
L_con = λ * MSE(recon_image, input_image) + BPP_y + BPP_z
```

- **目标**：在残差编码中引入比特率约束
- **重建**：使用完整的残差解码结果
- **比特率**：只考虑残差相关的比特率

#### Stage 4 损失

```python
L_all = λ * MSE(recon_image, input_image) + BPP_total
```

其中：
```python
BPP_total = BPP_y + BPP_z + BPP_mv_y + BPP_mv_z
```

- **目标**：端到端联合优化
- **重建**：使用完整的解码结果
- **比特率**：包含所有组件的比特率

### 比特率计算

每个组件的比特率计算方式：

```python
BPP = -log2(likelihood).sum() / (batch_size * height * width)
```

- `BPP_y`: 残差特征的比特率
- `BPP_z`: 残差先验的比特率
- `BPP_mv_y`: 运动向量特征的比特率
- `BPP_mv_z`: 运动向量先验的比特率

---

## 训练参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--train_video_dir` | 训练视频目录路径（MP4 文件） | `/path/to/train/videos` |
| `--checkpoint_dir` | 检查点保存目录 | `./checkpoints` |
| `--lambda_value` | 率失真权衡参数 | `2048` (越高质量越好，压缩率越低) |
| `--quality_index` | 质量索引 (1-4) | `4` |
| `--stage` | 训练阶段 (1-4) | `1` |

### 数据集参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--train_video_dir` | 训练视频目录路径（MP4 文件，必需） | `None` |
| `--val_video_dir` | 验证视频目录路径（MP4 文件，可选） | `None` |
| `--crop_size` | 训练时的随机裁剪大小 | `256` |
| `--num_workers` | 数据加载器工作进程数 | `4` |

**数据集说明**：
- 当前版本仅支持 MP4 视频数据集
- 训练视频目录必须包含 `.mp4` 文件
- 验证视频目录用于训练过程中的评估（如果提供）

### I-frame 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--i_frame_type` | I-frame 模型类型 (`compressai` 或 `vqgan`) | `compressai` |
| `--i_frame_model_name` | CompressAI I-frame 模型名称（仅当 `--i_frame_type=compressai` 时使用） | `cheng2020-anchor` |
| `--i_frame_quality` | I-frame 质量等级 (1-8)（仅当 `--i_frame_type=compressai` 时使用） | `6` |
| `--i_frame_pretrained` | 使用预训练 I-frame 权重（仅当 `--i_frame_type=compressai` 时使用） | `True` |
| `--vqgan_config` | VQGAN 配置文件路径（仅当 `--i_frame_type=vqgan` 时使用，必需） | `None` |
| `--vqgan_checkpoint` | VQGAN 检查点路径（仅当 `--i_frame_type=vqgan` 时使用，必需） | `None` |

**CompressAI I-frame 模型**（`--i_frame_type=compressai`）：
- `bmshj2018-factorized`
- `bmshj2018-hyperprior`
- `cheng2020-anchor`
- `cheng2020-attn`
- 等等（见 CompressAI 文档）

**VQGAN I-frame 模型**（`--i_frame_type=vqgan`）：
- 需要提供 VQGAN 配置文件和检查点
- 使用最简单的均匀分布假设计算 BPP
- 支持与 CompressAI 模型相同的接口

### 训练超参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 每个阶段的训练轮数 | `10` |
| `--batch_size` | 批次大小 | `4` |
| `--learning_rate` | 初始学习率 | `1e-4` |
| `--force_learning_rate` | 强制设置学习率（用于 GOP 切换） | `None` |
| `--gop_size` | GOP 大小（帧数） | `7` |
| `--grad_clip_max_norm` | 梯度裁剪最大范数 | `1.0` |

**注意**：当前版本不支持 Stage 5 微调参数。

### 优化器与调度器

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_scheduler` | 使用学习率调度器 | `False` |
| `--scheduler_factor` | 学习率衰减因子 | `0.5` |
| `--scheduler_patience` | 调度器耐心值（epochs） | `3` |
| `--scheduler_min_lr` | 最小学习率 | `1e-6` |

### EMA (指数移动平均)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_ema` | 使用 EMA | `True` |
| `--ema_decay` | EMA 衰减率 | `0.999` |
| `--evaluate_both` | 同时评估正常模型和 EMA 模型 | `True` |

### 评估参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--eval_freq` | 评估频率（每 N 个 epoch） | `1` |
| `--eval_max_frames` | HEVC-B 评估最大帧数 | `96` |
| `--skip_eval` | 跳过评估 | `False` |

### 检查点管理

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--resume` | 恢复训练的检查点路径 | `None` |
| `--load_pretrained` | 仅加载预训练权重（不恢复训练状态） | `None` |

### Stage 4 特殊参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--stage4_gop_opt` | 启用 GOP 级别优化 | `False` |

**GOP 级别优化说明**：
- 在 GOP 级别累积梯度（而不是逐帧优化）
- 考虑整个 GOP 的误差传播
- 适合长 GOP 序列的微调
- 训练速度较慢，但可能提高性能

### Accelerate 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mixed_precision` | 混合精度训练 (`no`, `fp16`, `bf16`) | `no` |
| `--gradient_accumulation_steps` | 梯度累积步数 | `1` |

### 其他参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--seed` | 随机种子 | `42` |
| `--compile` | 编译模型（PyTorch 2.0+） | `False` |

---

## 评估与验证

### 视频评估

训练过程中会自动在验证视频数据集上进行评估（如果提供了 `--val_video_dir`）。

**评估指标**：
- **PSNR** (Peak Signal-to-Noise Ratio): 峰值信噪比
- **BPP** (Bits Per Pixel): 每像素比特数
- **MSE** (Mean Squared Error): 均方误差
- **RD Loss**: 率失真损失 = BPP + λ * MSE

**评估输出**：
- 整体帧的平均指标（包括 I-frame 和 P-frame）
- I-frame 单独指标
- P-frame 平均指标
- 前 3 个 P-frame 的详细统计

**评估频率**：
- 由 `--eval_freq` 控制（默认每个 epoch 评估一次）
- 使用 `--skip_eval` 可以跳过评估

### 独立评估脚本

训练完成后，可以使用独立的评估脚本：

```bash
# 完整视频评估（I-frame + P-frames，使用 CompressAI I-frame）
python DCVC-family/DCVC/test_video.py \
  --p_frame_model_path ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_4_best.pth \
  --val_video_dir /path/to/val/videos \
  --i_frame_model cheng2020-anchor \
  --i_frame_quality 6 \
  --max_frames 96

# 完整视频评估（使用 VQGAN I-frame）
# 注意：如果 test_video.py 支持 VQGAN，需要添加相应参数
# python DCVC-family/DCVC/test_video.py \
#   --p_frame_model_path ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_4_best.pth \
#   --val_video_dir /path/to/val/videos \
#   --i_frame_type vqgan \
#   --vqgan_config /path/to/vqgan_config.json \
#   --vqgan_checkpoint /path/to/vqgan_checkpoint.pth \
#   --max_frames 96

# 真实压缩模式（使用实际熵编码，较慢）
python DCVC-family/DCVC/test_video.py \
  --p_frame_model_path /path/to/checkpoint.pth \
  --val_video_dir /path/to/val/videos \
  --i_frame_model cheng2020-anchor \
  --i_frame_quality 6 \
  --max_frames 96 \
  --real_compression
```

---

## 检查点管理

### 检查点命名规则

检查点文件按照以下规则命名：

```
model_dcvc_lambda_{lambda_value}_quality_{quality_index}_stage_{stage}_{type}.pth
```

**类型后缀**：
- `latest`: 最新检查点（每个 epoch 保存）
- `best`: 最佳检查点（基于验证损失）
- `best_ema`: 最佳 EMA 检查点
- `global_best`: 全局最佳检查点
- `final`: 阶段完成后的最终检查点

**示例**：
```
model_dcvc_lambda_2048.0_quality_4_stage_1_latest.pth
model_dcvc_lambda_2048.0_quality_4_stage_1_best.pth
model_dcvc_lambda_2048.0_quality_4_stage_4_global_best.pth
```

### 检查点内容

每个检查点包含：

```python
{
    'epoch': int,                    # 当前 epoch
    'model_state_dict': dict,        # 模型权重
    'optimizer_state_dict': dict,     # 优化器状态
    'loss': float,                   # 当前损失
    'best_loss': float,              # 最佳损失
    'global_best_loss': float,       # 全局最佳损失
    'stage': int,                   # 训练阶段
    'quality_index': int,            # 质量索引
    'lambda_value': float,           # Lambda 值
    'ema_state_dict': dict,          # EMA 状态（如果使用）
    'scheduler_state_dict': dict,     # 调度器状态（如果使用）
    'is_finetuning': bool            # 是否为微调阶段
}
```

### 自动检查点加载

训练脚本会自动处理检查点加载：

1. **当前阶段检查点优先**：如果存在当前阶段的检查点，会加载它
2. **前一个阶段检查点**：如果当前阶段没有检查点，会自动加载前一个阶段的最佳检查点
3. **显式指定**：使用 `--resume` 或 `--load_pretrained` 可以显式指定检查点

### 检查点使用场景

#### 继续训练（同一阶段）

```bash
--resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_latest.pth
```

- 恢复训练状态（epoch、优化器、调度器等）
- 继续同一阶段的训练

#### 开始新阶段

```bash
# 不需要 --resume，脚本会自动从前一阶段加载
--stage 2
```

- 自动加载 Stage 1 的最佳检查点
- 开始 Stage 2 的训练

#### 仅加载权重（不恢复训练状态）

```bash
--load_pretrained ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_best.pth
```

- 只加载模型权重
- 不恢复训练状态（从 epoch 0 开始）
- 适用于迁移学习场景

#### GOP 大小切换

```bash
--gop_size 3 \
--force_learning_rate 1e-4 \
--resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_latest.pth
```

- 从 GOP 2 切换到 GOP 3
- 重置学习率
- 继续训练

---

## 常见问题与解决方案

### 1. 训练不稳定

**问题**：损失出现 NaN 或 Inf

**解决方案**：
- 检查输入数据是否正常
- 降低学习率（使用 `--force_learning_rate 1e-5`）
- 减小批次大小
- 启用梯度裁剪（`--grad_clip_max_norm 1.0`）

### 2. 内存不足

**问题**：CUDA out of memory

**解决方案**：
- 减小批次大小（`--batch_size 2`）
- 减小裁剪大小（`--crop_size 128`）
- 减小 GOP 大小（但要注意遵循渐进式协议）
- 使用梯度累积（`--gradient_accumulation_steps 2`）

### 3. 忘记重置学习率

**问题**：切换 GOP 大小时性能下降

**解决方案**：
- **必须**在切换 GOP 大小时使用 `--force_learning_rate 1e-4`
- 检查当前学习率是否合适
- 如果学习率调度器降低了学习率，重置调度器

### 4. 检查点加载失败

**问题**：无法加载检查点或模型结构不匹配

**解决方案**：
- 检查检查点路径是否正确
- 确认 `--lambda_value` 和 `--quality_index` 与检查点匹配
- 检查模型结构是否发生变化
- 使用 `--load_pretrained` 仅加载权重（如果结构不匹配）

### 5. 评估速度慢

**问题**：HEVC-B 评估耗时过长

**解决方案**：
- 减少评估频率（`--eval_freq 5`）
- 减少评估帧数（`--eval_max_frames 32`）
- 使用 `--skip_eval` 跳过训练中的评估，训练后单独评估

### 6. Stage 切换问题

**问题**：从 Stage 1 切换到 Stage 2 时性能下降

**解决方案**：
- 确保 Stage 1 训练充分（至少完成 GOP 7 的训练）
- 检查 Stage 1 的最佳检查点是否正常
- 确认运动模块在 Stage 2 被正确冻结

### 7. GOP 渐进式训练问题

**问题**：直接使用 GOP 7 训练效果不好

**解决方案**：
- **必须**遵循渐进式协议：2 → 3 → 5 → 7
- 每个 GOP 大小训练足够的 epoch
- 每次切换时重置学习率

### 8. I-frame 模型加载失败

**问题**：无法加载 I-frame 模型（CompressAI 或 VQGAN）

**解决方案**：

**CompressAI 模型**：
- 检查 CompressAI 是否正确安装
- 确认模型名称正确（`--i_frame_model_name`）
- 检查质量等级是否在有效范围内（1-8）
- 确认网络连接（如果需要下载预训练权重）

**VQGAN 模型**：
- 检查 `--vqgan_config` 和 `--vqgan_checkpoint` 路径是否正确
- 确认配置文件格式正确（JSON 格式）
- 验证检查点文件是否完整且可读
- 检查 VQGAN 模型依赖是否正确安装
- 确认 `--i_frame_type vqgan` 参数已指定

### 9. 数据集路径问题

**问题**：找不到数据集或数据加载错误

**解决方案**：
- **视频数据集**：
  - 检查 `--train_video_dir` 路径是否正确（必需）
  - 检查 `--val_video_dir` 路径是否正确（可选，用于评估）
  - 确认目录中包含 `.mp4` 文件
  - 验证视频文件可以正常打开（使用 `cv2.VideoCapture` 测试）
  - 检查视频格式是否支持（建议使用 MP4）
  - 确认视频分辨率足够（至少 256x256）
  - 确认视频长度至少包含 GOP 大小的帧数

### 10. 分布式训练问题

**问题**：多 GPU 训练时出现问题

**解决方案**：
- 使用 `accelerate launch` 而不是直接运行
- 检查 Accelerate 配置
- 确认所有进程可以访问相同的数据集路径
- 检查检查点保存/加载逻辑（只在主进程执行）

---

## 训练最佳实践

### 1. 训练顺序

严格按照以下顺序进行训练：

```
Stage 1 (GOP 2→3→5→7) 
  → Stage 2 (GOP 2→3→5→7) 
    → Stage 3 (GOP 2→3→5→7) 
      → Stage 4 (GOP 2→3→5→7) 
        → Stage 4 GOP 优化（可选）
```

### 2. 学习率管理

- **初始学习率**：1e-4（所有阶段）
- **GOP 切换**：必须重置为 1e-4
- **Stage 4**：建议使用学习率调度器
- **Stage 5**：使用较小的学习率（1e-4 或更小）

### 3. 批次大小选择

- **Stage 1-4**：根据 GPU 内存选择（通常 4-8）
- **Stage 5**：使用较小的批次（通常 2-4），因为 GOP 更大

### 4. 评估策略

- **训练中**：每个 epoch 评估一次（`--eval_freq 1`）
- **训练后**：使用独立评估脚本进行详细评估
- **真实压缩**：训练后使用 `--real_compression` 进行真实压缩测试

### 5. 检查点保存

- **定期保存**：每个 epoch 保存 `latest` 检查点
- **最佳模型**：自动保存 `best` 检查点
- **备份**：定期备份重要的检查点

### 6. 监控训练

- **损失曲线**：监控训练损失和验证损失
- **PSNR/BPP**：监控率失真性能
- **梯度**：检查梯度是否正常（无 NaN/Inf）
- **学习率**：确认学习率调度正常

### 7. 超参数调优

- **Lambda 值**：根据目标质量选择（256, 512, 1024, 2048, ...）
- **I-frame 质量**：通常使用 6（与 P-frame 质量匹配）
- **GOP 大小**：训练时使用 7，评估时可以使用 32

### 8. 资源管理

- **GPU 内存**：根据可用内存调整批次大小和裁剪大小
- **训练时间**：每个阶段可能需要数小时到数天
- **存储空间**：确保有足够的空间保存检查点和日志

---

## 总结

DCVC 模型的训练是一个复杂但系统化的过程。关键要点：

1. **遵循多阶段训练协议**：按顺序完成 Stage 1-4
2. **GOP 渐进式训练**：在每个阶段内，GOP 大小从 2 → 3 → 5 → 7 逐步增加
3. **学习率管理**：切换 GOP 大小时必须重置学习率
4. **检查点管理**：合理使用检查点，确保训练连续性
5. **评估与监控**：定期评估模型性能，及时发现问题
6. **视频数据集**：当前版本仅支持 MP4 视频数据集，使用 `--train_video_dir` 和 `--val_video_dir` 参数
7. **I-frame 模型选择**：
   - **CompressAI 模型**（默认）：使用 `--i_frame_type compressai`，配合 `--i_frame_model_name` 和 `--i_frame_quality` 参数
   - **VQGAN 模型**：使用 `--i_frame_type vqgan`，配合 `--vqgan_config` 和 `--vqgan_checkpoint` 参数（必需）
   - 两种模型类型完全兼容，可以在训练过程中切换

通过遵循本指南，您应该能够成功训练 DCVC 模型。如果遇到问题，请参考"常见问题与解决方案"部分，或查看项目文档和代码注释。

---

**祝训练顺利！**
