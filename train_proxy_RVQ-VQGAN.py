#!/usr/bin/env python3
"""
DCVC 训练代理脚本

自动执行完整的4阶段训练流程，包括：
- Stage 1-4 的自动训练
- 每个阶段内的 GOP 渐进式训练（2→3→5→7）
- 自动管理检查点恢复和学习率重置
- 累计 epoch 计数

使用方法：
    python train_proxy.py --config config.json
    或
    python train_proxy.py --config config.json --start_stage 2 --start_gop 3
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class TrainingProxy:
    """训练代理类，管理完整的训练流程"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练代理
        
        Args:
            config: 配置字典，包含所有训练参数
        """
        self.config = config
        self.checkpoint_dir = config['checkpoint_dir']
        self.lambda_value = config['lambda_value']
        self.quality_index = config['quality_index']
        self.epochs_per_gop = config.get('epochs_per_gop', 10)  # 每个GOP训练的epoch数
        
        # 训练阶段和GOP配置
        self.stages = [1, 2, 3, 4]
        self.gop_sequence = [2, 3, 5, 7]  # GOP渐进式训练顺序
        
        # 累计epoch计数
        self.current_epoch = 0
        
    def get_checkpoint_path(self, stage: int, suffix: str = 'latest') -> str:
        """
        获取检查点路径
        
        Args:
            stage: 训练阶段
            suffix: 检查点后缀（latest, best等）
        
        Returns:
            检查点文件路径
        """

        filename = f'model_dcvc_lambda_{self.lambda_value}.0_quality_{self.quality_index}_stage_{stage}_{suffix}.pth'
        return os.path.join(self.checkpoint_dir, filename)
    
    def find_checkpoint(self, stage: int, gop: int) -> Optional[str]:
        """
        查找可用的检查点
        
        优先级：
        1. 当前stage和GOP的最新检查点
        2. 当前stage的前一个GOP的检查点
        3. 前一个stage的最佳检查点
        
        Args:
            stage: 当前训练阶段
            gop: 当前GOP大小
        
        Returns:
            检查点路径，如果未找到则返回None
        """
        # 1. 尝试当前stage的最新检查点（优先级最高）
        # 当从GOP 2切换到GOP 3时，应该使用GOP 2训练完成后保存的latest检查点
        current_stage_latest = self.get_checkpoint_path(stage, 'latest')
        if os.path.exists(current_stage_latest):
            print(f"  [Checkpoint] Found: {current_stage_latest}")
            return current_stage_latest
        
        # 2. 尝试当前stage的最佳检查点
        current_stage_best = self.get_checkpoint_path(stage, 'best')
        if os.path.exists(current_stage_best):
            print(f"  [Checkpoint] Found: {current_stage_best}")
            return current_stage_best
        
        # 3. 尝试前一个stage的最佳检查点（跨stage时使用）
        if stage > 1:
            prev_stage_best = self.get_checkpoint_path(stage - 1, 'best')
            if os.path.exists(prev_stage_best):
                print(f"  [Checkpoint] Found: {prev_stage_best}")
                return prev_stage_best
        
        # 4. 尝试前一个stage的最新检查点（跨stage时使用）
        if stage > 1:
            prev_stage_latest = self.get_checkpoint_path(stage - 1, 'latest')
            if os.path.exists(prev_stage_latest):
                print(f"  [Checkpoint] Found: {prev_stage_latest}")
                return prev_stage_latest
        
        # 如果都没找到，打印调试信息
        print(f"  [Checkpoint] Not found. Searched:")
        print(f"    - {current_stage_latest}")
        print(f"    - {current_stage_best}")
        if stage > 1:
            print(f"    - {self.get_checkpoint_path(stage - 1, 'best')}")
            print(f"    - {self.get_checkpoint_path(stage - 1, 'latest')}")
        print(f"  [Checkpoint] Starting from scratch (no resume)")
        return None
    
    def build_train_command(self, stage: int, gop: int, 
                          start_epoch: int, total_epochs: int,
                          resume_checkpoint: Optional[str] = None,
                          force_lr: Optional[float] = None) -> list:
        """
        构建训练命令
        
        Args:
            stage: 训练阶段
            gop: GOP大小
            start_epoch: 起始epoch（用于累计）
            total_epochs: 总epoch数（当前GOP要训练的epoch数）
            resume_checkpoint: 恢复训练的检查点路径
            force_lr: 强制设置的学习率（用于GOP切换时重置）
        
        Returns:
            命令参数列表
        """
        cmd = ['accelerate', 'launch', 'DCVC-family/DCVC/train_dcvc.py']
        
        # 数据集参数
        cmd.extend(['--train_video_dir', self.config['train_video_dir']])
        if self.config.get('val_video_dir'):
            cmd.extend(['--val_video_dir', self.config['val_video_dir']])
        
        # I-frame 模型配置（本代理专用于 RVQ-VQGAN I-frame）
        if not self.config.get('vqgan_config') or not self.config.get('vqgan_checkpoint'):
            raise ValueError("vqgan_config and vqgan_checkpoint must be provided for RVQ-VQGAN I-frame")
        cmd.extend(['--vqgan_config', self.config['vqgan_config']])
        cmd.extend(['--vqgan_checkpoint', self.config['vqgan_checkpoint']])
        # 固定 I-frame 大小（用于 BPP 计算），可选
        if self.config.get('i_frame_size') is not None:
            cmd.extend(['--i_frame_size', str(self.config['i_frame_size'])])
        # 固定 I-frame 大小（用于 BPP 计算），可选
        if self.config.get('i_frame_size') is not None:
            cmd.extend(['--i_frame_size', str(self.config['i_frame_size'])])
        
        # 训练参数
        cmd.extend(['--checkpoint_dir', self.checkpoint_dir])
        cmd.extend(['--lambda_value', str(self.lambda_value)])
        cmd.extend(['--quality_index', str(self.quality_index)])
        cmd.extend(['--stage', str(stage)])
        cmd.extend(['--gop_size', str(gop)])
        # 使用累计epoch数（从开始到当前GOP结束的总epoch数）
        cumulative_epochs = start_epoch + total_epochs
        cmd.extend(['--epochs', str(cumulative_epochs)])
        cmd.extend(['--batch_size', str(self.config.get('batch_size', 4))])
        cmd.extend(['--crop_size', str(self.config.get('crop_size', 256))])
        cmd.extend(['--num_workers', str(self.config.get('num_workers', 4))])
        
        # 学习率
        if force_lr is not None:
            cmd.extend(['--force_learning_rate', str(force_lr)])
        else:
            cmd.extend(['--learning_rate', str(self.config.get('learning_rate', 1e-4))])
        
        # 检查点恢复
        if resume_checkpoint:
            cmd.extend(['--resume', resume_checkpoint])
        
        # 评估参数
        cmd.extend(['--eval_freq', str(self.config.get('eval_freq', 1))])
        cmd.extend(['--eval_max_frames', str(self.config.get('eval_max_frames', 96))])
        if self.config.get('skip_eval', False):
            cmd.append('--skip_eval')
        
        # 学习率调度器
        if self.config.get('use_scheduler', False):
            cmd.append('--use_scheduler')
            cmd.extend(['--scheduler_factor', str(self.config.get('scheduler_factor', 0.5))])
            cmd.extend(['--scheduler_patience', str(self.config.get('scheduler_patience', 3))])
            cmd.extend(['--scheduler_min_lr', str(self.config.get('scheduler_min_lr', 1e-6))])
        
        # EMA
        if self.config.get('use_ema', True):
            cmd.append('--use_ema')
            cmd.extend(['--ema_decay', str(self.config.get('ema_decay', 0.999))])
            if self.config.get('evaluate_both', True):
                cmd.append('--evaluate_both')
        
        # Stage 4 GOP优化
        if stage == 4 and self.config.get('stage4_gop_opt', False):
            cmd.append('--stage4_gop_opt')
        
        # 其他参数
        if self.config.get('grad_clip_max_norm'):
            cmd.extend(['--grad_clip_max_norm', str(self.config['grad_clip_max_norm'])])
        if self.config.get('compile', False):
            cmd.append('--compile')
        if self.config.get('seed'):
            cmd.extend(['--seed', str(self.config['seed'])])
        if self.config.get('mixed_precision'):
            cmd.extend(['--mixed_precision', self.config['mixed_precision']])
        if self.config.get('gradient_accumulation_steps', 1) > 1:
            cmd.extend(['--gradient_accumulation_steps', str(self.config['gradient_accumulation_steps'])])
        
        return cmd
    
    def run_training(self, stage: int, gop: int, 
                    start_epoch: int, total_epochs: int,
                    resume_checkpoint: Optional[str] = None,
                    force_lr: Optional[float] = None) -> bool:
        """
        执行单次训练
        
        Args:
            stage: 训练阶段
            gop: GOP大小
            start_epoch: 起始epoch（累计）
            total_epochs: 总epoch数
            resume_checkpoint: 恢复训练的检查点
            force_lr: 强制学习率
        
        Returns:
            是否成功
        """
        cmd = self.build_train_command(stage, gop, start_epoch, total_epochs, 
                                      resume_checkpoint, force_lr)
        
        print("=" * 100)
        print(f"Stage {stage}, GOP {gop}")
        print(f"Epochs: {start_epoch} → {start_epoch + total_epochs} (累计: {start_epoch + total_epochs})")
        if resume_checkpoint:
            print(f"Resume from: {resume_checkpoint}")
        else:
            print("Resume from: None (starting from scratch)")
        if force_lr:
            print(f"Force learning rate: {force_lr}")
        print("=" * 100)
        print(f"Command: {' '.join(cmd)}")
        print("=" * 100)
        print()
        
        # 执行训练命令
        try:
            result = subprocess.run(cmd, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"Error: Training failed with return code {e.returncode}")
            return False
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            return False
    
    def run_full_training(self, start_stage: int = 1, start_gop: int = 2):
        """
        执行完整的训练流程
        
        Args:
            start_stage: 起始阶段（用于断点续训）
            start_gop: 起始GOP大小（用于断点续训）
        """
        print("=" * 100)
        print("DCVC 完整训练流程")
        print("=" * 100)
        print(f"Lambda: {self.lambda_value}")
        print(f"Quality Index: {self.quality_index}")
        print(f"Epochs per GOP: {self.epochs_per_gop}")
        print(f"Checkpoint Dir: {self.checkpoint_dir}")
        print("=" * 100)
        print()
        
        # 确定起始位置
        start_stage_idx = self.stages.index(start_stage) if start_stage in self.stages else 0
        start_gop_idx = self.gop_sequence.index(start_gop) if start_gop in self.gop_sequence else 0
        
        # 计算起始累计epoch
        # 假设每个stage的每个GOP都训练epochs_per_gop个epoch
        self.current_epoch = 0
        for s in range(1, start_stage):
            self.current_epoch += len(self.gop_sequence) * self.epochs_per_gop
        for gop_idx in range(start_gop_idx):
            self.current_epoch += self.epochs_per_gop
        
        print(f"Starting from Stage {start_stage}, GOP {start_gop}")
        print(f"Initial cumulative epoch: {self.current_epoch}")
        print()
        
        # 遍历所有阶段
        for stage_idx, stage in enumerate(self.stages[start_stage_idx:], start=start_stage_idx):
            # 确定当前阶段的GOP起始位置
            gop_start_idx = start_gop_idx if stage_idx == start_stage_idx else 0
            
            # 遍历当前阶段的所有GOP
            for gop_idx, gop in enumerate(self.gop_sequence[gop_start_idx:], start=gop_start_idx):
                # 查找检查点
                resume_checkpoint = self.find_checkpoint(stage, gop)
                
                # 确定是否重置学习率（GOP切换时或stage切换时）
                force_lr = None
                if gop_idx > 0 or stage_idx > start_stage_idx:
                    force_lr = self.config.get('learning_rate', 1e-4)
                
                # 执行训练
                success = self.run_training(
                    stage=stage,
                    gop=gop,
                    start_epoch=self.current_epoch,
                    total_epochs=self.epochs_per_gop,
                    resume_checkpoint=resume_checkpoint,
                    force_lr=force_lr
                )
                
                if not success:
                    print(f"Training failed at Stage {stage}, GOP {gop}")
                    print("Please check the error messages above.")
                    return
                
                # 更新累计epoch
                self.current_epoch += self.epochs_per_gop
                
                print()
                print(f"✓ Completed Stage {stage}, GOP {gop}")
                print(f"  Cumulative epochs: {self.current_epoch}")
                print()
            
            # 重置GOP索引（下一个stage从GOP 2开始）
            start_gop_idx = 0
        
        print("=" * 100)
        print("✓ 所有训练阶段完成！")
        print("=" * 100)
        print(f"总累计epoch: {self.current_epoch}")
        print(f"最终检查点目录: {self.checkpoint_dir}")
        print("=" * 100)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径（JSON格式）
    
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def create_default_config(output_path: str):
    """
    创建默认配置文件模板
    
    Args:
        output_path: 输出配置文件路径
    """
    default_config = {
        # 数据集配置
        "train_video_dir": "/path/to/train/videos",
        "val_video_dir": "/path/to/val/videos",
        
        # RVQ-VQGAN I-frame 模型配置（本代理专用）
        "vqgan_config": "/path/to/rvq_vqgan_config.json",
        "vqgan_checkpoint": "/path/to/rvq_vqgan_checkpoint.pth",
        # 若希望直接用固定 I-frame 尺寸控制 BPP（单位：bytes），可设置为数值，默认 None 表示按模型 likelihood 计算
        "i_frame_size": None,
        
        # 训练参数
        "checkpoint_dir": "./checkpoints",
        "lambda_value": 2048,
        "quality_index": 4,
        "epochs_per_gop": 10,  # 每个GOP训练的epoch数
        "batch_size": 4,
        "crop_size": 256,
        "num_workers": 4,
        "learning_rate": 1e-4,
        
        # 评估参数
        "eval_freq": 1,  # 每N个epoch评估一次
        "eval_max_frames": 96,
        "skip_eval": False,
        
        # 学习率调度器
        "use_scheduler": False,
        "scheduler_factor": 0.5,
        "scheduler_patience": 3,
        "scheduler_min_lr": 1e-6,
        
        # EMA
        "use_ema": True,
        "ema_decay": 0.999,
        "evaluate_both": True,
        
        # Stage 4特殊选项
        "stage4_gop_opt": False,  # Stage 4是否使用GOP优化
        
        # 其他选项
        "grad_clip_max_norm": 1.0,
        "compile": False,
        "seed": 42,
        "mixed_precision": "no",  # "no", "fp16", "bf16"
        "gradient_accumulation_steps": 1
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)
    
    print(f"默认配置文件已创建: {output_path}")
    print("请编辑配置文件后重新运行训练代理。")


def main():
    parser = argparse.ArgumentParser(description='DCVC训练代理 - 自动执行完整训练流程')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径（JSON格式）')
    parser.add_argument('--start_stage', type=int, default=1, choices=[1, 2, 3, 4],
                       help='起始训练阶段（用于断点续训）')
    parser.add_argument('--start_gop', type=int, default=2, choices=[2, 3, 5, 7],
                       help='起始GOP大小（用于断点续训）')
    parser.add_argument('--create_config', type=str, default=None,
                       help='创建默认配置文件并退出')
    
    args = parser.parse_args()
    
    # 如果指定了创建配置文件，则创建并退出
    if args.create_config:
        create_default_config(args.create_config)
        return
    
    # 加载配置
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        print("使用 --create_config <path> 创建默认配置文件")
        return
    
    config = load_config(args.config)
    
    # 验证必需参数
    required_params = ['train_video_dir', 'checkpoint_dir', 'lambda_value', 'quality_index']
    for param in required_params:
        if param not in config:
            print(f"错误: 配置文件中缺少必需参数: {param}")
            return
    
    # 创建训练代理并执行
    proxy = TrainingProxy(config)
    proxy.run_full_training(start_stage=args.start_stage, start_gop=args.start_gop)


if __name__ == '__main__':
    main()
