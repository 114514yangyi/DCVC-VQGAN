import math
import os
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import Normalize
from torchvision.utils import make_grid
import logging
import math
import os
from typing import Tuple


# ========================================
# train_utils.py 文件详细注释
# ========================================
#
# 本文件为训练过程中常用的工具函数与辅助类的集合，适用于深度学习训练（如 VQ-VAE 等模型）。
#
# 结构与各部分功能说明如下：
#
# - 导入常用库（numpy, torch, torchvision、matplotlib、logging 等），以及Python标准库（如 os, datetime, math）。
#
# - get_model_size(model):
#     用于统计指定模型中可训练参数的总数，便于快速了解模型规模。
#
# - ProgressMeter:
#     进度条工具类，结合一组 AverageMeter 对象，格式化/打印每个 batch 的当前迭代进度和各项指标（如 loss, 准确率等）。
#
# - AverageMeter:
#     滑动统计类。维护和计算某项指标（如 loss、准确率）在训练或验证阶段的当前值、总和、平均值。用于方便地M记录和显示 batch/epoch 统计信息，是 PyTorch 训练脚本常规工具组件。
#
# - save_checkpoint(folder_name, state, filename):
#     保存训练过程中的模型/优化器/其它状态，可以实现断点续训、模型备份。具体实现略，文件用 os.path 拼接，通常与 torch.save 结合。
#
# - 其它（见文件剩余部分，未粘贴）可能还包含：
#     - 反标准化工具 NormalizeInverse：用于将归一化图像转换回原始像素空间，便于可视化。
#     - 模型架构分析 analyze_model_architecture：可将模型结构写入文件、计算参数量，辅助实验管理。
#     - 训练运行 id/name 生成 build_id_name：快速标记实验超参数组合。
#     - BadCaseDetector：用于自动识别训练过程中的 bad case（如 loss 异常激增的样本/批次），便于调试数据或分析模型问题。
#
# ========================================


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def save_checkpoint(folder_name, state, filename='checkpoint.pth.tar'):
    filename = os.path.join(folder_name, filename)
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, device_id=0):
    loc = 'cuda:{}'.format(device_id)
    checkpoint = torch.load(checkpoint_path, map_location=loc)
    return checkpoint


def train_visualize(unnormalize: torch.nn.Module, n_images: int, images: torch.Tensor,
                    image_recs: torch.Tensor) -> Tuple:
    images, recs = map(lambda t: unnormalize(t).detach().cpu(), (images, image_recs))
    images, recs = map(lambda t: make_grid(t.float(), nrow=int(math.sqrt(n_images))), (images, recs))

    return images, recs


def save_images(file_name: str, image: torch.Tensor) -> None:
    npimg = image.numpy()
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(file_name)


class NormalizeInverse(Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def analyze_model_architecture(model: torch.nn.Module, input_shape: tuple, output_file: str, logger: logging.Logger):
    """
    分析模型架构，记录各模块的输入输出形状和参数量
    
    Args:
        model: 要分析的模型
        input_shape: 输入张量的形状 (batch, channels, height, width)
        output_file: 输出文件路径
        logger: logger对象
    """
    logger.info("开始分析模型架构...")
    
    # 获取模型所在的设备
    device = next(model.parameters()).device
    logger.info(f"模型设备: {device}")
    
    # 存储模型信息
    model_info = []
    model_info.append("=" * 80)
    model_info.append("VQ-VAE2 模型架构详细分析")
    model_info.append("=" * 80)
    model_info.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    model_info.append(f"模型设备: {device}")
    model_info.append("")
    
    # 模型总体信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info.append("【模型总体信息】")
    model_info.append(f"总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    model_info.append(f"可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    model_info.append(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")
    model_info.append("")
    
    # 模型配置信息
    if hasattr(model, 'group_count'):
        model_info.append("【模型配置】")
        model_info.append(f"Group Count: {model.group_count}")
        model_info.append(f"Hidden Features (n_hid): {model.n_hid}")
        model_info.append(f"Codebook Dim (n_init): {model.n_init}")
        model_info.append(f"Blocks per Group: {model.n_blk_per_group}")
        model_info.append(f"Input Channels: {model.input_channels}")
        model_info.append(f"Output Channels: {model.output_channels}")
        model_info.append(f"Vocab Size: {model.vocab_size}")
        model_info.append(f"Sequence Length: {model.sequence_length}")
        model_info.append(f"Commitment Cost: {model.commitment_cost}")
        model_info.append(f"Decay (EMA): {model.decay}")
        model_info.append(f"Downsample: {model.downsample}")
        model_info.append(f"Upsample: {model.upsample}")
        model_info.append("")
    
    # 分析各个模块
    model_info.append("=" * 80)
    model_info.append("【模块详细分析】")
    model_info.append("=" * 80)
    model_info.append("")
    
    # 分析Encoder
    if hasattr(model, 'encoder'):
        model_info.append("【Encoder 模块】")
        encoder = model.encoder
        encoder_params = sum(p.numel() for p in encoder.parameters())
        model_info.append(f"参数量: {encoder_params:,} ({encoder_params / 1e6:.2f}M)")
        
        # 尝试前向传播获取输入输出形状
        # VQ-VAE2的Encoder期望输入格式: (batch * sequence_length, channels, height, width)
        try:
            if hasattr(model, 'sequence_length'):
                seq_len = model.sequence_length
                # 使用最小的batch size (1) 和 sequence_length
                test_batch = 1
                test_input_shape = (test_batch * seq_len, input_shape[1], input_shape[2], input_shape[3])
                dummy_input = torch.randn(*test_input_shape, device=device)
                with torch.no_grad():
                    encoder_output = encoder(dummy_input)
                model_info.append(f"输入形状: {dummy_input.shape} (batch*seq={test_batch}*{seq_len}, channels, h, w)")
                model_info.append(f"输出形状: {encoder_output.shape}")
            else:
                # 如果没有sequence_length属性，使用原始方法
                dummy_input = torch.randn(1, *input_shape[1:], device=device)
                with torch.no_grad():
                    encoder_output = encoder(dummy_input)
                model_info.append(f"输入形状: {dummy_input.shape}")
                model_info.append(f"输出形状: {encoder_output.shape}")
        except Exception as e:
            model_info.append(f"无法获取输入输出形状: {str(e)}")
        
        # 分析encoder的子模块
        if hasattr(encoder, 'blocks_spatial'):
            model_info.append("\nEncoder 空间编码子模块结构:")
            for name, module in encoder.blocks_spatial.named_children():
                module_params = sum(p.numel() for p in module.parameters())
                model_info.append(f"  - {name}: {type(module).__name__}, 参数量: {module_params:,}")
        if hasattr(encoder, 'blocks_tempo'):
            model_info.append("\nEncoder 时间编码子模块结构:")
            for name, module in encoder.blocks_tempo.named_children():
                module_params = sum(p.numel() for p in module.parameters())
                model_info.append(f"  - {name}: {type(module).__name__}, 参数量: {module_params:,}")
        model_info.append("")
    
    # 分析VQ模块
    # 支持不同的模型类型：传统模型使用 vq_vae，taming 模型使用 quantize
    vq = None
    if hasattr(model, 'vq_vae') and model.vq_vae is not None:
        vq = model.vq_vae
    elif hasattr(model, 'quantize') and model.quantize is not None:
        # Taming 模型使用 quantize 属性
        vq = model.quantize
    elif hasattr(model, '_get_model'):
        # 适配器模型，尝试访问底层模型
        try:
            underlying_model = model._get_model()
            if hasattr(underlying_model, 'quantize') and underlying_model.quantize is not None:
                vq = underlying_model.quantize
            elif hasattr(underlying_model, 'vq_vae') and underlying_model.vq_vae is not None:
                vq = underlying_model.vq_vae
        except:
            pass
    
    if vq is not None:
        model_info.append("【Vector Quantizer 模块】")
        # 处理 ModuleList 的情况（VQVAE 模型使用 codebooks ModuleList）
        if isinstance(vq, torch.nn.ModuleList):
            vq_params = sum(p.numel() for codebook in vq for p in codebook.parameters())
            model_info.append(f"参数量: {vq_params:,} ({vq_params / 1e6:.2f}M)")
            model_info.append(f"Codebook层级数: {len(vq)}")
            for i, codebook in enumerate(vq):
                if hasattr(codebook, 'n_embed'):
                    model_info.append(f"  层级 {i}: Codebook大小={codebook.n_embed}, Embedding维度={codebook.dim}")
        else:
            # 对于 EMA 量化器，可能没有可训练参数（码本通过 EMA 更新）
            try:
                vq_params = sum(p.numel() for p in vq.parameters())
                model_info.append(f"参数量: {vq_params:,} ({vq_params / 1e6:.2f}M)")
            except:
                model_info.append("参数量: 0 (EMA 量化器，码本通过指数移动平均更新，无梯度参数)")
            
            # 尝试获取码本大小和嵌入维度（不同量化器可能使用不同的属性名）
            codebook_size = None
            embed_dim = None
            
            # 尝试不同的属性名
            for attr_name in ['n_e', 'n_embed', 'num_embeddings', 'num_tokens', 're_embed']:
                if hasattr(vq, attr_name):
                    codebook_size = getattr(vq, attr_name)
                    break
            
            for attr_name in ['e_dim', 'embed_dim', 'embedding_dim', 'dim']:
                if hasattr(vq, attr_name):
                    embed_dim = getattr(vq, attr_name)
                    break
            
            if codebook_size is not None:
                model_info.append(f"Codebook大小: {codebook_size}")
            if embed_dim is not None:
                model_info.append(f"Embedding维度: {embed_dim}")
            
            # 显示量化器类型
            model_info.append(f"量化器类型: {type(vq).__name__}")
        model_info.append("")
    
    # 分析Decoder
    if hasattr(model, 'decoder'):
        model_info.append("【Decoder 模块】")
        decoder = model.decoder
        decoder_params = sum(p.numel() for p in decoder.parameters())
        model_info.append(f"参数量: {decoder_params:,} ({decoder_params / 1e6:.2f}M)")
        
        # 尝试获取输入输出形状
        # VQ-VAE2的Decoder期望输入格式: (batch, n_init, height, width)
        try:
            if hasattr(model, 'encoder') and hasattr(model, 'sequence_length'):
                seq_len = model.sequence_length
                # 先通过encoder获取正确的输出形状
                test_batch = 1
                test_input_shape = (test_batch * seq_len, input_shape[1], input_shape[2], input_shape[3])
                dummy_input = torch.randn(*test_input_shape, device=device)
                with torch.no_grad():
                    encoder_out = model.encoder(dummy_input)
                    # Decoder的输入是encoder的输出: (batch, n_init, h, w)
                    decoder_output = decoder(encoder_out)
                model_info.append(f"Encoder输出(Decoder输入): {encoder_out.shape} (batch, n_init, h, w)")
                model_info.append(f"Decoder输出: {decoder_output.shape} (batch*seq, channels, h, w)")
            else:
                # 如果没有sequence_length属性，尝试直接使用n_init
                if hasattr(model, 'n_init'):
                    # 估计空间维度（根据下采样）
                    estimated_h, estimated_w = input_shape[2], input_shape[3]
                    if hasattr(model, 'group_count') and hasattr(model, 'downsample') and model.downsample:
                        downsample_factor = 2 ** (model.group_count - 1)
                        estimated_h //= downsample_factor
                        estimated_w //= downsample_factor
                    decoder_input = torch.randn(1, model.n_init, estimated_h, estimated_w, device=device)
                    with torch.no_grad():
                        decoder_output = decoder(decoder_input)
                    model_info.append(f"输入形状: {decoder_input.shape}")
                    model_info.append(f"输出形状: {decoder_output.shape}")
                else:
                    model_info.append("无法确定Decoder输入形状（缺少必要属性）")
        except Exception as e:
            model_info.append(f"无法获取输入输出形状: {str(e)}")
        
        # 分析decoder的子模块
        if hasattr(decoder, 'blocks_tempo'):
            model_info.append("\nDecoder 时间解码子模块结构:")
            for name, module in decoder.blocks_tempo.named_children():
                module_params = sum(p.numel() for p in module.parameters())
                model_info.append(f"  - {name}: {type(module).__name__}, 参数量: {module_params:,}")
        if hasattr(decoder, 'blocks_spatial'):
            model_info.append("\nDecoder 空间解码子模块结构:")
            for name, module in decoder.blocks_spatial.named_children():
                module_params = sum(p.numel() for p in module.parameters())
                model_info.append(f"  - {name}: {type(module).__name__}, 参数量: {module_params:,}")
        model_info.append("")
    
    # 完整的前向传播流程
    model_info.append("=" * 80)
    model_info.append("【前向传播流程】")
    model_info.append("=" * 80)
    try:
        # VQ-VAE2的输入格式: (batch * sequence_length, channels, height, width)
        # input_shape 已经是正确的格式，直接使用
        dummy_input = torch.randn(*input_shape, device=device)
        model_info.append(f"1. 模型输入: {dummy_input.shape} (batch*seq, channels, h, w)")
        
        with torch.no_grad():
            # 检查 vq_vae 是否是 ModuleList（某些模型使用 ModuleList 存储多个 codebook）
            vq_is_modulelist = hasattr(model, 'vq_vae') and isinstance(model.vq_vae, torch.nn.ModuleList)
            
            if vq_is_modulelist:
                # 对于使用 ModuleList 的模型，直接使用模型的 forward 方法
                # 这样可以正确处理多层级量化
                model_info.append("检测到多层级 VQ 模块（ModuleList），使用模型 forward 方法")
                output = model(dummy_input)
                if isinstance(output, tuple):
                    if len(output) >= 2:
                        vq_loss = output[0] if isinstance(output[0], torch.Tensor) else None
                        x_recon = output[1]
                        model_info.append(f"2. 模型输出 (x_recon): {x_recon.shape} (batch*seq, channels, h, w)")
                        if vq_loss is not None:
                            model_info.append(f"   VQ Loss: {vq_loss.item():.6f}")
                    else:
                        x_recon = output[0]
                        model_info.append(f"2. 模型输出 (x_recon): {x_recon.shape} (batch*seq, channels, h, w)")
                else:
                    x_recon = output
                    model_info.append(f"2. 模型输出 (x_recon): {x_recon.shape} (batch*seq, channels, h, w)")
                model_info.append("3. 注意: 多层级 VQ 模型的内部流程由模型 forward 方法处理")
            else:
                # 对于单层级 VQ 模型，可以逐步测试
                if hasattr(model, 'encoder'):
                    # Encoder: (batch*seq, channels, h, w) -> (batch, n_init, h', w')
                    z = model.encoder(dummy_input)
                    model_info.append(f"2. Encoder输出 (z): {z.shape} (batch, n_init, h', w')")
                    
                    # VQ: (batch, n_init, h', w') -> (batch, n_init, h', w')
                    if hasattr(model, 'vq_vae') and callable(model.vq_vae):
                        vq_output = model.vq_vae(z)
                        if isinstance(vq_output, tuple):
                            vq_loss, quantized = vq_output[0], vq_output[1]
                            if len(vq_output) > 2:
                                perplexity = vq_output[2]
                                model_info.append(f"   Perplexity: {perplexity.item():.6f}")
                        else:
                            quantized = vq_output
                            vq_loss = None
                        model_info.append(f"3. VQ量化后 (quantized): {quantized.shape} (batch, n_init, h', w')")
                        if vq_loss is not None:
                            model_info.append(f"   VQ Loss: {vq_loss.item():.6f}")
                        
                        # Decoder: (batch, n_init, h', w') -> (batch*seq, channels, h, w)
                        if hasattr(model, 'decoder'):
                            x_recon = model.decoder(quantized)
                            model_info.append(f"4. Decoder输出 (x_recon): {x_recon.shape} (batch*seq, channels, h, w)")
                        else:
                            x_recon = quantized
                    else:
                        # 如果没有 vq_vae 或不可调用，直接使用模型 forward
                        output = model(dummy_input)
                        if isinstance(output, tuple):
                            x_recon = output[1] if len(output) > 1 else output[0]
                        else:
                            x_recon = output
                        model_info.append(f"3. 模型输出 (x_recon): {x_recon.shape} (batch*seq, channels, h, w)")
                else:
                    # 如果没有 encoder，直接使用模型 forward
                    output = model(dummy_input)
                    if isinstance(output, tuple):
                        x_recon = output[1] if len(output) > 1 else output[0]
                    else:
                        x_recon = output
                    model_info.append(f"2. 模型输出 (x_recon): {x_recon.shape} (batch*seq, channels, h, w)")
            
            model_info.append("")
            model_info.append("注意: VQ-VAE2处理视频序列，输入和输出都是 (batch*sequence_length, channels, h, w) 格式")
    except Exception as e:
        model_info.append(f"无法执行前向传播测试: {str(e)}")
        logger.warning(f"前向传播测试失败: {str(e)}")
    
    model_info.append("")
    model_info.append("=" * 80)
    model_info.append("分析完成")
    model_info.append("=" * 80)
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(model_info))
    
    logger.info(f"模型架构分析完成，详细信息已保存到: {output_file}")
    logger.debug('\n'.join(model_info))


def build_id_name(params: dict) -> str:
    # 使用学习率+vocab_size+n_hid+n_init+commitment_cost+decay+steps+batch_size参数来构造
    id_name = ''
    # INSERT_YOUR_CODE
    model_args = params.get('model_args', {})
    data_args = params.get('data_args', {})
    train_args = params.get('train_args', {})

    # 获取超参数，使用默认值防止缺失
    lr = train_args.get('lr', 0) or params.get('lr', 0)
    lr_decay = train_args.get('lr_decay', 0) or params.get('lr_decay', 0)
    vocab_size = model_args.get('vocab_size', 0)
    n_hid = model_args.get('n_hid', 0)
    n_init = model_args.get('n_init', 0)
    commitment_cost = model_args.get('commitment_cost', 0)
    decay = model_args.get('decay', 0)
    num_steps = train_args.get('num_steps', 0) or params.get('num_steps', 0)
    batch_size = data_args.get('batch_size', 0)
    default_gpu = train_args.get('default_gpu', 0) or params.get('default_gpu', 0)
    model_type = params.get('model_type', '')
    use_gan = train_args.get('use_gan', False) or params.get('use_gan', False)
    time_state = params.get('time', '')
    scaling_rates = params.get('scaling_rates', [])
    # 组装字符串，易读简洁
    id_name = (
        f"lr{lr}_vocab{vocab_size}_hid{n_hid}_init{n_init}"
        f"_bs{batch_size}"
        f"_gpu{default_gpu.split(':')[-1]}"
        f"_model{model_type}"
        f"_use_gan{use_gan}"
        f"_time{time_state}"
        f"_scaling_rates{'_'.join(str(rate) for rate in scaling_rates)}"
    )
    return id_name

from collections import defaultdict, deque
from typing import List

class BadCaseDetector:
    """
    检测训练过程中的 bad case（loss 突然增大的 step）
    
    算法：
    - 维护最近 N 个 step 的全局 loss 历史（滑动窗口）
    - 计算当前 step 的 loss 与最近 N 个 step 平均 loss 的比值
    - 如果当前 loss > 历史平均 * threshold，记录当前 step 的所有视频路径
    """
    
    def __init__(self, bad_case_file: str = "bad_case.txt", 
                 threshold: float = 1.5, 
                 window_size: int = 20,
                 min_samples: int = 3):
        """
        Args:
            bad_case_file: 记录 bad case 的文件路径
            threshold: loss 增大的阈值倍数（默认 1.5，即当前 loss 是历史平均的 1.5 倍）
            window_size: 滑动窗口大小，保留最近 N 个 step 的 loss（默认 20）
            min_samples: 最少需要多少个历史样本才开始检测
        """
        self.bad_case_file = bad_case_file
        self.threshold = threshold
        self.window_size = window_size
        self.min_samples = min_samples
        # 全局 loss 历史（每个 step 的平均 loss）
        self.step_loss_history = deque(maxlen=window_size)
        # 已记录的 bad case step，避免重复记录
        self.recorded_steps = set()
        # 确保文件存在
        if not os.path.exists(bad_case_file):
            with open(bad_case_file, 'w', encoding='utf-8') as f:
                f.write(f"# Bad Case Detection Log\n")
                f.write(f"# Threshold: {threshold}x, Window Size: {window_size}, Min Samples: {min_samples}\n")
                f.write(f"# Format: step | video_path | current_loss | avg_loss | ratio\n\n")
    
    def update(self, step: int, video_paths: List[str], current_loss: float):
        """
        更新 step 的 loss 记录，并检测 bad case
        
        Args:
            step: 当前训练步数
            video_paths: 当前 batch 的视频路径列表
            current_loss: 当前 step 的平均 loss（整个 batch 的平均值）
        
        Returns:
            bad_cases: 如果是 bad case，返回视频路径列表；否则返回空列表
        """
        # 如果历史样本不足，只记录 loss，不检测
        if len(self.step_loss_history) < self.min_samples:
            self.step_loss_history.append(current_loss)
            return []
        
        # 计算最近 N 个 step 的平均 loss
        avg_loss = sum(self.step_loss_history) / len(self.step_loss_history)
        
        # 检测 bad case：当前 loss 是否显著大于历史平均
        bad_cases = []
        if avg_loss > 0:
            ratio = current_loss / avg_loss
            if ratio > self.threshold:
                # 避免重复记录同一个 step
                if step not in self.recorded_steps:
                    self.recorded_steps.add(step)
                    # 记录当前 step 的所有视频路径
                    bad_cases = video_paths
                    
                    # 写入 bad case 文件
                    with open(self.bad_case_file, 'a', encoding='utf-8') as f:
                        for video_path in video_paths:
                            video_name = os.path.basename(video_path)
                            f.write(
                                f"{step} | {video_path} | "
                                f"loss={current_loss:.6f} | "
                                f"avg={avg_loss:.6f} | "
                                f"ratio={ratio:.2f}x\n"
                            )
        
        # 更新历史记录
        self.step_loss_history.append(current_loss)
        
        return bad_cases
