#!/usr/bin/env python3
"""
Transformer 模型测试脚本

主要测试：
1. Transformer 在相同输入下的输出一致性
2. 逐步预测 vs 一次性预测的概率一致性
3. 使用 numpyAc 进行压缩和解压的一致性
"""


import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


# import torch
# import numpy as np
import random

def set_deterministic(seed=42):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # 设置CUDA确定性选项
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁用自动优化
    
    # 设置环境变量
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

set_deterministic(42)


# 添加项目根目录到 Python 路径
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from transformer.train_transformer import TransformerEntropyModel
from transformer.encode_methods import (
    compress_arithmetic_coding,
    decompress_arithmetic_coding_incremental
)

PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = PYTORCH_CUDA_ALLOC_CONF


def load_transformer_model(ckpt_path: str, device: str):
    """加载 Transformer 模型"""
    print(f"加载 Transformer 模型: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 从 checkpoint 获取参数
    args = checkpoint.get('args', {})
    vocab_size = args.get('vocab_size', 1024)
    seq_len = args.get('seq_len', 4096)
    d_model = args.get('d_model', 256)
    num_layers = args.get('num_layers', 8)
    
    # 创建模型
    model = TransformerEntropyModel(
        num_codes=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        max_seq_len=seq_len
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型参数:")
    print(f"  vocab_size (num_codes): {vocab_size}")
    print(f"  seq_len (max_seq_len): {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  num_layers: {num_layers}")
    
    return model, vocab_size, seq_len


def test_consistency(model, device, seq_len=4096, vocab_size=1024, num_tests=3):
    """
    测试1: Transformer 输出一致性
    
    测试内容：
    1. 相同输入下，Transformer 的输出是否相同
    2. 逐步预测 vs 一次性预测的概率是否一致
    """
    print("\n" + "="*60)
    print("测试1: Transformer 输出一致性")
    print("="*60)
    
    model.eval()
    bos_token_id = model.transformer.bos_token_id
    
    with torch.no_grad():
        for test_idx in range(num_tests):
            print(f"\n--- 测试 {test_idx + 1}/{num_tests} ---")
            
            # 生成模拟的 VQGAN index 序列
            indices = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long, device=device)
            print(f"生成的序列长度: {seq_len}")
            print(f"索引范围: [{indices.min().item()}, {indices.max().item()}]")
            print(f"前10个索引: {indices[:10].cpu().numpy()}")
            
            # 添加 batch 维度和 BOS token
            indices_with_bos = torch.cat([
                torch.tensor([bos_token_id], dtype=torch.long, device=device),
                indices
            ]).unsqueeze(0)  # (1, seq_len + 1)
            
            # ========== 测试1.1: 相同输入下的输出一致性 ==========
            print("\n[测试1.1] 相同输入下的输出一致性")
            
            # 第一次前向传播
            mask1 = torch.triu(torch.ones(seq_len + 1, seq_len + 1), diagonal=1).to(device)
            logits1, probs1 = model.transformer(indices_with_bos, mask=mask1, use_rope=True)
            
            # 第二次前向传播（相同输入）
            mask2 = torch.triu(torch.ones(seq_len + 1, seq_len + 1), diagonal=1).to(device)
            logits2, probs2 = model.transformer(indices_with_bos, mask=mask2, use_rope=True)
            
            # 比较结果
            logits_diff = torch.abs(logits1 - logits2).max().item()
            probs_diff = torch.abs(probs1 - probs2).max().item()
            
            print(f"  Logits 最大差异: {logits_diff:.2e}")
            print(f"  Probabilities 最大差异: {probs_diff:.2e}")
            
            if logits_diff < 1e-6 and probs_diff < 1e-6:
                print("  ✓ 通过：相同输入下输出完全一致")
            else:
                print("  ✗ 失败：相同输入下输出不一致")
            
            # ========== 测试1.2: 完整序列 vs 后一半masked序列 ==========
            print("\n[测试1.2] 完整序列 vs 后一半masked序列的概率对比")
            
            # 使用完整的 index 序列
            indices_full = indices_with_bos  # (1, seq_len + 1)
            mask_full = torch.triu(torch.ones(seq_len + 1, seq_len + 1), diagonal=1).to(device)
            _, probs_full = model.transformer(indices_full, mask=mask_full, use_rope=True)
            probs_full_output = probs_full[0]
            
            # 将 index 序列的后一半设为 0
            indices_masked = indices_with_bos.clone()
            half_point = (seq_len + 1) // 2  # 后一半的起始位置（包含BOS）
            indices_masked[0, half_point:] = 0  # 将后一半设为0
            mask_masked = torch.triu(torch.ones(seq_len + 1, seq_len + 1), diagonal=1).to(device)
            _, probs_masked = model.transformer(indices_masked, mask=mask_masked, use_rope=True)
            probs_masked_output = probs_masked[0]
            
            # 打印对比信息
            print(f"  完整序列长度: {seq_len + 1} (包含BOS)")
            print(f"  Masked位置: 从位置 {half_point} 开始（共 {seq_len + 1 - half_point} 个位置）")
            print(f"  完整序列前10个索引: {indices_full[0, :10].cpu().numpy()}")
            print(f"  Masked序列前10个索引: {indices_masked[0, :10].cpu().numpy()}")
            print(f"  Masked序列后10个索引: {indices_masked[0, -10:].cpu().numpy()}")
            
            print(f"mask 前:{probs_full_output[half_point-1]}")
            print(f"mask 后:{probs_masked_output[half_point-1]}")
            
            # ========== 测试1.3: 逐步预测 vs 一次性预测 ==========
            print("\n[测试1.3] 逐步预测 vs 一次性预测的概率一致性")



            
            
            # 一次性预测整个序列（排除 BOS）
            one_shot_probs = probs1[0,:-1,:]  # (seq_len, vocab_size)，排除 BOS 位置
            
            # 逐步预测：一开始 context 就是完整的序列（seq_len=4096），然后逐步替换为真实 index
            # 初始化 context：BOS + 占位符（使用0或随机值，后续会被真实值替换）
            # 为了测试一致性，我们使用随机初始值，然后逐步替换为真实值
            placeholder_indices = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long, device=device)
            current_context = torch.cat([
                torch.tensor([bos_token_id], dtype=torch.long, device=device),
                placeholder_indices
            ])  # (seq_len + 1,)
            
            step_by_step_probs = []
            
            for pos in tqdm(range(seq_len)):
                # 预测当前位置（pos+1，因为第0位是BOS）的下一个 token 的概率
                # 注意：get_conditional_probs 会预测最后一个位置的概率
                # 所以我们需要传入 [BOS, ...已替换的真实值..., 当前位置的占位符, ...后续占位符...]
                # 但实际上，我们应该预测当前位置的概率，所以传入 [BOS, ...已替换的真实值...]
                
                # 获取当前 context 的前 pos+1 个位置（BOS + 前 pos 个真实值）
                context_for_prediction = current_context[:pos+1]  # (pos+1,)
                
                # 预测下一个 token 的概率
                next_probs = model.transformer.get_conditional_probs_by_index(
                    context_for_prediction,
                    pos,
                    temperature=1.0,
                    use_rope=True
                )  # (vocab_size,)
                step_by_step_probs.append(next_probs)
                
                # 将当前位置的占位符替换为真实 index
                current_context[pos + 1] = indices[pos]  # pos+1 因为第0位是BOS
            
            # 合并逐步预测的结果
            step_by_step_probs_tensor = torch.stack(step_by_step_probs, dim=0)  # (seq_len, vocab_size)
            
            
            
            # 比较结果
            probs_diff_step = torch.abs(one_shot_probs - step_by_step_probs_tensor).max().item()
            probs_diff_mean = torch.abs(one_shot_probs - step_by_step_probs_tensor).mean().item()
            
            print(f"  逐步预测 vs 一次性预测的最大差异: {probs_diff_step:.2e}")
            print(f"  逐步预测 vs 一次性预测的平均差异: {probs_diff_mean:.2e}")
            
            print(step_by_step_probs_tensor)
            print(one_shot_probs)
            # 检查前几个位置的差异
            print(f"  前5个位置的最大差异:")
            for i in range(min(5, seq_len)):
                pos_diff = torch.abs(one_shot_probs[i] - step_by_step_probs_tensor[i]).max().item()
                print(f"    位置 {i}: {pos_diff:.2e}")
            
            if probs_diff_step < 1e-5:
                print("  ✓ 通过：逐步预测和一次性预测的概率基本一致")
            elif probs_diff_step < 1e-3:
                print("  ⚠ 警告：逐步预测和一次性预测的概率有较小差异（可能是数值误差）")
            else:
                print("  ✗ 失败：逐步预测和一次性预测的概率差异较大")
            
            # 检查预测的 argmax 是否一致
            one_shot_pred = one_shot_probs.argmax(dim=-1)  # (seq_len,)
            step_by_step_pred = step_by_step_probs_tensor.argmax(dim=-1)  # (seq_len,)
            pred_match = (one_shot_pred == step_by_step_pred).sum().item()
            pred_match_ratio = pred_match / seq_len
            
            print(f"  预测 argmax 一致的位置数: {pred_match}/{seq_len} ({pred_match_ratio*100:.2f}%)")
            
            if pred_match_ratio > 0.95:
                print("  ✓ 通过：预测的 argmax 基本一致")
            else:
                print("  ⚠ 警告：预测的 argmax 有较多不一致")
            
            # ========== 测试1.4: 两次相同的逐步预测是否一致 ==========
            print("\n[测试1.4] 两次相同的逐步预测是否一致")
            
            # 第一次逐步预测
            placeholder_indices_1 = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long, device=device)
            current_context_1 = torch.cat([
                torch.tensor([bos_token_id], dtype=torch.long, device=device),
                placeholder_indices_1
            ])  # (seq_len + 1,)
            
            step_by_step_probs_1 = []
            
            for pos in tqdm(range(seq_len), desc="第一次逐步预测"):
                context_for_prediction_1 = current_context_1[:pos+1]  # (pos+1,)
                
                next_probs_1 = model.transformer.get_conditional_probs_by_index(
                    context_for_prediction_1,
                    pos,
                    temperature=1.0,
                    use_rope=True
                )  # (vocab_size,)
                step_by_step_probs_1.append(next_probs_1)
                
                # 将当前位置的占位符替换为真实 index
                current_context_1[pos + 1] = indices[pos]  # pos+1 因为第0位是BOS
            
            step_by_step_probs_tensor_1 = torch.stack(step_by_step_probs_1, dim=0)  # (seq_len, vocab_size)
            
            # 第二次逐步预测（使用相同的输入和相同的占位符初始化）
            placeholder_indices_2 = placeholder_indices_1.clone()  # 使用相同的占位符
            current_context_2 = torch.cat([
                torch.tensor([bos_token_id], dtype=torch.long, device=device),
                placeholder_indices_2
            ])  # (seq_len + 1,)
            
            step_by_step_probs_2 = []
            
            for pos in tqdm(range(seq_len), desc="第二次逐步预测"):
                context_for_prediction_2 = current_context_2[:pos+1]  # (pos+1,)
                
                next_probs_2 = model.transformer.get_conditional_probs_by_index(
                    context_for_prediction_2,
                    pos,
                    temperature=1.0,
                    use_rope=True
                )  # (vocab_size,)
                step_by_step_probs_2.append(next_probs_2)
                
                # 将当前位置的占位符替换为真实 index（使用相同的真实值）
                current_context_2[pos + 1] = indices[pos]  # pos+1 因为第0位是BOS
            
            step_by_step_probs_tensor_2 = torch.stack(step_by_step_probs_2, dim=0)  # (seq_len, vocab_size)
            
            # 比较两次逐步预测的结果
            probs_diff_twice = torch.abs(step_by_step_probs_tensor_1 - step_by_step_probs_tensor_2).max().item()
            probs_diff_twice_mean = torch.abs(step_by_step_probs_tensor_1 - step_by_step_probs_tensor_2).mean().item()
            
            print(f"  两次逐步预测的最大差异: {probs_diff_twice:.2e}")
            print(f"  两次逐步预测的平均差异: {probs_diff_twice_mean:.2e}")
            
            # 检查前几个位置的差异
            print(f"  前5个位置的最大差异:")
            for i in range(min(5, seq_len)):
                pos_diff = torch.abs(step_by_step_probs_tensor_1[i] - step_by_step_probs_tensor_2[i]).max().item()
                print(f"    位置 {i}: {pos_diff:.2e}")
            
            if probs_diff_twice < 1e-6:
                print("  ✓ 通过：两次逐步预测的结果完全一致")
            elif probs_diff_twice < 1e-5:
                print("  ✓ 通过：两次逐步预测的结果基本一致（可能是数值误差）")
            elif probs_diff_twice < 1e-3:
                print("  ⚠ 警告：两次逐步预测的结果有较小差异")
            else:
                print("  ✗ 失败：两次逐步预测的结果差异较大")
            
            # 检查预测的 argmax 是否一致
            pred_1 = step_by_step_probs_tensor_1.argmax(dim=-1)  # (seq_len,)
            pred_2 = step_by_step_probs_tensor_2.argmax(dim=-1)  # (seq_len,)
            pred_match_twice = (pred_1 == pred_2).sum().item()
            pred_match_ratio_twice = pred_match_twice / seq_len
            
            print(f"  两次预测 argmax 一致的位置数: {pred_match_twice}/{seq_len} ({pred_match_ratio_twice*100:.2f}%)")
            
            if pred_match_ratio_twice == 1.0:
                print("  ✓ 通过：两次预测的 argmax 完全一致")
            elif pred_match_ratio_twice > 0.99:
                print("  ✓ 通过：两次预测的 argmax 基本一致")
            else:
                print("  ⚠ 警告：两次预测的 argmax 有较多不一致")


def test_compression_decompression(model, device, seq_len=4096, vocab_size=1024, num_tests=3):
    """
    测试2: numpyAc 压缩和解压一致性
    
    测试内容：
    使用 numpyAc 库，对同一个输入进行压缩和解压，验证结果是否相同
    """
    print("\n" + "="*60)
    print("测试2: numpyAc 压缩和解压一致性")
    print("="*60)
    
    model.eval()
    bos_token_id = model.transformer.bos_token_id
    
    with torch.no_grad():
        for test_idx in range(num_tests):
            print(f"\n--- 测试 {test_idx + 1}/{num_tests} ---")
            
            # 生成模拟的 VQGAN index 序列
            original_indices = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long, device=device)
            print(f"原始序列长度: {seq_len}")
            print(f"索引范围: [{original_indices.min().item()}, {original_indices.max().item()}]")
            print(f"前10个索引: {original_indices[:10].cpu().numpy()}")
            
            # ========== 步骤1: 压缩 ==========
            print("\n[步骤1] 压缩序列")
            
            # 添加 batch 维度
            indices_batch = original_indices.unsqueeze(0)  # (1, seq_len)
            
            # 使用 Transformer 获取概率分布
            # 注意：需要添加 BOS token 来获取概率分布
            indices_with_bos = torch.cat([
                torch.tensor([bos_token_id], dtype=torch.long, device=device),
                original_indices
            ]).unsqueeze(0)  # (1, seq_len + 1)
            
            mask = torch.triu(torch.ones(seq_len + 1, seq_len + 1), diagonal=1).to(device)
            _, probs_full = model.transformer(indices_with_bos, mask=mask, use_rope=True)
            
            # 排除 BOS 位置的概率
            probs = probs_full[0, :-1, :].unsqueeze(0)  # (1, seq_len, vocab_size)
            
            # 使用 numpyAc 压缩
            from transformer.encode_methods import compress_arithmetic_coding
            actual_vocab_size = probs.shape[-1]
            compressed_bytes, metadata = compress_arithmetic_coding(
                indices_batch,
                probs,
                vocab_size=actual_vocab_size,
                store_pmf=False
            )
            
            compressed_size = len(compressed_bytes)
            print(f"  压缩大小: {compressed_size} bytes")
            print(f"  压缩率: {compressed_size * 8 / seq_len:.4f} bits/index")
            
            # ========== 步骤2: 解压 ==========
            print("\n[步骤2] 解压序列")
            
            # 使用逐步解码（因为 store_pmf=False）
            decompressed_indices = decompress_arithmetic_coding_incremental(
                compressed_bytes,
                metadata,
                model,
                device,
                original_indices=original_indices  # 用于验证
            )
            
            print(f"  解压序列长度: {len(decompressed_indices)}")
            print(f"  解压前10个索引: {decompressed_indices[:10].cpu().numpy()}")
            
            # ========== 步骤3: 验证一致性 ==========
            print("\n[步骤3] 验证压缩/解压一致性")
            
            # 比较原始和解压的索引
            if len(decompressed_indices) != len(original_indices):
                print(f"  ✗ 失败：长度不匹配 ({len(decompressed_indices)} != {len(original_indices)})")
                continue
            
            # 转换为 CPU 进行比较
            original_cpu = original_indices.cpu()
            decompressed_cpu = decompressed_indices.cpu()
            
            # 检查是否完全一致
            is_equal = torch.equal(original_cpu, decompressed_cpu)
            
            if is_equal:
                print("  ✓ 通过：压缩和解压后的序列完全一致")
            else:
                # 统计差异
                diff_mask = (original_cpu != decompressed_cpu)
                num_diff = diff_mask.sum().item()
                diff_ratio = num_diff / seq_len
                
                print(f"  ✗ 失败：压缩和解压后的序列不一致")
                print(f"    不一致的位置数: {num_diff}/{seq_len} ({diff_ratio*100:.2f}%)")
                
                # 显示前几个不一致的位置
                diff_positions = torch.nonzero(diff_mask, as_tuple=False)
                print(f"    前10个不一致的位置:")
                for idx, pos in enumerate(diff_positions[:10]):
                    pos_val = pos.item()
                    orig_val = original_cpu[pos_val].item()
                    decomp_val = decompressed_cpu[pos_val].item()
                    print(f"      位置 {pos_val}: 原始={orig_val}, 解压={decomp_val}")
                
                # 统计差异值的分布
                diff_values_orig = original_cpu[diff_mask]
                diff_values_decomp = decompressed_cpu[diff_mask]
                if len(diff_values_orig) > 0:
                    print(f"    差异值统计:")
                    print(f"      原始值范围: [{diff_values_orig.min().item()}, {diff_values_orig.max().item()}]")
                    print(f"      解压值范围: [{diff_values_decomp.min().item()}, {diff_values_decomp.max().item()}]")
                    print(f"      平均差异: {(diff_values_orig.float() - diff_values_decomp.float()).abs().mean().item():.4f}")


def main():
    parser = argparse.ArgumentParser(description='Transformer 模型测试')
    parser.add_argument('--transformer_ckpt', type=str, 
                       default="/home/huyang/VqVaeVideo-master/VqVaeVideo-master/transformer/outputs/best_model.pth",
                       help='Transformer checkpoint 路径')
    parser.add_argument('--device', type=str, default='cuda:2',
                       help='设备')
    parser.add_argument('--seq_len', type=int, default=100,
                       help='测试序列长度')
    parser.add_argument('--vocab_size', type=int, default=1024,
                       help='词汇表大小（如果 checkpoint 中没有，使用此值）')
    parser.add_argument('--num_tests', type=int, default=3,
                       help='每个测试的重复次数')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model, vocab_size, seq_len_ckpt = load_transformer_model(args.transformer_ckpt, device)
    
    # 使用 checkpoint 中的 seq_len，如果没有则使用参数
    seq_len = args.seq_len
    
    print(f"\n测试配置:")
    print(f"  seq_len: {seq_len}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  num_tests: {args.num_tests}")
    
    # 运行测试
    try:
        # 测试1: Transformer 输出一致性
        test_consistency(model, device, seq_len=seq_len, vocab_size=vocab_size, num_tests=args.num_tests)
        
        # 测试2: 压缩和解压一致性
        test_compression_decompression(model, device, seq_len=seq_len, vocab_size=vocab_size, num_tests=args.num_tests)
        
        print("\n" + "="*60)
        print("所有测试完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

