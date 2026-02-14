#!/usr/bin/env python3
"""
哈夫曼编码压缩 codebook index 示例

使用 Python 标准库和常用库实现哈夫曼编码，对 VQ-VAE codebook index 进行压缩。
"""

import os
import numpy as np
import argparse
from pathlib import Path
from collections import Counter
from typing import Tuple, Dict, List

# 尝试使用第三方库（如果可用）
try:
    import huffman
    HAS_HUFFMAN_LIB = True
except ImportError:
    HAS_HUFFMAN_LIB = False
    print("⚠ 未安装 huffman 库，将使用自定义实现")
    print("  安装方法: pip install huffman")


class HuffmanEncoder:
    """哈夫曼编码器（使用标准库实现）"""
    
    def __init__(self):
        self.code_table = {}  # 符号 -> 编码的映射
        self.reverse_table = {}  # 编码 -> 符号的映射
        
    def build_tree(self, frequencies: Dict[int, int]) -> Tuple:
        """
        构建哈夫曼树（使用元组表示）
        返回: (频率, 符号或子树)
        """
        # 创建叶子节点列表
        nodes = [(freq, symbol) for symbol, freq in frequencies.items()]
        
        # 构建哈夫曼树
        while len(nodes) > 1:
            # 按频率排序
            nodes.sort(key=lambda x: x[0])
            # 取出频率最小的两个节点
            left = nodes.pop(0)
            right = nodes.pop(0)
            # 合并为新节点
            merged = (left[0] + right[0], (left, right))
            nodes.append(merged)
        
        return nodes[0] if nodes else None
    
    def generate_codes(self, tree, prefix=''):
        """从哈夫曼树生成编码表"""
        if tree is None:
            return
        
        if isinstance(tree[1], tuple):
            # 内部节点，递归处理左右子树
            left, right = tree[1]
            self.generate_codes(left, prefix + '0')
            self.generate_codes(right, prefix + '1')
        else:
            # 叶子节点，记录编码
            symbol = tree[1]
            self.code_table[symbol] = prefix
            if prefix:  # 避免空编码
                self.reverse_table[prefix] = symbol
    
    def encode(self, data: np.ndarray) -> Tuple[bytes, Dict]:
        """
        编码数据
        
        Args:
            data: 输入的一维 numpy 数组（codebook indices）
            
        Returns:
            (编码后的字节流, 编码表字典)
        """
        # 统计频率
        unique, counts = np.unique(data, return_counts=True)
        frequencies = dict(zip(unique.tolist(), counts.tolist()))
        
        # 构建哈夫曼树并生成编码表
        tree = self.build_tree(frequencies)
        self.generate_codes(tree)
        
        # 编码数据
        encoded_bits = ''.join([self.code_table[int(symbol)] for symbol in data])
        
        # 转换为字节
        # 填充到8的倍数
        padding = (8 - len(encoded_bits) % 8) % 8
        encoded_bits += '0' * padding
        
        # 转换为字节数组
        encoded_bytes = bytearray()
        for i in range(0, len(encoded_bits), 8):
            byte_str = encoded_bits[i:i+8]
            encoded_bytes.append(int(byte_str, 2))
        
        # 保存元数据：原始长度、填充位数、编码表
        metadata = {
            'original_length': len(data),
            'padding': padding,
            'code_table': self.code_table,
            'dtype': str(data.dtype)
        }
        
        return bytes(encoded_bytes), metadata
    
    def decode(self, encoded_bytes: bytes, metadata: Dict) -> np.ndarray:
        """
        解码数据
        
        Args:
            encoded_bytes: 编码后的字节流
            metadata: 包含编码表的元数据
            
        Returns:
            解码后的 numpy 数组
        """
        # 恢复编码表
        self.code_table = metadata['code_table']
        self.reverse_table = {v: k for k, v in self.code_table.items()}
        
        # 转换为二进制字符串
        bits = ''.join([format(byte, '08b') for byte in encoded_bytes])
        
        # 移除填充
        if metadata['padding'] > 0:
            bits = bits[:-metadata['padding']]
        
        # 解码
        decoded = []
        current_code = ''
        for bit in bits:
            current_code += bit
            if current_code in self.reverse_table:
                decoded.append(self.reverse_table[current_code])
                current_code = ''
        
        return np.array(decoded, dtype=metadata.get('dtype', 'int64'))


def compress_with_huffman_lib(data: np.ndarray) -> Tuple[bytes, Dict]:
    """
    使用 huffman 库进行编码（如果可用）
    """
    if not HAS_HUFFMAN_LIB:
        raise ImportError("huffman 库未安装")
    
    # 统计频率
    unique, counts = np.unique(data, return_counts=True)
    frequencies = dict(zip(unique.tolist(), counts.tolist()))
    
    # 创建哈夫曼编码表
    code_table = huffman.codebook(frequencies.items())
    print(code_table)
    
    # 编码数据
    encoded_bits = ''.join([code_table[int(symbol)] for symbol in data])
    
    # 转换为字节
    padding = (8 - len(encoded_bits) % 8) % 8
    encoded_bits += '0' * padding
    
    encoded_bytes = bytearray()
    for i in range(0, len(encoded_bits), 8):
        byte_str = encoded_bits[i:i+8]
        encoded_bytes.append(int(byte_str, 2))
    
    metadata = {
        'original_length': len(data),
        'padding': padding,
        'code_table': code_table,
        'dtype': str(data.dtype)
    }
    
    return bytes(encoded_bytes), metadata


def process_npy_file(npy_path: str, output_dir: str = None, use_lib: bool = False) -> Dict:
    """
    处理单个 .npy 文件，进行哈夫曼编码
    
    Args:
        npy_path: .npy 文件路径
        output_dir: 输出目录（如果为 None，则在原文件同目录下保存）
        use_lib: 是否使用 huffman 库（如果可用）
        
    Returns:
        包含压缩统计信息的字典
    """
    print(f"\n{'='*60}")
    print(f"处理文件: {npy_path}")
    print('='*60)
    
    # 加载数据
    data = np.load(npy_path)
    original_shape = data.shape
    
    # 展平为一维数组
    data_flat = data.flatten()
    
    print(f"原始数据形状: {original_shape}")
    print(f"展平后长度: {len(data_flat)}")
    print(f"数据类型: {data.dtype}")
    print(f"数值范围: [{data_flat.min()}, {data_flat.max()}]")
    print(f"唯一值数量: {len(np.unique(data_flat))}")
    
    # 计算原始大小（字节）
    original_size_bytes = data.nbytes
    print(f"\n原始大小: {original_size_bytes:,} 字节 ({original_size_bytes / 1024:.2f} KB)")
    
    # 进行哈夫曼编码
    if use_lib and HAS_HUFFMAN_LIB:
        print("\n使用 huffman 库进行编码...")
        encoded_bytes, metadata = compress_with_huffman_lib(data_flat)
    else:
        print("\n使用自定义哈夫曼编码器...")
        encoder = HuffmanEncoder()
        encoded_bytes, metadata = encoder.encode(data_flat)
    
    # 计算压缩后大小（只计算编码后的数据，不包括元数据）
    compressed_size_bytes = len(encoded_bytes)
    
    print(f"编码后数据大小: {compressed_size_bytes:,} 字节 ({compressed_size_bytes / 1024:.2f} KB)")
    
    # 计算压缩率（只基于编码后的数据大小）
    compression_ratio = original_size_bytes / compressed_size_bytes
    compression_rate = (1 - compressed_size_bytes / original_size_bytes) * 100
    bits_per_index = (compressed_size_bytes * 8) / len(data_flat)
    
    print(f"\n压缩统计:")
    print(f"  压缩比: {compression_ratio:.2f}:1")
    print(f"  压缩率: {compression_rate:.2f}%")
    print(f"  平均每索引比特数: {bits_per_index:.4f} bits/index")
    
    # 固定长度编码对比
    max_value = data_flat.max()
    min_value = data_flat.min()
    value_range = max_value - min_value + 1
    fixed_bits_per_index = np.ceil(np.log2(value_range))
    fixed_total_bits = len(data_flat) * fixed_bits_per_index
    fixed_total_bytes = int(np.ceil(fixed_total_bits / 8))
    
    print(f"\n与固定长度编码对比:")
    print(f"  固定长度编码: {fixed_bits_per_index:.1f} bits/index")
    print(f"  固定长度总大小: {fixed_total_bytes:,} 字节")
    print(f"  哈夫曼编码节省: {(1 - compressed_size_bytes / fixed_total_bytes) * 100:.2f}%")
    
    # 验证解码
    print(f"\n验证解码...")
    if use_lib and HAS_HUFFMAN_LIB:
        # 使用自定义解码器（因为 huffman 库没有直接提供解码）
        encoder = HuffmanEncoder()
        encoder.code_table = metadata['code_table']
        encoder.reverse_table = {v: k for k, v in encoder.code_table.items()}
        decoded_data = encoder.decode(encoded_bytes, metadata)
    else:
        encoder = HuffmanEncoder()
        decoded_data = encoder.decode(encoded_bytes, metadata)
    
    # 验证数据是否一致
    if np.array_equal(data_flat, decoded_data):
        print("  ✓ 解码验证成功，数据完全一致")
    else:
        print("  ✗ 解码验证失败，数据不一致")
        mismatch_count = np.sum(data_flat != decoded_data)
        print(f"    不匹配数量: {mismatch_count}/{len(data_flat)}")
    
    # 保存压缩结果（只保存压缩后的数据，不保存元数据）
    if output_dir is None:
        output_dir = os.path.dirname(npy_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存压缩后的数据
    base_name = Path(npy_path).stem
    compressed_path = os.path.join(output_dir, f"{base_name}_huffman_compressed.bin")
    
    with open(compressed_path, 'wb') as f:
        f.write(encoded_bytes)
    
    print(f"\n保存文件:")
    print(f"  压缩数据: {compressed_path}")
    
    # 返回统计信息
    stats = {
        'file_path': npy_path,
        'original_shape': original_shape,
        'original_size_bytes': original_size_bytes,
        'compressed_size_bytes': compressed_size_bytes,
        'compression_ratio': compression_ratio,
        'compression_rate': compression_rate,
        'bits_per_index': bits_per_index,
        'num_indices': len(data_flat),  # 索引总数
        'fixed_bits_per_index': float(fixed_bits_per_index),
        'fixed_total_bytes': fixed_total_bytes,
        'num_unique_values': len(np.unique(data_flat)),
        'value_range': (int(min_value), int(max_value))
    }
    
    return stats


def process_directory(data_dir: str, output_dir: str = None, use_lib: bool = False):
    """
    批量处理目录中的所有 .npy 文件
    
    Args:
        data_dir: 包含 .npy 文件的目录
        output_dir: 输出目录
        use_lib: 是否使用 huffman 库
    """
    data_dir = Path(data_dir)
    npy_files = list(data_dir.glob('*.npy'))
    
    if not npy_files:
        print(f"在目录 {data_dir} 中未找到 .npy 文件")
        return
    
    print(f"找到 {len(npy_files)} 个 .npy 文件")
    
    all_stats = []
    total_original = 0
    total_compressed = 0
    total_indices = 0  # 总索引数
    
    for npy_file in npy_files:
        try:
            stats = process_npy_file(str(npy_file), output_dir, use_lib)
            all_stats.append(stats)
            total_original += stats['original_size_bytes']
            total_compressed += stats['compressed_size_bytes']
            total_indices += stats['num_indices']
        except Exception as e:
            print(f"处理文件 {npy_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总体统计
    if all_stats:
        # 计算平均每个index的比特数
        avg_bits_per_index = (total_compressed * 8) / total_indices if total_indices > 0 else 0
        
        print(f"\n{'='*60}")
        print("总体统计")
        print('='*60)
        print(f"处理文件数: {len(all_stats)}")
        print(f"总索引数: {total_indices:,}")
        print(f"总原始大小: {total_original:,} 字节 ({total_original / 1024 / 1024:.2f} MB)")
        print(f"总压缩大小: {total_compressed:,} 字节 ({total_compressed / 1024 / 1024:.2f} MB)")
        print(f"总体压缩比: {total_original / total_compressed:.2f}:1")
        print(f"总体压缩率: {(1 - total_compressed / total_original) * 100:.2f}%")
        print(f"平均每个index使用的比特数: {avg_bits_per_index:.4f} bits/index")


def main():
    parser = argparse.ArgumentParser(
        description='使用哈夫曼编码压缩 codebook index (.npy 文件)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default="/data/huyang/data/vaild_indices",
        help='输入 .npy 文件路径或包含 .npy 文件的目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="/data/huyang/output",
        help='输出目录（默认为输入文件同目录）'
    )
    parser.add_argument(
        '--use_lib',
        action='store_true',
        help='使用 huffman 库（如果已安装）'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 处理单个文件
        process_npy_file(str(input_path), args.output_dir, args.use_lib)
    elif input_path.is_dir():
        # 处理目录
        process_directory(str(input_path), args.output_dir, args.use_lib)
    else:
        print(f"错误: {args.input} 不是有效的文件或目录")


if __name__ == '__main__':
    main()

