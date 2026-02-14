import tensorflow as tf
import tensorflow_compression as tfc
import numpy as np

def simple_arithmetic_coding_example():
    """
    简单的算术编码示例
    使用TFC的RangeCoder进行实际的算术编码
    """
    print("简单算术编码示例")
    print("=" * 50)
    
    # 1. 创建一些简单的测试数据
    # 假设我们有一个符号序列，每个符号有4种可能的值
    symbols = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 0, 1], dtype=np.int32)
    print(f"原始符号序列: {symbols}")
    print(f"符号数量: {len(symbols)}")
    
    # 2. 计算符号频率（用于构建概率模型）
    unique, counts = np.unique(symbols, return_counts=True)
    frequencies = counts / len(symbols)
    
    print(f"\n符号频率:")
    for sym, freq in zip(unique, frequencies):
        print(f"  符号 {sym}: {freq:.3f}")
    
    # 3. 使用TFC的RangeCoder进行编码
    # 注意：TFC的RangeCoder主要用于内部使用，但我们可以通过熵模型来使用它
    
    # 创建一个简单的分类分布熵模型
    class SimpleCategoricalEntropyModel:
        def __init__(self, frequencies):
            self.frequencies = frequencies
            self.cdf = np.cumsum(frequencies)
            self.cdf = np.insert(self.cdf, 0, 0)  # 在开始处添加0
            
        def encode(self, symbols):
            """模拟算术编码过程"""
            # 在实际的TFC中，这是自动处理的
            # 这里我们只是演示概念
            
            low = 0
            high = 1 << 32  # 使用32位精度
            
            for symbol in symbols:
                range_width = high - low
                symbol_low = low + int(range_width * self.cdf[symbol])
                symbol_high = low + int(range_width * self.cdf[symbol + 1])
                
                low = symbol_low
                high = symbol_high
            
            # 返回最终区间中的任意值作为编码结果
            encoded_value = (low + high) // 2
            return encoded_value
        
        def decode(self, encoded_value, num_symbols):
            """模拟算术解码过程"""
            decoded_symbols = []
            value = encoded_value
            
            for _ in range(num_symbols):
                # 找到包含该值的符号
                for symbol in range(len(self.frequencies)):
                    if (value >= self.cdf[symbol] * (1 << 32) and 
                        value < self.cdf[symbol + 1] * (1 << 32)):
                        decoded_symbols.append(symbol)
                        
                        # 更新值
                        range_low = self.cdf[symbol] * (1 << 32)
                        range_high = self.cdf[symbol + 1] * (1 << 32)
                        range_width = range_high - range_low
                        
                        value = (value - range_low) * (1 << 32) // range_width
                        break
            
            return decoded_symbols
    
    # 4. 使用我们的简单模型
    print("\n使用简单算术编码模型:")
    model = SimpleCategoricalEntropyModel(frequencies)
    
    # 编码
    encoded = model.encode(symbols)
    print(f"编码结果 (32位整数): {encoded}")
    
    # 解码
    decoded = model.decode(encoded, len(symbols))
    print(f"解码结果: {decoded}")
    
    # 验证
    if list(symbols) == decoded:
        print("✓ 编码解码成功！")
    else:
        print("✗ 编码解码失败！")
    
    # 5. 使用TFC的实际熵模型
    print("\n使用TFC的实际熵模型:")
    
    # 创建数据张量
    data_tensor = tf.constant(symbols.reshape(1, -1, 1), dtype=tf.float32)
    
    # 创建一个简单的熵模型
    entropy_model = tfc.ContinuousBatchedEntropyModel(
        prior=tfc.distributions.NoisyLogistic,
        coding_rank=1,
        compression=True,
        laplace_tail_mass=0.0,
    )
    
    # 模拟量化（在实际中，这通常与神经网络输出结合）
    quantized = tf.cast(data_tensor * 10, tf.int32)  # 简单缩放
    
    # 估计比特率
    _, bits = entropy_model(quantized, training=False)
    print(f"TFC估计的比特数: {bits.numpy():.2f}")
    print(f"比特率: {bits.numpy() / len(symbols):.4f} bits/symbol")
    
    return symbols, encoded, decoded

# 运行示例
if __name__ == "__main__":
    symbols, encoded, decoded = simple_arithmetic_coding_example()
    
    # 额外：展示压缩效果
    print("\n" + "=" * 50)
    print("压缩效果分析:")
    print("=" * 50)
    
    # 原始大小（假设每个符号用2位表示）
    original_bits = len(symbols) * 2
    print(f"原始大小: {original_bits} bits")
    
    # 算术编码的理论大小
    unique, counts = np.unique(symbols, return_counts=True)
    frequencies = counts / len(symbols)
    entropy = -np.sum(frequencies * np.log2(frequencies))
    theoretical_bits = len(symbols) * entropy
    
    print(f"理论最小大小: {theoretical_bits:.2f} bits")
    print(f"理论压缩比: {original_bits/theoretical_bits:.2f}:1")
