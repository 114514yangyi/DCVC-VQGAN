# Taming Transformers VQGAN 集成说明

## 概述

已成功将 Taming Transformers 项目中的 VQGAN 模型（包括 `VQModel` 和 `EMAVQ`）集成到现有的训练体系中。

## 主要修改

### 1. 创建适配器类 (`models/model_adapter.py`)

新增 `TamingVQGANAdapter` 类，将 taming 的 PyTorch Lightning 模型包装成符合现有训练体系的接口：

- **forward()**: 将 taming 的 `(dec, diff)` 返回值转换为 `(vq_loss, images_recon, perplexity, encoding_indices)`
- **encode()**: 编码函数，返回码本索引
- **decode()**: 解码函数，从码本索引重建图像
- **属性**: `output_channels`, `vocab_size` 等

### 2. 修复 taming 模型依赖 (`models/taming/models/vqgan.py`)

- 添加了 `get_obj_from_str()` 和 `instantiate_from_config()` 函数
- 移除了对 `main.py` 的依赖，使模型可以独立使用

### 3. 修复量化器 Bug (`models/taming/modules/vqvae/quantize.py`)

- 修复了 `EMAVectorQuantizer` 初始化时的参数名称错误（`codebook_dim` 和 `num_tokens`）

### 4. 注册新模型类型

在 `_MODEL_REGISTRY` 中注册了以下模型类型：
- `TamingVQGAN`, `taming_vqgan`
- `TamingVQModel`, `taming_vqmodel`
- `EMAVQ`, `emavq`
- `VQGAN`, `vqgan`

## 使用方法

### 在 `param.json` 中配置

```json
{
    "model_type": "TamingVQGAN",
    "model_args": {
        "model_variant": "EMAVQ",  // 或 "VQModel"
        "n_embed": 1024,
        "embed_dim": 256,
        "ddconfig": {
            "double_z": false,
            "z_channels": 256,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 1, 2, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [16],
            "dropout": 0.0
        },
        "lossconfig": {
            "target": "models.taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
            "params": {
                "disc_conditional": false,
                "disc_in_channels": 3,
                "disc_start": 10000,
                "disc_weight": 0.8,
                "codebook_weight": 1.0
            }
        }
    }
}
```

### 参数说明

#### 必需参数

- **ddconfig**: 编解码器配置
  - `z_channels`: 潜在空间通道数
  - `in_channels`: 输入图像通道数（通常为 3）
  - `out_ch`: 输出图像通道数（通常为 3）
  - `ch`: 基础通道数
  - `ch_mult`: 通道倍数列表，决定下采样次数
  - `num_res_blocks`: 每个分辨率级别的残差块数量
  - `attn_resolutions`: 应用注意力机制的分辨率列表
  - `resolution`: 输入图像分辨率

- **n_embed**: 码本大小（离散向量的数量）
- **embed_dim**: 量化特征的维度

#### 可选参数

- **model_variant**: 模型变体
  - `"VQModel"`: 标准 VQGAN（默认）
  - `"EMAVQ"`: 使用 EMA 更新码本的变体（训练更稳定）

- **lossconfig**: 损失函数配置
  - 如果不提供，将使用默认的 `VQLPIPSWithDiscriminator` 配置
  - `disc_start`: 开始使用判别器的训练步数
  - `disc_weight`: 判别器损失权重
  - `codebook_weight`: 码本损失权重

- **ckpt_path**: 预训练 checkpoint 路径（可选）

## 模型变体说明

### VQModel（标准 VQGAN）

- 使用标准的向量量化器
- 码本通过梯度下降更新
- 优化器包含量化器参数

### EMAVQ（推荐）

- 使用 EMA（指数移动平均）更新码本
- 码本更新更平滑，训练更稳定
- 优化器不包含量化器参数（因为码本通过 EMA 更新）
- **推荐用于生产环境**

## 与现有训练体系的兼容性

### 完全兼容

- ✅ 使用相同的 `TrainVqVae` 训练类
- ✅ 使用相同的 `LossCalculator` 损失计算
- ✅ 使用相同的数据加载器
- ✅ 支持 checkpoint 保存和恢复
- ✅ 支持所有现有的评估指标

### 注意事项

1. **损失计算**: taming 的模型内部已经包含了判别器损失计算，但用户的训练体系也使用 `LossCalculator`。两个系统可以共存，但建议：
   - 如果使用 taming 的 `lossconfig`，可以禁用 `LossCalculator` 中的 GAN 损失
   - 或者只使用 `LossCalculator`，将 taming 的 `lossconfig` 设置为仅包含感知损失

2. **优化器**: taming 的模型使用双优化器（自编码器和判别器），但用户的训练体系使用单优化器。适配器已处理这个问题，通过设置 `model.automatic_optimization = False` 禁用 PyTorch Lightning 的自动优化。

3. **训练步骤**: taming 的 `training_step` 需要 `optimizer_idx` 参数，但适配器不直接使用它，而是通过 `forward()` 方法返回损失，由用户的训练体系统一处理。

## 示例训练命令

```bash
python train/train_custom_videos.py \
    --device cuda:0 \
    --config param.json
```

## 故障排除

### 问题 1: 导入错误

如果遇到 `from main import instantiate_from_config` 错误，确保已修复 `models/taming/models/vqgan.py` 文件（已添加本地实现）。

### 问题 2: 量化器属性错误

如果遇到 `codebook_dim` 或 `num_tokens` 未定义错误，确保已修复 `models/taming/modules/vqvae/quantize.py` 文件。

### 问题 3: 损失计算冲突

如果训练不稳定，检查 `lossconfig` 和 `LossCalculator` 的配置，确保没有重复计算损失。

## 参考

- Taming Transformers 原始项目: https://github.com/CompVis/taming-transformers
- VQGAN 论文: https://arxiv.org/abs/2012.09841

