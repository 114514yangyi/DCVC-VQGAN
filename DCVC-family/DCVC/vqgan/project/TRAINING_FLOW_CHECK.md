# 训练流程完整检查文档

## 概述

本文档详细说明了从训练脚本到模型的完整训练流程，特别是 Taming VQGAN 模型的集成。

## 完整训练流程

### 1. 训练入口 (`train/train_custom_videos.py`)

**流程：**
1. 解析命令行参数（设备、配置文件路径等）
2. 调用 `train_videos()` 函数

**关键代码：**
```python
train_videos(device_override=args.device, config_path=config_path)
```

### 2. 训练初始化 (`train/train_custom_videos.py::train_videos()`)

**流程：**
1. 加载配置文件 `param.json`
2. 初始化日志系统 (`LogManager`)
3. 创建模型（通过 `model_adapter.create_model()`）
4. 创建数据加载器 (`video_pipe`)
5. 创建训练对象 (`TrainVqVae`)
6. 开始训练 (`train_object.train()`)

**关键代码：**
```python
# 创建模型（使用适配器）
model = create_model(model_args=model_args, config_path=config_path)

# 创建训练对象
train_object = TrainVqVae(
    model=model,
    training_loader=training_loader,
    validation_loader=validation_loader,
    log_mgr=log_mgr,
    normalize=normalize,
    unnormalize=unnormalize,
    config_path=config_path
)

# 开始训练
train_object.train()
```

### 3. 模型创建 (`models/model_adapter.py::create_model()`)

**流程：**
1. 从配置文件读取 `model_type`
2. 根据 `model_type` 查找对应的适配器类
3. 创建适配器实例

**支持的模型类型：**
- `TamingVQGAN`, `taming_vqgan`, `EMAVQ`, `emavq`, `VQGAN`, `vqgan` → `TamingVQGANAdapter`
- `VQVAE2`, `vqvae2` → `VQVAEAdapter`
- `VqVae`, `vqvae` → `VqVae2Adapter`
- 等等...

**Taming VQGAN 适配器创建流程：**
1. 读取配置参数（`ddconfig`, `lossconfig`, `n_embed`, `embed_dim` 等）
2. 根据 `model_variant` 创建 `VQModel` 或 `EMAVQ`
3. 设置 `automatic_optimization = False`（禁用 PyTorch Lightning 自动优化）
4. 包装为适配器

### 4. 训练对象初始化 (`train/trainVqVae.py::__attrs_post_init__()`)

**流程：**
1. 从配置文件读取训练参数（`num_steps`, `lr`, `lr_decay` 等）
2. **创建损失计算器 (`LossCalculator`)**：
   - 传递模型对象以检测是否是 taming 模型
   - 如果是 taming 模型，从模型中获取损失函数和判别器
   - 如果不是，初始化传统的判别器和感知损失函数
3. **初始化优化器**：
   - **Taming 模型**：创建双优化器（自编码器和判别器）
   - **传统模型**：创建单优化器
4. 初始化学习率调度器
5. 加载 checkpoint（如果提供）

**关键代码：**
```python
# 创建损失计算器（会自动检测是否是 taming 模型）
self.loss_calculator = LossCalculator(
    decoder=self.model.decoder if hasattr(self.model, 'decoder') else None,
    num_steps=self.num_steps,
    output_channels=self.model.output_channels,
    device=device,
    base_lr=self.lr,
    config_path=self.config_path,
    model=self.model  # 传递模型以检测类型
)

# 检测是否是 taming 模型
self.use_taming_loss = self.loss_calculator.use_taming_loss

# 根据模型类型初始化优化器
if self.use_taming_loss:
    # 双优化器
    opt_ae, opt_disc = self.loss_calculator.get_taming_optimizers()
    self.optimizer = opt_ae
    self.optimizer_disc = opt_disc
else:
    # 单优化器
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    self.optimizer_disc = None
```

### 5. 损失计算器初始化 (`train/lossconfig.py::LossCalculator.__init__()`)

**流程：**
1. 检测模型是否是 taming 模型（通过 `_is_taming_model()`）
2. **如果是 taming 模型**：
   - 从 taming 模型中获取 `loss` 对象（`VQLPIPSWithDiscriminator`）
   - 从 `loss` 对象中获取判别器
   - 设置 `use_taming_loss = True`
3. **如果不是 taming 模型**：
   - 初始化传统的判别器和感知损失函数
   - 设置 `use_taming_loss = False`

**关键代码：**
```python
# 检测是否是 taming 模型
self.is_taming_model = self._is_taming_model()

if self.is_taming_model and self.use_taming_loss is not False:
    # 使用 taming 损失函数
    taming_model = self._get_taming_model()
    self.taming_loss_fn = taming_model.loss
    self.discriminator = taming_model.loss.discriminator
    self.use_taming_loss = True
else:
    # 使用传统损失函数
    self.use_taming_loss = False
    # 初始化传统判别器和感知损失函数
    ...
```

### 6. 训练循环 (`train/trainVqVae.py::train()`)

**每个训练步骤的流程：**

#### 6.1 数据加载
```python
data_batch = next(data_iter)
images, video_paths = data_batch
```

#### 6.2 数据预处理
```python
# 重新排列维度: (batch, sequence, h, w, c) -> (batch*sequence, c, h, w)
images = rearrange(images, 'b d h w c -> (b d) c h w')
# 归一化
images = self.normalize(images.float() / 255.)
```

#### 6.3 模型前向传播
```python
model_out = self.model(images)
vq_loss, images_recon, perplexity, encoding_indices = model_out
```

**Taming 模型的前向传播：**
- `TamingVQGANAdapter.forward()` 调用 `model.encode()` 和 `model.decode()`
- 返回 `(vq_loss, images_recon, perplexity, encoding_indices)`

#### 6.4 损失计算

**Taming 模型（双优化器）：**
```python
if self.use_taming_loss:
    # 1. 自编码器损失（optimizer_idx=0）
    self.optimizer.zero_grad()
    loss_dict_ae = self.loss_calculator.compute_total_loss(
        images=images,
        images_recon=images_recon,
        vq_loss=vq_loss,
        step=i,
        optimizer_idx=0  # 自编码器
    )
    total_loss_ae = loss_dict_ae['total_loss']
    total_loss_ae.backward()
    self.optimizer.step()
    
    # 2. 判别器损失（optimizer_idx=1，每3步训练一次）
    if self.optimizer_disc is not None and i % 3 == 0:
        self.optimizer_disc.zero_grad()
        loss_dict_disc = self.loss_calculator.compute_total_loss(
            images=images,
            images_recon=images_recon,
            vq_loss=vq_loss,
            step=i,
            optimizer_idx=1  # 判别器
        )
        total_loss_disc = loss_dict_disc['total_loss']
        total_loss_disc.backward()
        self.optimizer_disc.step()
```

**传统模型（单优化器）：**
```python
else:
    self.optimizer.zero_grad()
    loss_dict = self.loss_calculator.compute_total_loss(
        images=images,
        images_recon=images_recon,
        vq_loss=vq_loss,
        step=i,
        optimizer_idx=0
    )
    total_loss = loss_dict['total_loss']
    total_loss.backward()
    self.optimizer.step()
    # 判别器优化（在 LossCalculator 内部管理）
    self.loss_calculator.step_discriminator(loss_dict['gan_loss'], i)
```

#### 6.5 Taming 损失函数计算 (`train/lossconfig.py::_compute_taming_loss()`)

**流程：**
1. 获取 taming 模型对象
2. 获取解码器最后一层（用于自适应权重计算）
3. 调用 taming 损失函数：
   ```python
   loss, log_dict = self.taming_loss_fn(
       codebook_loss=vq_loss,
       inputs=images,
       reconstructions=images_recon,
       optimizer_idx=optimizer_idx,  # 0=自编码器，1=判别器
       global_step=step,
       last_layer=last_layer,
       cond=None,
       split="train"
   )
   ```
4. 返回损失字典

**Taming 损失函数 (`models/taming/modules/losses/vqperceptual.py::VQLPIPSWithDiscriminator.forward()`)**：
- **optimizer_idx=0（自编码器）**：
  - 计算重建损失（L1 + 感知损失）
  - 计算生成器损失（对抗损失）
  - 计算自适应权重（`calculate_adaptive_weight`）
  - 总损失 = 重建损失 + 自适应权重 × 判别器因子 × 生成器损失 + 码本损失
- **optimizer_idx=1（判别器）**：
  - 计算判别器损失（hinge loss 或 vanilla loss）
  - 返回判别器损失

#### 6.6 日志记录和检查点保存
- 记录损失和指标到 wandb/tensorboard
- 定期保存 checkpoint（包含优化器状态）

### 7. Checkpoint 保存和加载

**保存：**
```python
checkpoint_data = {
    'steps': i,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'scheduler': self.scheduler.state_dict(),
}

# Taming 模型：保存判别器优化器
if self.use_taming_loss and self.optimizer_disc is not None:
    checkpoint_data['optimizer_disc'] = self.optimizer_disc.state_dict()

# 传统模型：保存判别器和优化器状态
if not self.use_taming_loss:
    checkpoint_data.update(self.loss_calculator.get_checkpoint_state())
```

**加载：**
```python
self.model.load_state_dict(checkpoint['state_dict'])
self.optimizer.load_state_dict(checkpoint['optimizer'])
self.scheduler.load_state_dict(checkpoint['scheduler'])

# Taming 模型：加载判别器优化器
if self.use_taming_loss and self.optimizer_disc is not None:
    if 'optimizer_disc' in checkpoint:
        self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
```

## 关键差异总结

### Taming 模型 vs 传统模型

| 特性 | Taming 模型 | 传统模型 |
|------|------------|---------|
| 损失函数 | `VQLPIPSWithDiscriminator`（模型内部） | `LossCalculator`（外部管理） |
| 优化器 | 双优化器（自编码器 + 判别器） | 单优化器 + 内部判别器优化 |
| 判别器 | 从 `model.loss.discriminator` 获取 | `LossCalculator` 内部创建 |
| 感知损失 | 包含在 taming 损失函数中 | `LossCalculator` 内部管理 |
| 训练步骤 | 分别优化自编码器和判别器 | 统一优化，判别器每3步优化一次 |

## 配置示例

### Taming VQGAN 配置 (`param.json`)

```json
{
    "model_type": "TamingVQGAN",
    "model_args": {
        "model_variant": "EMAVQ",
        "n_embed": 1024,
        "embed_dim": 256,
        "ddconfig": {
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
    },
    "train_args": {
        "use_taming_loss": true,  // 可选：自动检测，也可以显式设置
        "lr": 1e-4,
        "num_steps": 100000,
        ...
    }
}
```

## 验证清单

- [x] 模型创建流程正确
- [x] 损失计算器自动检测 taming 模型
- [x] 双优化器正确初始化
- [x] 训练循环正确处理双优化器
- [x] Checkpoint 保存和加载包含所有优化器状态
- [x] 损失计算正确调用 taming 损失函数
- [x] 判别器训练频率正确（每3步一次）
- [x] 日志记录包含所有损失信息

## 注意事项

1. **Taming 模型的损失函数**：已经包含了感知损失、重建损失、生成器损失和判别器损失，不需要额外的损失计算。

2. **优化器管理**：Taming 模型使用双优化器，需要分别优化自编码器和判别器。

3. **判别器训练频率**：每3步训练一次判别器，避免判别器过强。

4. **自适应权重**：Taming 损失函数使用自适应权重（`calculate_adaptive_weight`）来平衡重建损失和生成器损失。

5. **EMA 量化器**：EMAVQ 模型的量化器参数不在优化器中（通过 EMA 更新），需要特殊处理。

