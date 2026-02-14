# VQ-VAE 视频压缩项目详细文档

## 目录

1. [项目概述](#项目概述)
2. [项目架构](#项目架构)
3. [模型系统详解](#模型系统详解)
4. [数据加载系统详解](#数据加载系统详解)
5. [训练流程详解](#训练流程详解)
6. [损失函数系统详解](#损失函数系统详解)
7. [日志系统详解](#日志系统详解)
8. [配置文件详解](#配置文件详解)
9. [快速开始](#快速开始)
10. [文件结构说明](#文件结构说明)

---

## 项目概述

本项目是一个基于 VQ-VAE（Vector Quantized Variational AutoEncoder）的视频压缩与重建系统。项目支持多种 VQ-VAE 模型变体，包括：

- **VqVae2**：基础的 VQ-VAE2 模型
- **VQVAE2**：多层级 VQ-VAE 模型
- **VQVAE2_OptVQ**：使用 OptVQ 量化器的 VQ-VAE 模型
- **TamingVQGAN**：Taming Transformers 的 VQGAN 模型（支持 VQModel 和 EMAVQ 两种变体）

项目采用**适配器模式**统一管理不同模型，通过配置文件灵活切换模型类型，支持视频和图像两种数据模式，并提供完整的训练、验证、日志记录和模型保存功能。

---

## 项目架构

### 整体架构图

```
项目根目录/
├── main.py                    # 训练启动器（支持单任务/多任务模式）
├── param.json                  # 主配置文件
├── train/
│   ├── train_custom_videos.py # 训练主入口
│   ├── trainVqVae.py         # 训练循环核心类
│   ├── video_utils.py         # 数据加载工具
│   ├── train_utils.py         # 训练工具函数
│   └── lossconfig.py          # 损失计算配置
├── models/
│   ├── model_adapter.py       # 模型适配器（核心）
│   ├── vq_vae/               # VqVae2 模型实现
│   ├── vqvae2/               # VQVAE2 模型实现
│   ├── vqvae2_optvq/         # VQVAE2 OptVQ 模型实现
│   └── taming/               # Taming VQGAN 模型实现
├── log_utils/
│   └── log_utils.py           # 日志管理系统
└── metric_utils/
    └── metric_utils.py       # 指标评估系统
```

### 核心设计模式

1. **适配器模式**：通过 `model_adapter.py` 统一不同模型的接口
2. **配置驱动**：所有参数通过 `param.json` 配置文件管理
3. **模块化设计**：训练、数据加载、损失计算、日志记录各自独立

---

## 模型系统详解

### 模型适配器架构

项目使用 `models/model_adapter.py` 中的适配器系统来统一管理不同的模型实现。

#### 1. 适配器基类：`BaseVqVaeAdapter`

所有模型适配器都继承自 `BaseVqVaeAdapter`，它定义了统一的接口：

```python
class BaseVqVaeAdapter(nn.Module):
    """
    提供统一的模型接口：
    - forward(): 前向传播，返回 (vq_loss, images_recon, perplexity, encoding_indices)
    - encoder: 编码器属性
    - decoder: 解码器属性
    - encode(): 编码函数（推理时使用）
    - decode(): 解码函数（推理时使用）
    """
```

**关键特性**：
- 使用 `object.__setattr__` 和 `object.__getattribute__` 避免 PyTorch 的 `__setattr__` 拦截
- 通过 `__getattr__` 代理所有属性访问到内部模型
- 支持设备移动（`to()`, `cuda()`, `cpu()`）

#### 2. 模型创建流程

**步骤 1：读取配置文件**

```python
# train/train_custom_videos.py
with open(config_path, 'r', encoding='utf-8') as f:
    params = json.load(f)

model_args = params['model_args']
model_type = params.get('model_type', 'VqVae2')  # 默认使用 VqVae2
```

**步骤 2：通过适配器创建模型**

```python
# train/train_custom_videos.py (第195行)
from models.model_adapter import create_model

model = create_model(model_args=model_args, config_path=config_path)
```

**步骤 3：适配器内部处理**

```python
# models/model_adapter.py (第838-888行)
def create_model(model_args, config_path='param.json'):
    # 1. 加载配置文件
    config = _load_config(config_path)
    
    # 2. 从配置读取模型类型
    model_type = config.get('model_type', 'VqVae2')
    
    # 3. 合并参数（配置文件 + 传入参数）
    model_args = {**config.get('model_args', {}), **model_args}
    
    # 4. 从注册表查找适配器类
    adapter_class = _MODEL_REGISTRY.get(model_type)
    
    # 5. 创建适配器实例
    return adapter_class(model_args, config_path)
```

#### 3. 支持的模型类型

**模型注册表**（`models/model_adapter.py` 第892-912行）：

```python
_MODEL_REGISTRY = {
    'VqVae': VqVae2Adapter,
    'vq_vae': VqVae2Adapter,
    'VQVAE2': VQVAEAdapter,
    'vqvae2': VQVAEAdapter,
    'VQVAE2_OptVQ': VqvaeOptVQAdapter,
    'vqvae2optvq': VqvaeOptVQAdapter,
    'TamingVQGAN': TamingVQGANAdapter,
    'taming_vqgan': TamingVQGANAdapter,
    'EMAVQ': TamingVQGANAdapter,
    'VQGAN': TamingVQGANAdapter,
}
```

**各适配器说明**：

1. **VqVae2Adapter**（第187-219行）
   - 适配 `models.vq_vae.vq_vae.VqVae2`
   - 参数：`group_count`, `n_hid`, `vocab_size`, `n_init` 等

2. **VQVAEAdapter**（第256-391行）
   - 适配 `models.vqvae2.vqvae.VQVAE`
   - 支持多层级量化（`nb_levels`）
   - 参数转换：`input_channels` → `in_channels`, `vocab_size` → `nb_entries`

3. **VqvaeOptVQAdapter**（第670-816行）
   - 适配 `models.vqvae2_optvq.vqvae.VQVAE`
   - 使用 OptVQ 量化器（优化量化）
   - 额外参数：`epsilon`, `n_iters`, `normalize_mode` 等

4. **TamingVQGANAdapter**（第394-667行）
   - 适配 `models.taming.models.vqgan.VQModel` 和 `EMAVQ`
   - 支持 PyTorch Lightning 的 VQGAN 模型
   - 参数转换：`vocab_size` → `n_embed`, `n_init` → `embed_dim`
   - 支持 `ddconfig` 和 `lossconfig` 配置

#### 4. 模型前向传播接口

所有适配器都实现统一的 `forward()` 接口：

```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    前向传播
    
    Args:
        x: 输入张量，形状为 (batch*sequence, channels, height, width)
    
    Returns:
        Tuple[vq_loss, images_recon, perplexity, encoding_indices]:
        - vq_loss: VQ 损失（标量）
        - images_recon: 重建图像，形状与输入相同
        - perplexity: 困惑度（标量）
        - encoding_indices: 码本索引（可选）
    """
    return self._get_model()(x)
```

**注意**：不同模型的内部实现可能不同，但通过适配器统一了接口。

---

## 数据加载系统详解

### 数据加载流程

#### 1. 配置文件读取

```python
# train/train_custom_videos.py (第108-111行)
if 'data_process' in params:
    params['data_args']['training_data_files'] = list_videos(params["data_process"]["train"])
    params['data_args']['validation_data_files'] = list_videos(params["data_process"]["vaild"])
```

#### 2. 数据加载器创建

```python
# train/train_custom_videos.py (第281-308行)
training_loader = video_pipe(
    filenames=data_args['training_data_files'],
    batch_size=data_args['batch_size'],
    num_threads=data_args['num_threads'],
    device_id=data_args['device_id'],
    sequence_length=data_args['sequence_length'],
    shard_id=data_args['shard_id'],
    num_shards=data_args['num_shards'],
    initial_prefetch_size=data_args['initial_prefetch_size'],
    seed=data_args['seed'],
    random_shuffle=False,
    config_path=config_path  # 用于读取 use_images 参数
)
```

### 数据加载器详解

#### 1. `video_pipe()` 函数（`train/video_utils.py` 第477-604行）

**功能**：根据配置创建视频或图片数据加载器

**关键逻辑**：

```python
def video_pipe(filenames, config_path, batch_size=4, ...):
    # 1. 读取配置文件，检查是否使用图片模式
    use_images = config.get('use_images', False)
    
    # 2. 根据 use_images 选择数据集类型
    if use_images:
        dataset = ImageDataset(filenames, ...)
    else:
        dataset = VideoDataset(filenames, ...)
    
    # 3. 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_threads,
        collate_fn=collate_fn,  # 自定义的 collate 函数
        ...
    )
    
    # 4. 返回包装后的 DataLoader
    return DataLoaderWrapper(dataloader)
```

#### 2. 数据集类

**VideoDataset**（`train/video_utils.py` 第380-474行）：
- 基于 OpenCV 的视频数据集
- 只加载指定数量的帧（`sequence_length`），不加载整个视频
- 支持随机选择起始帧
- 立即调整视频分辨率以减少内存占用

**ImageDataset**（`train/video_utils.py` 第297-377行）：
- 基于图片文件的数据集
- 随机选择图片组成序列
- 所有图片都是独立的，不需要分组

#### 3. 数据预处理流程

**训练循环中的数据预处理**（`train/trainVqVae.py` 第579-594行）：

```python
# 1. DataLoader 返回 (tensor, paths) 元组
images, video_paths = data_batch

# 2. 转换为 float 类型
images = images.float()

# 3. 移动到模型设备
images = images.to(device)

# 4. 获取维度信息
b, d, h, w, c = images.size()  # (batch, sequence, height, width, channels)

# 5. 重新排列维度: (batch, sequence, height, width, channels) -> (batch*sequence, channels, height, width)
images = rearrange(images, 'b d h w c -> (b d) c h w')

# 6. 归一化: 将像素值从 [0, 255] 缩放到 [0, 1]，然后应用标准化
images = self.normalize(images.float() / 255.)
```

**标准化参数**（`train/train_custom_videos.py` 第276-279行）：

```python
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
normalize = transforms.Normalize(mean=mean, std=std)
unnormalize = NormalizeInverse(mean=mean, std=std)
```

#### 4. 显存优化措施

1. **只读取需要的帧数**：不读取整个视频
2. **立即调整分辨率**：减少内存占用
3. **使用 uint8 类型**：直到需要时再转换为 float32
4. **及时释放资源**：释放视频捕获对象和临时数据
5. **减少预取数量**：`prefetch_factor=2`

---

## 训练流程详解

### 训练入口

**主入口**：`train/train_custom_videos.py` 的 `train_videos()` 函数

**执行流程**：

```python
def train_videos(device_override=None, config_path='param.json'):
    # 1. 加载配置文件
    params = json.load(open(config_path))
    
    # 2. 初始化日志系统
    log_mgr = LogManager(...)
    
    # 3. 创建模型（通过适配器）
    model = create_model(model_args=model_args, config_path=config_path)
    
    # 4. 分析并保存模型架构
    analyze_model_architecture(model, input_shape, architecture_file, logger)
    
    # 5. 创建数据加载器
    training_loader = video_pipe(...)
    validation_loader = video_pipe(...)  # 可选
    
    # 6. 创建训练对象并开始训练
    train_object = TrainVqVae(
        model=model,
        training_loader=training_loader,
        validation_loader=validation_loader,
        log_mgr=log_mgr,
        normalize=normalize,
        unnormalize=unnormalize,
        logger=logger,
        default_gpu=device,
        config_path=config_path
    )
    train_object.train()
```

### 训练循环详解

**核心类**：`TrainVqVae`（`train/trainVqVae.py` 第68-869行）

#### 1. 初始化阶段（`__attrs_post_init__`）

```python
def __attrs_post_init__(self):
    # 1. 从配置文件读取参数
    train_args = get_train_args_from_config(self.config_path)
    self.num_steps = train_args.get('num_steps', 100000)
    self.lr = train_args.get('lr', 0.001)
    ...
    
    # 2. 初始化损失计算器
    self.loss_calculator = LossCalculator(
        decoder=self.model.decoder,
        num_steps=self.num_steps,
        output_channels=self.model.output_channels,
        device=device,
        base_lr=self.lr,
        config_path=self.config_path,
        model=self.model
    )
    
    # 3. 根据模型类型初始化优化器
    if self.use_taming_loss:
        # taming 模型使用双优化器
        opt_ae, opt_disc = self.loss_calculator.get_taming_optimizers()
        self.optimizer = opt_ae
        self.optimizer_disc = opt_disc
    else:
        # 传统模型使用单优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    # 4. 学习率调度器
    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
    
    # 5. 加载 checkpoint（如果提供）
    if self.checkpoint_path is not None:
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
...
```

#### 2. 训练循环（`train()` 方法）

**主循环结构**（第515-869行）：

```python
def train(self):
    self.model.train()
    global_step = self.start_step
    
    # 创建数据加载器的迭代器（用于无限循环）
    data_iter = iter(self.training_loader)
    
    while global_step < self.num_steps:
        # 1. 获取数据批次
        data_batch = next(data_iter)
        
        # 2. 数据预处理
        images = rearrange(images, 'b d h w c -> (b d) c h w')
        images = self.normalize(images.float() / 255.)
        
        # 3. 前向传播
        model_out = self.model(images)
        vq_loss, images_recon, perplexity, encoding_indices = model_out
        
        # 4. 计算损失
        if self.use_taming_loss:
            # taming 模型：双优化器训练
            # 先训练自编码器
            self.optimizer.zero_grad()
            loss_dict_ae = self.loss_calculator.compute_total_loss(..., optimizer_idx=0)
            total_loss_ae.backward()
            self.optimizer.step()
            
            # 再训练判别器
            if self.use_gan:
                self.optimizer_disc.zero_grad()
                loss_dict_disc = self.loss_calculator.compute_total_loss(..., optimizer_idx=1)
                total_loss_disc.backward()
                self.optimizer_disc.step()
        else:
            # 传统模型：单优化器训练
            self.optimizer.zero_grad()
            loss_dict = self.loss_calculator.compute_total_loss(...)
            total_loss.backward()
            self.optimizer.step()
            # 判别器优化（每3步一次）
            self.loss_calculator.step_discriminator(gan_loss, global_step)
        
        # 5. 记录日志和指标
        if global_step % self.wandb_log_interval == 0:
            self._log_wandb_metrics(step=global_step, metrics=metrics_to_log)
        
        # 6. 定期保存 checkpoint
        if global_step % self.save_interval == 0:
            save_checkpoint(...)
        
        # 7. 定期验证
        if global_step % self.validation_interval == 0:
            val_metrics = self.evaluate()
            self._check_and_save_best_models(global_step, val_metrics)
        
        # 8. 更新学习率
        if global_step % 1000 == 0:
            self.scheduler.step()
        
        global_step += 1
```

#### 3. 验证流程（`evaluate()` 方法）

```python
@torch.no_grad()
def evaluate(self, max_images: int = 10000):
    self.model.eval()
    
    for batch_idx, data_batch in enumerate(self.validation_loader):
        # 1. 数据预处理（与训练相同）
        images = rearrange(images, 'b d h w c -> (b d) c h w')
        images = self.normalize(images.float() / 255.)
        
        # 2. 前向传播
        model_out = self.model(images)
        vq_loss, images_recon, perplexity, encoding_indices = model_out
        
        # 3. 计算损失
        recon_error = F.mse_loss(images_recon, images)
        loss = recon_error + vq_loss
        
        # 4. 计算指标（PSNR, FID, LPIPS 等）
        metric_vals = self.metrics.compute_all(
            orig=images,
            recon=images_recon,
            recon_mse=recon_error,
            perplexity=perplexity,
            encoding_indices=encoding_indices,
            b=b,
            d=d,
            compute_fid_lpips=True
        )
    
    # 5. 汇总指标
    aggregated = {
        "val/loss/total": _mean(loss_accum),
        "val/loss/recon": _mean(recon_accum),
        "val/stats/psnr": _mean(psnr_accum),
        "val/stats/fid": final_fid,
        "val/stats/lpips": _mean(lpips_accum),
        ...
    }
    
    return aggregated
```

---

## 损失函数系统详解

### 损失计算器：`LossCalculator`

**位置**：`train/lossconfig.py` 第55-686行

**功能**：统一管理所有损失的计算，包括：
- 重建损失（MSE, L1）
- VQ 损失
- 感知损失（LPIPS）
- GAN 损失（生成器和判别器）
- Taming VQGAN 损失（如果使用 taming 模型）

### 损失计算流程

#### 1. 初始化（`__init__`）

```python
def __init__(self, decoder, num_steps, output_channels, device, base_lr, config_path, model):
    # 1. 从配置文件读取参数
    train_args = get_train_args_from_config(config_path)
    self.use_gan = train_args.get('use_gan', True)
    self.disc_factor = train_args.get('disc_factor', 1.0)
    self.disc_start = train_args.get('disc_start', 20000)
    ...
    
    # 2. 检测是否是 taming 模型
    self.is_taming_model = self._is_taming_model()
    
    # 3. 初始化判别器和感知损失函数（如果使用 GAN）
    if self.use_gan:
        self.discriminator = Discriminator(...).to(device)
        self.perceptual_loss_fn = LPIPS().eval().to(device)
        self.optimizer_disc = torch.optim.Adam(...)
```

#### 2. 总损失计算（`compute_total_loss`）

**传统模型**（第436-541行）：

```python
def compute_total_loss(self, images, images_recon, vq_loss, step, optimizer_idx=0):
    # 1. 计算重建损失
    recon_error, pixel_loss = self.compute_reconstruction_loss(images, images_recon)
    
    # 2. 如果使用 GAN
    if self.use_gan:
        # 计算判别器权重因子（逐渐增加）
        disc_factor = VQGANLossHelper.adopt_weight(self.disc_factor, step, self.disc_start)
        
        # 计算感知损失（LPIPS）
        perceptual_loss = self.compute_perceptual_loss(images, images_recon)
        
        # 计算感知重建损失（加权后的感知+像素损失）
        perceptual_rec_loss = (
            self.perceptual_loss_factor * perceptual_loss +
            self.pixel_loss_factor * pixel_loss
        )
        
        # 计算生成器损失
        g_fake = self.discriminator(images_recon)
        g_loss = -g_fake.mean()
        
        # 计算 lambda_gan（自适应权重）
        lambda_gan = VQGANLossHelper.calculate_lambda(
            decoder=self.decoder,
            perceptual_loss=perceptual_rec_loss,
            gan_loss=g_loss
        )
        
        # 总损失（VQ-GAN）
        total_loss = perceptual_rec_loss + vq_loss + disc_factor * lambda_gan * g_loss
    else:
        # 不使用 GAN，使用简单的 VQ-VAE 损失
        total_loss = recon_error + vq_loss
    
    # 3. 计算判别器损失（用于判别器训练）
    gan_loss = self.compute_gan_discriminator_loss(images, images_recon, step, disc_factor)
    
    return {
        'total_loss': total_loss,
        'recon_error': recon_error,
        'pixel_loss': pixel_loss,
        'vq_loss': vq_loss,
        'perceptual_loss': perceptual_loss,
        'g_loss': g_loss,
        'gan_loss': gan_loss,
        'disc_factor': disc_factor,
        'lambda_gan': lambda_gan
    }
```

**Taming 模型**（第543-634行）：

```python
def _compute_taming_loss(self, images, images_recon, vq_loss, step, optimizer_idx):
    taming_model = self._get_taming_model()
    
    # 调用 taming 损失函数
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
    
    # 提取关键损失值
    total_loss = loss
    recon_error = log_dict.get('train/nll_loss', ...)
    pixel_loss = log_dict.get('train/rec_loss', ...)
    perceptual_loss = log_dict.get('train/p_loss', ...)
    g_loss = log_dict.get('train/g_loss', ...)
    gan_loss = log_dict.get('train/disc_loss', ...)
    
    return {
        'total_loss': total_loss,
        'recon_error': recon_error,
        'pixel_loss': pixel_loss,
        'vq_loss': codebook_loss,
        'perceptual_loss': perceptual_loss,
        'g_loss': g_loss,
        'gan_loss': gan_loss,
        'log_dict': log_dict  # 保存完整的日志字典
    }
```

### 损失函数组成

#### 1. 重建损失

- **MSE 损失**：`F.mse_loss(images_recon, images)`
- **L1 损失**（可选）：`F.l1_loss(images_recon, images)`

#### 2. VQ 损失

- 由模型内部计算，包括：
  - **Codebook 损失**：量化误差
  - **Commitment 损失**：编码器输出与量化向量的对齐

#### 3. 感知损失（LPIPS）

- 使用预训练的 VGG 网络计算感知距离
- 衡量重建图像与原始图像在感知空间的距离

#### 4. GAN 损失

**生成器损失**：
```python
g_fake = self.discriminator(images_recon)
g_loss = -g_fake.mean()  # 让判别器认为生成图像是真实的
```

**判别器损失**：
```python
d_real = self.discriminator(images_noisy.detach())
d_fake = self.discriminator(images_recon_noisy.detach())
d_loss_real = F.relu(1. - d_real).mean()
d_loss_fake = F.relu(1. + d_fake).mean()
gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)
```

**判别器权重因子**：
```python
disc_factor = VQGANLossHelper.adopt_weight(self.disc_factor, step, self.disc_start)
# 在 disc_start 步之前，disc_factor = 0
# 之后逐渐增加到 disc_factor
```

**自适应 lambda**：
```python
lambda_gan = VQGANLossHelper.calculate_lambda(
    decoder=self.decoder,
    perceptual_loss=perceptual_rec_loss,
    gan_loss=g_loss
)
# 根据感知损失和 GAN 损失的梯度比例自动调整权重
```

---

## 日志系统详解

### 日志管理器：`LogManager`

**位置**：`log_utils/log_utils.py` 第20-192行

**功能**：统一管理本地日志和 wandb 日志

#### 1. 初始化

```python
class LogManager:
    def __init__(self, log_dir, use_wandb=False, project_name=None, config=None, resume=False, id_name=""):
        # 1. 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 2. 初始化本地日志
        self.logger = self._build_logger(logger_name, log_dir)
        
        # 3. 初始化 wandb（如果启用）
        if use_wandb and _WANDB_AVAILABLE:
            self.run_wandb = wandb.init(
                project=project_name,
                job_type="train_model",
                config=config,
                id=id_name,
                resume="allow",
                dir="/data/huyang/wandb"
            )
```

#### 2. 日志记录方法

**记录指标**（第87-97行）：
```python
def log_metrics(self, step: int, metrics: Dict):
    if self.run_wandb is not None:
        self.run_wandb.log(metrics, step=step)
```

**记录图像**（第99-179行）：
```python
def log_images(self, step, images, images_recon, b, d, ...):
    # 1. 提取要可视化的帧
    images_frames = images_vis[0, :n_show]
    images_recon_frames = images_recon_vis[0, :n_show]
    
    # 2. 反归一化
    images_orig = unnormalize(images_frames).detach().cpu()
    images_recs = unnormalize(images_recon_frames).detach().cpu()
    
    # 3. 拼接图像（左右布局）
    combined_image = torch.cat([left_column, right_column], dim=2)
    
    # 4. 记录到 wandb
    self.run_wandb.log({
        'viz/recon-original': wandb.Image(combined_image_np, caption=f'...')
    }, step=step)
```

#### 3. 日志文件位置

**本地日志**：
- 路径：`{log_dir}/training_{timestamp}.log`
- 格式：`%(asctime)s - %(name)s - %(levelname)s - %(message)s`

**wandb 日志**：
- 项目名称：从配置文件读取 `project_name`
- Run ID：由 `build_id_name()` 生成（基于超参数）

### 日志记录时机

1. **每个 batch**：DEBUG 级别记录详细训练信息
2. **每 100 步**：INFO 级别记录进度和统计信息
3. **每 wandb_log_interval 步**：记录指标到 wandb
4. **每 wandb_image_interval 步**：记录图像到 wandb
5. **每 save_interval 步**：保存 checkpoint 并记录统计信息
6. **每 validation_interval 步**：验证并记录验证指标

---

## 配置文件详解

### 配置文件结构：`param.json`

```json
{
  "model_type": "vqvae2optvq",           // 模型类型
  "use_images": true,                     // 是否使用图片模式
  "scaling_rates": [4, 2],               // 下采样率（某些模型使用）
  
  "model_args": {                        // 模型参数
    "group_count": 5,
    "n_hid": 256,
    "vocab_size": 4096,
    "n_init": 1024,
    "input_channels": 3,
    "output_channels": 3,
    "commitment_cost": 0.25,
    "decay": 0.99
  },
  
  "data_args": {                         // 数据参数
    "batch_size": 1,
    "num_threads": 6,
    "device_id": 0,
    "training_data_files": [...],        // 训练数据文件列表
    "validation_data_files": [...],      // 验证数据文件列表
    "seed": 2025,
    "sequence_length": 8,
    "shard_id": 0,
    "num_shards": 1,
    "initial_prefetch_size": 1024
  },
  
  "train_args": {                        // 训练参数
    "num_steps": 300020,
    "lr": 0.0001,
    "lr_decay": 0.99,
    "folder_name": "/data/huyang/save_data",
    "checkpoint_path": null,             // null 表示从头开始
    "device": "auto",
    "save_interval": 10000,
    "validation_interval": 1000,
    "use_gan": true,                     // 是否使用 GAN
    "disc_factor": 0.8,                  // 判别器权重因子
    "disc_start": 50000,                 // 开始使用判别器的步数
    "pixel_loss_factor": 1.0,
    "perceptual_loss_factor": 1.0,
    "disc_lr": 1e-05,
    "disc_beta1": 0.5,
    "disc_beta2": 0.9
  },
  
  "use_wandb": false,                     // 是否使用 wandb
  "project_name": "train_vqgan_3.2.0_new_data"
}
```

### 关键参数说明

#### 模型参数（`model_args`）

- **group_count**：编码/解码组的数量，决定模型深度
- **n_hid**：初始隐藏特征维度
- **vocab_size**：码本大小，离散向量的数量
- **n_init**：编码器输出维度，也是 codebook 嵌入维度
- **commitment_cost**：VQ 损失中的 commitment cost
- **decay**：EMA 衰减率（如果使用 EMA 量化器）

#### 数据参数（`data_args`）

- **batch_size**：批次大小
- **sequence_length**：视频序列长度，一次处理的帧数
- **training_data_files**：训练数据文件列表（通过 `list_videos()` 生成）
- **validation_data_files**：验证数据文件列表

#### 训练参数（`train_args`）

- **num_steps**：训练总步数
- **lr**：初始学习率
- **lr_decay**：学习率衰减率（指数衰减）
- **save_interval**：保存 checkpoint 的间隔（步数）
- **validation_interval**：验证的间隔（步数）
- **use_gan**：是否使用 GAN 训练
- **disc_factor**：判别器权重因子
- **disc_start**：开始使用判别器的步数（在此之前 disc_factor=0）

---

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 安装 PyTorch（根据 CUDA 版本）
pip install torch torchvision

# 安装其他依赖（如需要）
pip install einops wandb lpips
```

### 2. 准备数据

**视频模式**：
```bash
# 将视频文件放在指定目录
/data/huyang/data/train/  # 训练数据
/data/huyang/data/vaild/  # 验证数据
```

**图片模式**：
```bash
# 将图片文件放在指定目录
/data/huyang/data/train/  # 训练数据
/data/huyang/data/vaild/  # 验证数据
```

### 3. 配置参数

编辑 `param.json`：
- 修改 `data_process` 中的数据路径
- 设置 `model_type` 选择模型类型
- 调整 `train_args` 中的训练参数

### 4. 启动训练

**方式 1：使用主启动器**
```bash
python main.py
# 选择单任务模式或多任务模式
```

**方式 2：直接运行训练脚本**
```bash
python train/train_custom_videos.py --device cuda:0 --config param.json
```

**方式 3：多任务模式**
```bash
# 在 multitask/ 目录下创建配置文件
# 例如：multitask/task1.json, multitask/task2.json
# 创建 multitask/config.json 配置 GPU 分配
python main.py
# 选择多任务模式
```

### 5. 监控训练

**本地日志**：
```bash
tail -f {log_dir}/training_*.log
```

**wandb**（如果启用）：
```bash
# 在浏览器中打开 wandb 项目页面
```

### 6. 恢复训练

在 `param.json` 中设置：
```json
{
  "train_args": {
    "checkpoint_path": "/path/to/checkpoint_step10000.pth.tar"
  }
}
```

---

## 文件结构说明

### 核心文件

1. **main.py**：训练启动器，支持单任务/多任务模式
2. **train/train_custom_videos.py**：训练主入口
3. **train/trainVqVae.py**：训练循环核心类
4. **models/model_adapter.py**：模型适配器系统
5. **train/video_utils.py**：数据加载工具
6. **train/lossconfig.py**：损失计算配置
7. **log_utils/log_utils.py**：日志管理系统

### 模型实现

- **models/vq_vae/**：VqVae2 模型实现
- **models/vqvae2/**：VQVAE2 模型实现
- **models/vqvae2_optvq/**：VQVAE2 OptVQ 模型实现
- **models/taming/**：Taming VQGAN 模型实现

### 工具模块

- **metric_utils/**：指标评估（PSNR, FID, LPIPS 等）
- **models/gan/**：GAN 相关（判别器、损失函数等）

---

## 总结

本项目通过**适配器模式**统一管理多种 VQ-VAE 模型，通过**配置驱动**的方式灵活切换模型类型和训练参数，提供了完整的训练、验证、日志记录和模型保存功能。核心特点：

1. **统一的模型接口**：所有模型通过适配器提供相同的接口
2. **灵活的数据加载**：支持视频和图片两种模式
3. **完整的损失系统**：支持重建损失、VQ 损失、感知损失、GAN 损失
4. **详细的日志记录**：本地日志 + wandb 集成
5. **模块化设计**：各组件独立，易于扩展

通过阅读本文档，您应该能够：
- 理解项目的整体架构和设计思路
- 了解模型适配器的工作原理
- 掌握数据加载和预处理的流程
- 理解训练循环和损失计算的细节
- 配置和启动训练任务

如有问题，请参考代码中的详细注释或查看相关文档。
