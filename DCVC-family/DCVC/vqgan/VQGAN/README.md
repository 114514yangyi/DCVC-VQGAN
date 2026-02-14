## VQGAN（独立版，仅 taming-vqgan）

这个目录是从 `project/` 中**抽离出来的最小可训练子工程**，只保留 taming-vqgan（`VQModel/EMAVQ`）的训练相关代码，并保留：
- **本地日志**（控制台 + 文件）
- **checkpoint 保存/恢复**
- **model_adapter 适配器架构**（但只注册一个模型）

### 目录结构

- `VQGAN/models/taming/`: taming-vqgan 原始模型代码（从 `project/models/taming` 复制）
- `VQGAN/models/model_adapter.py`: 适配器 + 工厂函数（仅 `TamingVQGANAdapter`）
- `VQGAN/data/datasets.py`: 最小数据集/loader（视频或图片序列）
- `VQGAN/log_utils/log_utils.py`: 最小日志器
- `VQGAN/train/train.py`: 最小训练脚本
- `VQGAN/config.json`: 最小配置示例
- `VQGAN/metric_utils/metric_utils.py`: 验证指标（PSNR + 可选 FID/LPIPS）

### 训练

1. 修改 `VQGAN/config.json`：
   - `data.train_dir` 指向训练数据目录（视频模式：目录下放 `*.mp4` 等；图片模式：目录下放 `*.jpg/*.png`）
   - 需要图片模式则设 `data.use_images=true`
   - 如需验证/保存 best model：设置 `data.val_dir`，并配置 `train.validation_interval`
   - 如需 wandb：设置 `wandb.enabled=true`

2. 安装依赖（建议在虚拟环境中）：

```bash
pip install -r VQGAN/requirements.txt
```

3. 启动训练（在仓库根目录）：

```bash
python VQGAN/train/train.py --config VQGAN/config.json
```

### 输出

- 日志：`train.log_dir/` 下 `train_*.log`
- checkpoint：`train.save_dir/` 下 `checkpoint_step*.pt`

