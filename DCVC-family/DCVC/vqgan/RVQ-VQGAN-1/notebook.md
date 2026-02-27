## 基于 VQGAN + RVQ + Transformer 熵模型的视频压缩方案说明文档

本项目实现了一条完整的“**VQGAN + 残差向量量化（RVQ）离散表征 → Transformer 概率模型 → 熵编码**”的视频压缩链路。  
该方案将高维视频信号映射为多层残差码本索引序列，并利用自回归 Transformer 学习索引的时空分布，以达到**端到端可学习的熵模型**，最终实现高效的视频压缩。

下面按论文撰写的思路，对整体框架、方法细节以及可作为创新点书写的内容进行系统梳理。

---

## 一、整体流程链路概述

- **输入**: 原始视频序列（或图像序列），分辨率统一到 \(256\times256\)。
- **步骤 1 – VQGAN+RVQ 表征学习**  
  使用基于 taming-transformers 改造的 VQGAN（模型类型 `TamingVQGAN`，变体 `EMAVQ`，带残差多层量化）对视频帧进行编码，得到：
  - 视觉质量较高的重建图像；
  - 多层残差 VQ（RVQ）产生的离散码本索引 \(z\_\text{idx}\)，形状为 \((B, D, H, W)\)，其中 \(D\) 为 RVQ 深度（本实现中为 8 层）。
- **步骤 2 – 从视频到索引序列的数据准备**  
  将视频分段为固定长度的时间片（例如 8 或 16 帧），对每段使用 VQGAN+RVQ 编码得到一串索引，并按 \(\text{(帧, RVQ 层, 空间位置)}\) 展开为一维 token 序列，作为 Transformer 的训练数据。
- **步骤 3 – Transformer 概率模型训练**  
  使用**仅解码器的自回归 Transformer + RoPE（旋转位置编码）**，对索引序列进行建模，学习条件分布：
  \[
    p(z\_t \mid z\_{<t})
  \]
  训练损失为交叉熵，对应的比特开销为平均负对数似然（NLL）转为 bit。
- **步骤 4 – 利用概率估算压缩率 / 可接入实际熵编码器**  
  当前实现中，直接用 Transformer 输出的概率估算理论压缩比特数：
  \[
    \text{bits} = -\sum\_t \log\_2 p(z\_t)
  \]
  再与固定码长 \(\log\_2 K\)（\(K\) 为码本大小）比较，得到压缩率；同时生成针对每个视频的 CSV，记录“原视频路径–重构视频路径–估算 size（字节）”。  
  后续可将该概率模型无缝接入算术编码 / ANS 等熵编码器，实现真正的比特流压缩。
- **输出**:  
  - 压缩后的视频比特估算（或实际比特流）；  
  - 通过 VQGAN 解码后的重构视频，以及对应的 PSNR / LPIPS / FID 等质量指标。

这一链路将“**感知质量导向的生成式图像编码（VQGAN）**”与“**指数族级别的强表达能力（Transformer）**”结合起来，实现了端到端可学习、可度量的视频压缩方案。

---

## 二、VQGAN + RVQ 视频表征前端

### 2.1 模型配置与结构

配置文件 `config.json` 中核心字段如下：

- **模型类型与变体**
  - `model_type: "TamingVQGAN"`：通过适配器封装 taming-vqgan。
  - `model_args.model_variant: "EMAVQ"`：采用 **EMA 码本更新** 的 VQ 变体，并在此基础上进一步扩展为残差多层量化（RVQ）。
- **码本与嵌入维度**
  - `n_embed = 16384`：码本大小 \(K\)，对应每个 token 的符号空间。
  - `embed_dim = 1024`：每个 codebook 向量的维度。
- **残差量化深度**
  - `rvq_levels = 8`：在 `EMAVQ` 变体中通过 `ResidualEMAVectorQuantizer` 将单层量化扩展为 8 层**共享码本的残差量化**，即 RQ-VAE 风格的 RVQ。
- **编码器/解码器结构 (`ddconfig`)**
  - 分辨率：`resolution = 256`，输入输出均为 \(256\times256\)。
  - 通道与层数：`ch=128`，`ch_mult=[1,1,2,2,4]`，`num_res_blocks=2`，在 16 分辨率层使用自注意力（`attn_resolutions=[16]`）。
  - 输入输出通道：`in_channels = 3`，`out_ch = 3`。

对应代码位于：

- VQ 结构主体：`models/taming/models/vqgan.py` 中的 `VQModel` / `EMAVQ`。
- RVQ 码本实现：`models/taming/modules/vqvae/quantize.py` 中的 `ResidualEMAVectorQuantizer`。
- 模型适配器：`models/model_adapter.py` 中的 `TamingVQGANAdapter` 与 `create_model`。

### 2.2 残差向量量化（RVQ）设计与实现

**基本思想**：在单一共享码本的基础上，迭代地对残差进行量化，得到深度为 \(D\) 的索引序列 \((k\_1,\dots,k\_D)\)。每一层的量化结果叠加起来近似原始连续特征：
\[
  z \approx \sum\_{d=1}^{D} e\_{k\_d}, \quad e\_{k\_d} \in \mathbb{R}^{\text{embed\_dim}}
\]

- **共享 EMA 码本**  
  `ResidualEMAVectorQuantizer` 使用单一 `EmbeddingEMA` 作为码本，对所有层共享参数。这意味着：
  - 各层在同一语义空间中分配码本向量；
  - 上层在下层的残差空间中继续细化表达。

- **多层残差迭代过程**（代码见 `ResidualEMAVectorQuantizer.forward`）：
  - 输入 flatten 后的特征 \(z\_\text{flat} \in \mathbb{R}^{N\times C}\)；
  - 令 `residual = z_flat`，循环 `num_quantizers = D` 次：
    1. 按最小欧式距离选择最近的码本索引 \(k_d\)；
    2. 得到当前层量化向量 \(e\_{k_d}\)，累加到 `quant_sum` 中；
    3. 更新残差 `residual = residual - e_{k_d}`；
    4. 保存各层的 residual 和 one-hot 编码以供后续 EMA 更新。
  - 最终输出：
    - `z_q`：所有层量化向量之和的映射回特征图；
    - `loss`：所有层的 commitment loss 之和；
    - `indices_stack`：形状为 \((B, D, H, W)\) 的多层码本索引。

- **EMA 更新策略**  
  通过 `EmbeddingEMA` 进行 EMA 更新：
  - 聚合所有层的 one-hot 编码和残差特征，作为新的聚类计数和中心；
  - 使用指数滑动平均更新 `cluster_size` 与 `embed_avg`，并归一化得到新的码本向量；
  - 此时码本向量**不参与梯度优化**，仅通过 EMA 更新，训练更稳定。

- **多层索引解码 (`embed_code`) 支持**  
  为了支持 RVQ 的多层索引重建，`ResidualEMAVectorQuantizer.embed_code` 显式支持：
  - 单层情况：\((B,H,W)\)；
  - 多层情况：\((B,D,H,W)\)。  
  对于多层情况，会对每个空间位置上 D 个 code 的 embedding 做求和，从而得到最终的量化特征。这一设计使得：
  - 可以直接根据不同的深度子集（例如只用前 2/4/6 层）重建图像；
  - 为后续在熵编码阶段按层级截断或自适应分配比特提供了结构基础。

**可写入论文的创新点**：

1. 在 taming-VQGAN 框架上引入**共享码本的 EMAVQ-RVQ**，将原有的单层量化扩展为多层残差量化，用统一 codebook 编码多级细节。
2. 实现了对 RVQ 深度的显式控制与逐层评估（见训练脚本中的 RVQ depth metrics），为研究“前几层粗码本 vs. 更多层精细码本”的率失真折中提供了实验工具。
3. `embed_code` 支持 \((B,D,H,W)\) 形状的多层索引解码，使得在“仅传输前若干层索引”的场景下可以方便地重建低比特率版本的视频，为多码率/渐进式视频传输提供了结构基础。

### 2.3 VQGAN 训练数据流与优化目标

训练脚本 `train/train.py` 构建了一个**最小但闭环完整**的 VQGAN 训练流程：

- **数据形态**（`data/datasets.py`）：
  - 支持两种模式：
    - `use_images = false`：从视频文件中随机采样长度为 `sequence_length` 的片段；
    - `use_images = true`：从图像目录中随机采样 `sequence_length` 张图片组成“伪视频”。
  - DataLoader 输出统一为 `uint8` 张量，形状 `(B, D, H, W, C)`，其中 `D = sequence_length`。
- **预处理与前向**：
  - 将 `(B,D,H,W,C)` 重新排列为 `(B*D, C, H, W)`；
  - 归一化到 \([0,1]\) 再线性映射到 \([-1,1]\)；
  - 进入 VQGAN 适配器 `TamingVQGANAdapter`，内部调用 `EMAVQ.encode` / `decode` 得到：
    - `vq_loss`（量化损失）；
    - `x_rec`（重建图像）；
    - `perplexity` 与 `encoding_indices`（索引用于后续分析）。
- **损失函数与优化**：
  - 使用 `VQLPIPSWithDiscriminator` 作为主要损失组件（感知重建 + 对抗损失 + 码本约束）；
  - 自编码器和判别器分别使用独立优化器，采用渐进训练；
  - 支持梯度累积（`grad_accum_steps`）与梯度裁剪，提升稳定性和大 batch 训练能力。
- **评估与 RVQ 分层指标**：
  - `evaluate` 函数中，除了整体 PSNR/FID/LPIPS，还会：
    - 调用 `taming_model.encode` 获取多层 RVQ 索引；
    - 对不同深度子集（如前 2 层、前 4 层等）通过 `embed_code` + `decode` 重建；
    - 分别计算每个深度下的 PSNR/LPIPS，得到 `val/psnr_rvq_d{depth}` 等指标。

**可写入论文的创新点**：

- 在训练阶段系统性地记录“不同 RVQ 深度的重建质量曲线”，使得 RVQ 结构不仅是一个编码器内部模块，更成为一个**可显式调节的多码率工具**。  
- 相比传统单层 VQ，本方案在模型评估阶段就已经内嵌了“按 RVQ 层数解码”的实验接口，为后续视频压缩中“按场景自适应选择 RVQ 深度”提供了坚实的实验与工程基础。

---

## 三、从视频到索引序列的数据准备

Transformer 训练并不直接面对原始像素，而是面对 VQGAN+RVQ 产生的码本索引序列。相关逻辑集中在 `transformer/data_prepare.py`。

### 3.1 VQGAN 模型加载与索引提取

- 使用 `load_vqgan_model(vq_config, vq_ckpt, device)`：
  - 读取 VQGAN 配置 `vq_config`（此处可直接复用前面训练好的 `config.json`）；
  - 通过 `create_model(config_path, model_args)` 构造 `TamingVQGANAdapter`；
  - 从 checkpoint 中加载权重，并切换到 eval 模式。
- 索引提取函数 `_get_indices_from_encode(vqgan_model, x)`：
  - 调用适配器内部的 `inner.encode(x)`，其中 `inner` 是底层 `EMAVQ` 模型；
  - 从返回的 `info[2]` 中拿到 `indices`：
    - 单层 VQ：`indices` 为一维向量，需 reshape 为 `(B,H,W)`；
    - 多层 RVQ：`indices` 直接为 `(B,D,H,W)`。

### 3.2 视频级索引编码

- `encode_video_to_indices(vqgan_model, video, device)`：
  - 输入视频形状为 `(num_frames, H, W, C)`；
  - 转换为 `(num_frames, C, H, W)` 并归一化到 `[-1,1]`；
  - 按 batch 维度切分帧，逐批调用 `_get_indices_from_encode`；
  - 拼接得到整段视频的索引：
    - 单层 VQ：`(num_frames, H', W')`；
    - 多层 RVQ：`(num_frames, D, H', W')`。

### 3.3 切分为 Transformer 训练样本

- `process_videos_to_indices(...)` 负责将视频目录转换为训练数据矩阵：
  - 对每个视频：
    1. 加载并统一分辨率的所有帧；
    2. `encode_video_to_indices` 得到索引张量 `indices`；
    3. 根据维度判断单层还是多层 RVQ：
       - 单层：`(num_frames, ih, iw)`；
       - 多层：`(num_frames, D, ih, iw)`。
    4. 按 `sequence_length` 划分时间片：`num_segments = num_frames // sequence_length`；
    5. 对每一段 `indices[start:end]` 做 `.flatten()`：
       - 单层：序列长度 `sequence_length * ih * iw`；
       - RVQ：序列长度 `sequence_length * D * ih * iw`；
    6. 转为 `np.int64` 追加到 `all_samples`。
  - 最终得到形状为 `[N, seq_len]` 的二维数组，直接作为 Transformer 输入。

### 3.4 验证集：索引 + 重构视频 + 元信息

- `process_val_videos_to_indices_and_recon(...)` 在编码验证视频时会同时：
  - 使用 VQGAN 解码整个（或完整片段）索引序列，生成重构视频；
  - 将重构视频和原视频裁剪到相同帧数，分别保存到 `recon_output_dir` 和 `original_output_dir`；
  - 同样按 `sequence_length` 切分索引，生成 `val_data`（形状 `[N, seq_len]`）；
  - 为每一段构造 `meta` 字典：
    - `{"original_path", "recon_path", "num_indices", "segment_idx", "video_path"}`；
  - 返回 `(val_data, val_meta)`，供后续 Transformer 验证和 CSV 统计使用。

**可写入论文的创新点**：

1. 将**视频压缩问题形式化为离散索引序列建模问题**：通过 VQGAN+RVQ 将连续像素空间映射到有限码本索引空间，进而将原本高维连续视频压缩问题转化为离散序列建模问题。
2. 在数据准备阶段显式支持 RVQ 输出的 `(num_frames, D, H, W)` 结构，并将“帧维度 + RVQ 层维度 + 空间位置维度”统一展开为单一 token 序列，使得后续 Transformer 能同时学习**时间、空间与层级残差的联合统计结构**。
3. 验证集同时生成“原视频片段–重构视频片段–索引序列”的三元组，使得后续可以针对任意编码策略（例如仅传输部分索引或剪裁层数）直接衡量感知质量与压缩率，为系统性率失真实验提供基础。

---

## 四、Transformer 熵模型与概率估算压缩

Transformer 概率模型及训练流程集中在 `transformer/train_transformer.py` 与 `transformer/encode_methods.py`。

### 4.1 索引序列数据集

- `VQIndexDataset`  
  - 输入：形状为 `[N, seq_len]` 的 `np.ndarray`；
  - 每次返回一个 `torch.LongTensor(seq_len)`，作为单个训练样本。
- `ValIndexDataset`  
  - 用于验证集，返回 `(indices, meta)`，其中：
    - `indices`: `[seq_len]`；
    - `meta`: 对应前面 `process_val_videos_to_indices_and_recon` 生成的 metadata，用于按视频/片段聚合压缩大小并写入 CSV。

### 4.2 Transformer 结构：Decoder-only + RoPE

核心模块包括：

- **多头自注意力 + RoPE**（`MultiheadAttention`）：
  - 将输入 `x` 映射为 query/key/value，再按标准多头自注意力计算注意力权重；
  - 使用**旋转位置编码（RoPE）**增强对相对位置关系的建模能力（对长序列更稳定）：
    - 构造 `sin/cos` 位置嵌入；
    - 对 `q,k` 的奇偶维度做旋转变换，实现相位编码；
  - mask 采用标准上三角因果 mask，确保自回归（仅依赖历史）。

- **前馈网络**（`FeedForward`）：
  - 使用门控结构：`gate = sigmoid(W1 x), transformed = ReLU(W2 x), out = W3 (gate * transformed)`；
  - 相比单一线性 + 激活，多了一个 gate，有利于提高表达能力与稳定性。

- **DecoderLayer**：
  - 结构：`x = x + MHA(LN(x)); x = x + FFN(LN(x))`；
  - 无编码器，仅堆叠解码层，典型的 decoder-only Transformer。

- **TransformerProbabilityModel**：
  - 词表大小：`vocab_size = num_codes + 1`，多出的一个 token 作为 BOS：
    - `bos_token_id = num_codes`；
  - 嵌入层：`nn.Embedding(vocab_size, d_model)`；
  - 堆叠若干 `DecoderLayer`（默认 `num_layers=6`，可配置）；
  - 输出头：线性层 `fc_out` 映射到 `num_codes` 维概率分布（不预测 BOS）。
  - 前向接口：
    - 输入：`x` 为索引序列，形状 `[B, L]`；
    - 输出：`logits, probs`，其中 `probs` 为 `softmax(logits)`。

- **TransformerEntropyModel**：
  - 封装 `TransformerProbabilityModel`，在 `forward` 中直接返回：
    - `rate`：即平均 NLL 换算到 bit；
    - `probs`：用于后续压缩率估算或熵编码。
  - 计算方式：
    - 将输入序列 `indices`（含 BOS）通过 Transformer 得到 logits；
    - `nll = cross_entropy(logits[:, :-1], target_indices=indices[:, 1:])`；
    - `rate = nll / ln 2`，即平均每 token 的 bit 数。

**可写入论文的创新点**：

1. 将 RVQ 码本索引作为 token，使用**仅解码器 Transformer + RoPE**学习其长程依赖，统一建模“时间–空间–层级残差”联合分布，相较于传统基于 Markov / 低阶上下文的熵模型具有更强表达能力。
2. 通过显式引入 BOS token，将视频段内索引序列视为条件生成过程，有利于在后续拓展为“跨视频 / 条件视频生成 + 压缩”的统一框架。

### 4.3 训练流程与优化目标

- **训练主循环（`train_epoch`）**：
  - 对每个 batch 的索引 `indices`（形状 `[B,L]`）：
    1. 在首位拼接 BOS，得到 `full = [BOS, z_1, ..., z_L]`；
    2. 模型输入为 `input_seq = full[:, :-1]`，目标为 `target_seq = full[:, 1:]`；
    3. 使用上三角因果 mask 保证自回归；
    4. 损失函数：交叉熵 NLL，换算到 bit 作为 `rate`；
    5. 计算预测准确率 `acc`（argmax 与 target 比较）；
    6. 支持混合精度（`torch.amp.autocast`）、梯度累积与梯度裁剪。

- **验证阶段（`validate`）**：
  - 与训练相同的 forward 过程，但不反向传播；
  - 返回 `(avg_loss, avg_rate, avg_acc)`；
  - 用 `ReduceLROnPlateau` scheduler 按验证损失自适应调节学习率。

- **模型保存与最优模型选择**：
  - 保存包含 `epoch`、`model_state_dict`、`optimizer_state_dict` 等的 checkpoint；
  - 以最小验证 `rate` 为标准保存 best model。

### 4.4 基于概率的压缩率估算 & CSV 评估

两处关键函数：

- `estimate_bits_from_probs(probs, indices)` / `estimate_compressed_bytes_from_probs`：
  - 对于每个 token：
    - 取出对应真实索引的预测概率 \(p(z_t)\)；
    - 计算 `nll_bits = -log2(p)`；
  - 对时间维度求和得到每个样本的总 bit 数；
  - 除以 8 后即得到估算的字节数。

- `evaluate_compression_rate_by_probs(model, dataloader, device, vocab_size)`：
  - 遍历验证集索引序列：
    - 对每段样本计算平均 `bits/index`；
    - 与固定码长 `log2(vocab_size)` 对比，得到压缩比 `ratio`。

- `evaluate_and_write_csv(...)`：
  - 结合 `ValIndexDataset` 提供的 meta 信息，对每个 `(original_path, recon_path)` 聚合 sum bytes；
  - 写入 CSV，包含列：`orig, recon, size`（size 为估算的压缩后字节数）。

**可写入论文的创新点**：

1. 使用 Transformer 输出的概率直接估算**信息论意义上的极限压缩率**（不依赖具体熵编码实现），从而将模型性能与 Shannon 理论界联系起来。
2. 基于 `orig/recon/size` 的 CSV 评估格式，为后续接入任意类型的熵编码器（例如 ANS / range coding）提供统一接口，只需将概率密度替换为真实编码器写入的比特数即可对比“理论上限 vs. 实际编码器效率”。

---



## 六、论文撰写建议结构提示

基于以上实现，论文主体结构可以按如下顺序组织：

- **方法部分**
  - 介绍 VQGAN+RVQ 前端（网络结构、RVQ 原理、训练损失、分层评估指标等）；
  - 介绍从视频到索引序列的数据准备与切分策略；
  - 介绍 Transformer 熵模型结构（RoPE、多头注意力、门控前馈等）以及自回归训练目标；
  - 介绍基于概率的压缩率估算公式与整体编码–解码流程。
- **实验部分**
  - 报告不同 RVQ 深度下的图像/视频重建质量曲线（PSNR / LPIPS / FID）；
  - 报告在给定 VQ 码本大小下，Transformer 模型达到的平均 bits/index 以及与固定码长的压缩比；
  - 对比不使用 Transformer 熵模型（假设均匀分布）的 baseline，展示学习到的概率模型在压缩上的收益；
  - 展示主观重建视频对比图（原始 vs. 重构）以及在不同 RVQ 深度下的可视化对比。

在写作时，可将本 notebook 中的内容作为“方法 + 系统实现细节”的蓝本，视篇幅对具体实现参数（如维度、层数、学习率）做适当裁剪或补充。

