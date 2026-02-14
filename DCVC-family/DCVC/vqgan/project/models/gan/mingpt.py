"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    """
    多头掩码自注意力层（带投影），仿照GPT自注意力实现。
    支持推理增量计算，生成时可缓存历史，训练时常规前向传播。
    """

    def __init__(self, config):
        super().__init__()
        # 确保embedding维度可均分给n_head个头
        assert config.n_embd % config.n_head == 0
        # Q、K、V变换，全头共享投影但每头切分
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # 对注意力概率做dropout防止过拟合
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        # 对残差连接做dropout
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # 所有头的输出拼接后再做一次线性投影
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # 构造下三角mask，保证只能关注自身左侧（含本位）
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        # 若配置支持n_unmasked，则前n_unmasked行全都能看到（如全局token等用法）
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        # 注册为buffer（不会作为参数更新，但会随模型迁移设备）
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head  # 注意力头数

    def forward(self, x, layer_past=None):
        """
        :param x: 输入，shape (B, T, C)，B=batch，T=序列长度，C=embedding维数
        :param layer_past: 缓存的(k,v)，推理用，加速自回归采样
        :return: (输出y, 本步的present(k,v))
        """
        B, T, C = x.size()

        # 计算Q,K,V张量，并拆分成多头，shape: (B, n_head, T, head_dim)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 用于推理采样时缓存历史K,V（增量注意力）
        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            # 连接历史K,V（跨步采样用）
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # 注意力分数，缩放点积注意力
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 只在训练或首次采样时进行mask（实际只对生成最右侧T步有用，历史内容已mask过）
        if layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        # softmax归一化后再dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # 用注意力权重聚合V：加权合并各位置信息，输出(B, n_head, T, head_dim)
        y = att @ v
        # 恢复为(B, T, C)，多头拼回原维度
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影加残差dropout
        y = self.resid_drop(self.proj(y))
        return y, present  # 返回输出和present(便于采样连续地拼接KV历史)


class Block(nn.Module):
    """ 一个简洁的Transformer模块。实现了标准的Attention+前馈结构。 """

    def __init__(self, config):
        super().__init__()
        # 第一层归一化
        self.ln1 = nn.LayerNorm(config.n_embd)
        # 第二层归一化
        self.ln2 = nn.LayerNorm(config.n_embd)
        # 多头自注意力模块
        self.attn = CausalSelfAttention(config)
        # 前馈网络（MLP），包括两层全连接（其中隐层为4倍扩展），GELU激活，末尾加残差Dropout
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # GELU激活函数
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        # 前向推理，支持output present（仅推理时，便于采样）
        # x: 输入特征，形状(B, T, n_embd)
        # layer_past: 用于增量推理（可跳过已算部分）
        # return_present: 若为True, 返回attention状态(present)供采样
        if return_present:
            # 要返回present时说明是采样/推理，不能在训练模式
            assert not self.training
        # 归一化后送入自注意力
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)
        # 残差连接加回原始输入
        x = x + attn
        # 第二次归一化后送入前馈MLP，再与上一步残差
        x = x + self.mlp(self.ln2(x))
        # 推理模式或传入layer_past时返回present（用于缓存高效采样）
        if layer_past is not None or return_present:
            return x, present
        # 否则只返回输出
        return x


class GPT(nn.Module):
    """GPT模型：标准的Transformer解码器架构，用于语言建模或类似任务，可配置自注意力堆叠层数等"""

    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        """
        初始化GPT模型

        Args:
            vocab_size (int): 词表大小
            block_size (int): 上下文序列最大长度
            n_layer (int): Transformer Block 堆叠的层数
            n_head (int): 多头自注意力中的头数
            n_embd (int): 每个token的embedding维度
            embd_pdrop (float): embedding层的dropout概率
            resid_pdrop (float): 残差连接的dropout概率
            attn_pdrop (float): 自注意力权重的dropout概率
            n_unmasked (int): 可选参数，没被掩盖的token数量，默认为0
        """
        super().__init__()
        # 构建模型配置
        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_unmasked=n_unmasked
        )
        # 词嵌入层：将token id映射为向量
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # 位置编码：可学习的位置向量，用于区分序列中不同位置
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        # embedding之后的dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        # Transformer主体：由若干Block（自注意力+前馈网络）组成，使用nn.Sequential按顺序堆叠
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # 最后归一化层
        self.ln_f = nn.LayerNorm(config.n_embd)
        # 输出头，将Block输出变换为词表大小的logits（用于softmax）
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 记录模型支持的最大序列长度
        self.block_size = config.block_size
        # 权重初始化
        self.apply(self._init_weights)
        # 保存config，便于后续引用
        self.config = config

    def get_block_size(self):
        """返回此模型可接受的最大序列长度"""
        return self.block_size

    def _init_weights(self, module):
        """根据GPT论文的规范初始化权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 权重正态分布初始化
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                # 偏置置零
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # LN的偏置置零，权重初始化为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None):
        """
        前向传播

        Args:
            idx (LongTensor): shape (B, T)，每个元素是token的索引
            embeddings (Tensor, optional): 额外需要拼接在输入前面的嵌入向量，shape为(B, n_prepended, n_embd)

        Returns:
            logits (Tensor): shape (B, T, vocab_size)，每个位置的输出logits
            None: 保留接口一致性（可用于未来返回cache等）
        """
        # 1. 词嵌入映射，将token id转换为向量表示 (B, T, n_embd)
        token_embeddings = self.tok_emb(idx)

        # 2. 如果有额外提供的embeddings，则将其拼接到token_embeddings前面
        if embeddings is not None:
            # embeddings: 形状为(B, n_prepended, n_embd)
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        # 当前序列长度（可能大于原输入，如果有额外embedding）
        t = token_embeddings.shape[1]
        assert t <= self.block_size, "输入长度超过了模型的block_size上限"
        # 3. 位置向量，与token_embeddings对应位置相加
        position_embeddings = self.pos_emb[:, :t, :]  # (1, t, n_embd)
        # 4. embedding+位置向量，输入dropout
        x = self.drop(token_embeddings + position_embeddings)
        # 5. 依次通过若干Transformer Block
        x = self.blocks(x)
        # 6. 输出层归一化
        x = self.ln_f(x)
        # 7. 变换为词表大小的logits，后续可用softmax得到概率分布
        logits = self.head(x)

        return logits, None  # None为占位，方便后续拓展（如返回cache/presents）

