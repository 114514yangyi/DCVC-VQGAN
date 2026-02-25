import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    # NOTE: this class contains a bug regarding beta; see VectorQuantizer2 for
    # a fix and use legacy=False to apply that fix. VectorQuantizer2 can be
    # used wherever VectorQuantizer has been used before and is additionally
    # more efficient.
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        #.........\end

        # with:
        # .........\start
        #min_encoding_indices = torch.argmin(d, dim=1)
        #z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class GumbelQuantize(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:,self.used,...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:,self.used,...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b*h*w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q


class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps        
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad = False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad = False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)   


class EMAVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5,
                remap=None, unknown_index="random"):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.n_embed = n_embed
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        b,c,h,w = z.shape
        z = rearrange(z, 'b c h w -> b h w c')
        
        z_flattened = z.reshape(-1, self.codebook_dim)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight) # 'n d -> d n'


        encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)     
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if self.training and self.embedding.update:
            #EMA cluster size
            encodings_sum = encodings.sum(0)            
            self.embedding.cluster_size_ema_update(encodings_sum)
            #EMA embedding average
            embed_sum = encodings.transpose(0,1) @ z_flattened            
            self.embedding.embed_avg_ema_update(embed_sum)
            #normalize embed_avg and update weight
            self.embedding.weight_update(self.num_tokens)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z) 

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q, 'b h w c -> b c h w'
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        encoding_indices = encoding_indices.reshape(b,h,w)
        return z_q, loss, (perplexity, encodings, encoding_indices)

    def embed_code(self, indices):
        """
        根据给定的 codebook indices 返回对应的 embedding
        
        Args:
            indices: codebook indices, shape (batch, height, width) or (batch, seq_len)
        
        Returns:
            z_q: quantized embedding vectors
                - If input is (batch, h, w): returns (batch, channel, height, width)
                - If input is (batch, seq_len): returns (batch, seq_len, channel)
        """
        # 保存原始形状
        original_shape = indices.shape
        
        # 如果支持 remap，先将 remapped indices 转换回原始 indices
        if self.remap is not None:
            # unmap_to_all 要求至少2个维度
            if len(original_shape) == 3:
                # (batch, h, w) -> (batch, h*w)
                indices = indices.reshape(original_shape[0], -1)
            elif len(original_shape) == 2:
                # (batch, seq_len) 或 (batch, h*w)，保持原样
                pass
            else:
                # 其他情况，添加 batch 维度
                indices = indices.unsqueeze(0)
            
            # 执行 unmap
            indices = self.unmap_to_all(indices)
            # 展平以便进行 embedding lookup
            indices = indices.reshape(-1)
        else:
            # 展平 indices 以便进行 embedding lookup
            indices = indices.reshape(-1)
        
        # 确保 indices 是 long 类型（embedding 需要 Long 或 Int 类型）
        indices = indices.long()
        
        # 获取对应的 embedding
        z_q = self.embedding(indices)
        
        # 根据原始形状重塑
        if len(original_shape) == 3:
            # (batch, h, w) -> (batch, h, w, c) -> (batch, c, h, w)
            b, h, w = original_shape
            z_q = z_q.view(b, h, w, self.codebook_dim)
            z_q = rearrange(z_q, 'b h w c -> b c h w')
        elif len(original_shape) == 2:
            # (batch, seq_len) -> (batch, seq_len, c)
            b, seq_len = original_shape
            z_q = z_q.view(b, seq_len, self.codebook_dim)
        else:
            # 保持扁平形状
            pass
        
        return z_q


class ResidualEMAVectorQuantizer(nn.Module):
    """
    RQ-VAE 风格的多层残差量化器，使用单一共享 EMA 码本。

    - 参考 "Autoregressive Image Generation using Residual Quantization" 中的 RQ-VAE：
      给定深度 D，使用同一个 codebook 递归量化残差 r_{d-1}，得到编码 (k1,...,kD)。
    - 这里沿用上面的 EMA 更新策略，对所有深度的聚类结果累积做一次 EMA 更新。
    - 返回的 indices 形状为 [B, D, H, W]，方便后续 transformer 按“深度维”展开。
    """

    def __init__(
        self,
        n_embed: int,
        embedding_dim: int,
        beta: float,
        num_quantizers: int = 1,
        decay: float = 0.99,
        eps: float = 1e-5,
        remap=None,
        unknown_index: str = "random",
    ):
        super().__init__()
        assert num_quantizers >= 1, "num_quantizers 必须 >= 1"
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.n_embed = n_embed
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.num_quantizers = num_quantizers

        # 单一共享 codebook（EMA）
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(
                0, self.re_embed, size=new[unknown].shape
            ).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        """
        输入:
            z: [B, C, H, W]

        输出:
            z_q: 量化后的特征 [B, C, H, W]，为所有深度编码向量之和
            loss: 多层 commitment loss 之和（对应论文 Eq.(7)）
            info: (perplexity, encodings, indices)，其中:
                  - perplexity: 依据所有深度的使用频率统计
                  - encodings: one-hot 编码，形状 [B*H*W*D, K]
                  - indices: 码本索引 [B, D, H, W]
        """
        b, c, h, w = z.shape
        # [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
        z_hw = rearrange(z, "b c h w -> b h w c")
        z_flat = z_hw.reshape(-1, self.codebook_dim)

        residual = z_flat
        all_indices = []
        all_encodings = []
        residuals = []
        partial_sums = []

        quant_sum = torch.zeros_like(z_flat)

        for _ in range(self.num_quantizers):
            # 记录当前深度量化前的 residual（对应 r_{d-1}）
            residuals.append(residual)

            d = (
                residual.pow(2).sum(dim=1, keepdim=True)
                + self.embedding.weight.pow(2).sum(dim=1)
                - 2 * torch.einsum("bd,nd->bn", residual, self.embedding.weight)
            )

            encoding_indices = torch.argmin(d, dim=1)
            all_indices.append(encoding_indices)

            z_q_d = self.embedding(encoding_indices)  # [N, C]
            quant_sum = quant_sum + z_q_d

            # one-hot 编码 [N, K]
            encodings = F.one_hot(encoding_indices, self.num_tokens).type(z_flat.dtype)
            all_encodings.append(encodings)

            # 更新 residual
            residual = residual - z_q_d

            # 记录部分和，对应 ˆZ^(d)
            partial_sums.append(quant_sum.clone())

        # EMA 更新（对所有深度一起）
        if self.training and self.embedding.update:
            residual_cat = torch.cat(residuals, dim=0)  # [N*D, C]
            encodings_cat = torch.cat(all_encodings, dim=0)  # [N*D, K]
            encodings_sum = encodings_cat.sum(0)
            self.embedding.cluster_size_ema_update(encodings_sum)

            embed_sum = encodings_cat.transpose(0, 1) @ residual_cat
            self.embedding.embed_avg_ema_update(embed_sum)
            self.embedding.weight_update(self.num_tokens)

        # commitment loss: 对每一层的部分和都计算一次，并求和
        loss = 0.0
        for ps in partial_sums:
            loss = loss + F.mse_loss(ps.detach(), z_flat)
        loss = self.beta * loss

        # preserve gradients（对最终 ˆZ(D) 做直通）
        z_q = quant_sum.view(z_hw.shape)
        z_q = rearrange(z_q, "b h w c -> b c h w")
        z_q = z + (z_q - z).detach()

        # perplexity 统计：基于所有深度的 one-hot 使用频率
        encodings_cat = torch.cat(all_encodings, dim=0)
        avg_probs = torch.mean(encodings_cat, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # indices 形状 [B, D, H, W]
        indices_stack = torch.stack(all_indices, dim=1)  # [N, D]
        indices_stack = indices_stack.view(b, h, w, self.num_quantizers)
        indices_stack = rearrange(indices_stack, "b h w d -> b d h w")

        return z_q, loss, (perplexity, encodings_cat, indices_stack)

    def embed_code(self, indices):
        """
        根据给定的 codebook indices 返回对应的 embedding 并做“多层残差和”。

        支持两种主要形状:
            - [B, H, W]：退化为单层 EMA 量化，行为同 EMAVectorQuantizer
            - [B, D, H, W]：D 层残差量化，按深度求和得到最终特征
        """
        original_shape = indices.shape

        # 支持 remap
        if self.remap is not None:
            if len(original_shape) >= 2:
                b = original_shape[0]
                rest = int(np.prod(original_shape[1:]))
                inds = indices.reshape(b, rest)
            else:
                inds = indices.unsqueeze(0)
                b = 1
            inds = self.unmap_to_all(inds)
            indices = inds.reshape(original_shape)

        if indices.dim() == 3:
            # [B, H, W] -> 与 EMAVectorQuantizer.embed_code 一致
            b, h, w = indices.shape
            flat = indices.reshape(-1)
            flat = flat.long()
            z_q = self.embedding(flat)
            z_q = z_q.view(b, h, w, self.codebook_dim)
            z_q = rearrange(z_q, "b h w c -> b c h w")
            return z_q

        if indices.dim() == 4:
            # [B, D, H, W]，每个位置有 D 个 code，相加得到最终 embedding
            b, d, h, w = indices.shape
            # [B, D, H, W] -> [B, H, W, D]
            inds = rearrange(indices, "b d h w -> b h w d").contiguous()
            # -> [B, H*W, D]
            inds = inds.view(b, h * w, d)
            # 利用 F.embedding 的广播能力: [B, HW, D, C]
            z_q_all = F.embedding(inds.long(), self.embedding.weight)
            # 沿深度求和 -> [B, HW, C]
            z_q = z_q_all.sum(dim=2)
            z_q = z_q.view(b, h, w, self.codebook_dim)
            z_q = rearrange(z_q, "b h w c -> b c h w")
            return z_q

        # 其他情况，退化为简单 lookup
        flat = indices.reshape(-1).long()
        z_q = self.embedding(flat)
        return z_q
