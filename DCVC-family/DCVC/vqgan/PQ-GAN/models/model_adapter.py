"""
VQGAN - 模型适配器（最小版）

目标：
- 保持 project/models/model_adapter.py 的“适配器 + 工厂函数”架构
- 但只支持 taming-vqgan 这一种模型
- 不依赖 project/ 目录（taming 源码已拷贝到 VQGAN/models/taming）
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch import nn


def _load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


class BaseVqVaeAdapter(nn.Module):
    """
    适配器基类：对外暴露统一接口

    统一 forward 输出：
        (vq_loss, images_recon, perplexity, encoding_indices)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        object.__setattr__(self, "_model", model)

    def _get_model(self) -> nn.Module:
        return object.__getattribute__(self, "_model")

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self._get_model()(x)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._get_model(), name)

    def to(self, device):
        m = object.__getattribute__(self, "_model")
        object.__setattr__(self, "_model", m.to(device))
        return self


class TamingVQGANAdapter(BaseVqVaeAdapter):
    """
    适配 taming 的 VQModel / EMAVQ

    将 taming 的 encode/decode 与 emb_loss/info 转成训练脚本需要的输出格式。
    """

    @staticmethod
    def _convert_params(params: Dict[str, Any]) -> Dict[str, Any]:
        converted = params.copy()
        if "input_channels" in converted and "ddconfig" not in converted:
            converted["ddconfig"] = {"in_channels": converted.pop("input_channels")}
        if "vocab_size" in converted and "n_embed" not in converted:
            converted["n_embed"] = converted.pop("vocab_size")
        if "n_init" in converted and "embed_dim" not in converted:
            converted["embed_dim"] = converted.pop("n_init")
        return converted

    def __init__(self, model_args: Dict[str, Any], config_path: str):
        from models.taming.models.vqgan import EMAVQ, VQModel, PQEMAVQ

        config = _load_config(config_path)
        cfg_args = config.get("model_args", {})
        merged = {**cfg_args, **(model_args or {})}
        merged = self._convert_params(merged)

        model_variant = merged.pop("model_variant", "VQModel")
        ddconfig = merged.pop("ddconfig")
        lossconfig = merged.pop("lossconfig", None)
        n_embed = merged.pop("n_embed")
        embed_dim = merged.pop("embed_dim")
        ckpt_path = merged.pop("ckpt_path", None)

        if lossconfig is None:
            lossconfig = {
                "target": "models.taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
                "params": {
                    "disc_conditional": False,
                    "disc_in_channels": ddconfig.get("in_channels", 3),
                    "disc_start": 10000,
                    "disc_weight": 0.8,
                    "codebook_weight": 1.0,
                },
            }

        if model_variant == "EMAVQ":
            model = EMAVQ(
                ddconfig=ddconfig,
                lossconfig=lossconfig,
                n_embed=n_embed,
                embed_dim=embed_dim,
                ckpt_path=ckpt_path,
                ignore_keys=[],
                image_key="image",
                colorize_nlabels=None,
                monitor=None,
                remap=None,
                sane_index_shape=False,
            )
        elif model_variant == "PQEMAVQ":
            # 额外的 PQ 超参数，从 config / model_args 读取
            pq_num_subspaces = int(merged.pop("pq_num_subspaces", 4))
            pq_decay = float(merged.pop("pq_decay", 0.99))
            pq_eps = float(merged.pop("pq_eps", 1e-5))

            model = PQEMAVQ(
                ddconfig=ddconfig,
                lossconfig=lossconfig,
                n_embed=n_embed,
                embed_dim=embed_dim,
                ckpt_path=ckpt_path,
                ignore_keys=[],
                image_key="image",
                colorize_nlabels=None,
                monitor=None,
                remap=None,
                sane_index_shape=False,
                pq_num_subspaces=pq_num_subspaces,
                pq_decay=pq_decay,
                pq_eps=pq_eps,
            )
        else:
            model = VQModel(
                ddconfig=ddconfig,
                lossconfig=lossconfig,
                n_embed=n_embed,
                embed_dim=embed_dim,
                ckpt_path=ckpt_path,
                ignore_keys=[],
                image_key="image",
                colorize_nlabels=None,
                monitor=None,
                remap=None,
                sane_index_shape=False,
            )

        model.automatic_optimization = False

        lr = config.get("train", {}).get("lr", 1e-4)
        model.learning_rate = float(lr)

        super().__init__(model)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        m = self._get_model()
        quant, emb_loss, info = m.encode(x)
        images_recon = m.decode(quant)

        if info is not None and len(info) >= 3:
            perplexity = info[0]
            encoding_indices = info[2]
            if perplexity is None:
                perplexity = torch.tensor(1.0, device=x.device)
            if encoding_indices is not None and encoding_indices.ndim == 1:
                b, _, h, w = quant.shape
                encoding_indices = encoding_indices.view(b, h, w)
        else:
            perplexity = torch.tensor(1.0, device=x.device)
            encoding_indices = None

        vq_loss = emb_loss
        return vq_loss, images_recon, perplexity, encoding_indices

    @property
    def output_channels(self) -> int:
        m = self._get_model()
        return m.decoder.conv_out.out_channels


_MODEL_REGISTRY: Dict[str, Type[BaseVqVaeAdapter]] = {
    "TamingVQGAN": TamingVQGANAdapter,
    "taming_vqgan": TamingVQGANAdapter,
    "VQModel": TamingVQGANAdapter,
    "EMAVQ": TamingVQGANAdapter,
}


def create_model(config_path: str, model_args: Optional[Dict[str, Any]] = None) -> BaseVqVaeAdapter:
    config = _load_config(config_path)
    model_type = config.get("model_type", "TamingVQGAN")
    adapter_cls = _MODEL_REGISTRY[model_type]
    return adapter_cls(model_args or {}, config_path=config_path)

