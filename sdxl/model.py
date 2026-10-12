"""Native Jittor text and VAE components for Stable Diffusion XL."""

from __future__ import annotations

import os
from typing import Tuple

import jittor as jt
from jittor import nn
import numpy as np

from masactrl.jittor_compat import enable_cuda
from masactrl.sd15_jittor import (
    CLIPSelfAttention,
    CLIPTextEmbeddings,
    SafeTensorStore,
    SD15VAE,
    _assign_layer_norm,
    _assign_linear,
    _gelu,
    _quick_gelu,
    assign_param,
)


def stable_norm(value: jt.Var, norm: nn.GroupNorm) -> jt.Var:
    """Group normalization without the unstable E[x²] - E[x]² variance."""

    shape = value.shape
    groups = norm.num_groups
    grouped = value.reshape(shape[0], groups, shape[1] // groups, -1)
    mean = grouped.mean(dims=[2, 3], keepdims=True)
    centered = grouped - mean
    variance = (centered * centered).mean(dims=[2, 3], keepdims=True)
    hidden = centered / jt.sqrt(variance + norm.eps)
    hidden = hidden.reshape(shape)
    weight = norm.weight.reshape(1, shape[1], 1, 1)
    bias = norm.bias.reshape(1, shape[1], 1, 1)
    return hidden * weight + bias


class ClipMlp(nn.Module):
    def __init__(self, width: int, inner: int, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(width, inner)
        self.fc2 = nn.Linear(inner, width)
        self.activation = activation

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_linear(self.fc1, store, prefix + ".fc1")
        _assign_linear(self.fc2, store, prefix + ".fc2")

    def execute(self, value: jt.Var) -> jt.Var:
        hidden = self.fc1(value)
        hidden = _quick_gelu(hidden) if self.activation == "quick_gelu" else _gelu(hidden)
        return self.fc2(hidden)


class ClipLayer(nn.Module):
    def __init__(self, width: int, heads: int, inner: int, activation: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = CLIPSelfAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = ClipMlp(width, inner, activation)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_layer_norm(self.norm1, store, prefix + ".layer_norm1")
        self.attn.load_from_store(store, prefix + ".self_attn")
        _assign_layer_norm(self.norm2, store, prefix + ".layer_norm2")
        self.mlp.load_from_store(store, prefix + ".mlp")

    def execute(self, value: jt.Var, mask: jt.Var) -> jt.Var:
        value = value + self.attn(self.norm1(value), mask)
        return value + self.mlp(self.norm2(value))


class ClipText(nn.Module):
    def __init__(
        self,
        width: int,
        inner: int,
        layers: int,
        heads: int,
        activation: str,
        projection: bool = False,
    ):
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(49408, 77, width)
        self.layers = [ClipLayer(width, heads, inner, activation) for _ in range(layers)]
        self.final_norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, width, bias=False) if projection else None

    def load_from_store(self, store: SafeTensorStore) -> None:
        self.embeddings.load_from_store(store, "text_model.embeddings")
        for index, layer in enumerate(self.layers):
            layer.load_from_store(store, f"text_model.encoder.layers.{index}")
        _assign_layer_norm(self.final_norm, store, "text_model.final_layer_norm")
        if self.projection is not None:
            assign_param(self.projection.weight, store.get("text_projection.weight"))

    @staticmethod
    def causal_mask(length: int) -> jt.Var:
        mask = np.triu(np.ones((length, length), dtype=np.float32), k=1) * -10000.0
        return jt.array(mask).reshape(1, length, length)

    @staticmethod
    def pool(value: jt.Var, input_ids: jt.Var) -> jt.Var:
        indices = input_ids.numpy().argmax(axis=-1).tolist()
        rows = [value[index:index + 1, token, :] for index, token in enumerate(indices)]
        return jt.concat(rows, dim=0)

    def execute(self, input_ids: jt.Var) -> Tuple[jt.Var, jt.Var, jt.Var]:
        value = self.embeddings(input_ids)
        mask = self.causal_mask(input_ids.shape[-1])
        penultimate = value
        for index, layer in enumerate(self.layers):
            value = layer(value, mask)
            if index == len(self.layers) - 2:
                penultimate = value
        final = self.final_norm(value)
        pooled = self.pool(final, input_ids)
        if self.projection is not None:
            pooled = self.projection(pooled)
        return penultimate, final, pooled


class SDXLParts(nn.Module):
    def __init__(self):
        super().__init__()
        enable_cuda()
        self.text1 = ClipText(768, 3072, 12, 12, "quick_gelu")
        self.text2 = ClipText(1280, 5120, 32, 20, "gelu", projection=True)
        self.vae = SD15VAE()
        self.scale = 0.13025
        self.root = ""

    @classmethod
    def from_pretrained(cls, root: str) -> "SDXLParts":
        model = cls()
        model.load_pretrained(root)
        return model

    def load_pretrained(self, root: str) -> None:
        paths = {
            "text1": os.path.join(root, "text_encoder", "model.fp16.safetensors"),
            "text2": os.path.join(root, "text_encoder_2", "model.fp16.safetensors"),
            "vae": os.path.join(root, "vae", "diffusion_pytorch_model.fp16.safetensors"),
        }
        missing = [path for path in paths.values() if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError("Missing SDXL stores: " + ", ".join(missing))
        self.text1.load_from_store(SafeTensorStore(paths["text1"]))
        self.text2.load_from_store(SafeTensorStore(paths["text2"]))
        self.vae.load_from_store(SafeTensorStore(paths["vae"]))
        self.root = root

    def encode_image(self, image: jt.Var) -> jt.Var:
        encoder = self.vae.encoder
        hidden = encoder.conv_in(image)
        for block in encoder.down_blocks:
            hidden = block(hidden)
        hidden = encoder.mid_block(hidden)
        hidden = encoder.conv_out(_silu(stable_norm(hidden, encoder.conv_norm_out)))
        moments = encoder.quant_conv(hidden)
        return moments[:, :4] * self.scale

    def decode_latent(self, latent: jt.Var) -> jt.Var:
        return self.vae.decode(latent / self.scale, split=latent.shape[-1] >= 128)
