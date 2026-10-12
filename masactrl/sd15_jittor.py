"""
Pure-Jittor Stable Diffusion v1.5 component helpers.

This module intentionally contains no legacy deep-learning backend imports. It provides the
weight-loading and executable component scaffolding needed to consume official
Stable Diffusion `.safetensors` checkpoints from Jittor.

The current classes cover the first verified SD1.5 submodules used by the
migration smoke tests:

- UNet input convolution
- VAE decoder output convolution
- CLIP token embedding table

They are small by design, but the loader and assignment code are the same path
the full Jittor UNet/VAE/CLIP implementation should use.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Dict, List, Mapping, Optional, Tuple

import jittor as jt
import numpy as np
from jittor import nn
from safetensors import safe_open

from .jittor_compat import enable_cuda


@dataclass(frozen=True)
class SD15ComponentPaths:
    root: str
    unet: str
    vae: str
    text_encoder: str

    @classmethod
    def from_pretrained_dir(cls, root: str) -> "SD15ComponentPaths":
        return cls(
            root=root,
            unet=_find_weight(root, "unet", "diffusion_pytorch_model"),
            vae=_find_weight(root, "vae", "diffusion_pytorch_model"),
            text_encoder=_find_weight(root, "text_encoder", "model"),
        )

    def validate(self) -> None:
        missing = [path for path in (self.unet, self.vae, self.text_encoder) if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError("Missing SD1.5 checkpoint files: " + ", ".join(missing))


def _find_weight(root: str, component: str, stem: str) -> str:
    folder = os.path.join(root, component)
    names = [f"{stem}.fp16.safetensors", f"{stem}.safetensors"]
    for name in names:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return os.path.join(folder, names[0])


class SafeTensorStore:
    """Small no-torch wrapper around safetensors numpy loading."""

    def __init__(self, path: str):
        self.path = path
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    def keys(self) -> List[str]:
        with safe_open(self.path, framework="numpy") as handle:
            return list(handle.keys())

    def get(self, key: str):
        with safe_open(self.path, framework="numpy") as handle:
            if key not in handle.keys():
                raise KeyError(f"{key} not found in {self.path}")
            return handle.get_tensor(key)


def assign_param(param: jt.Var, array, *, runtime_dtype: str = "float32") -> None:
    """Assign a numpy array from safetensors into a Jittor parameter.

    Official SD1.5 fp16 checkpoints are stored as half precision.  On the H100
    server used for this migration, Jittor/cuDNN is more reliable when runtime
    conv parameters are promoted to float32 while keeping the source checkpoint
    unchanged.
    """

    if runtime_dtype == "float32":
        array = array.astype("float32")
    tensor = jt.array(array)
    if tuple(param.shape) != tuple(tensor.shape):
        raise ValueError(f"shape mismatch: param={tuple(param.shape)}, tensor={tuple(tensor.shape)}")
    # Match Jittor's own ``Module.load_parameters`` implementation. ``update``
    # turns the value into persistent parameter storage; plain lazy ``assign``
    # can be discarded when the temporary checkpoint tensor leaves scope.
    param.update(tensor)
    param.sync(False, False)


def _zeros(shape, dtype="float32") -> jt.Var:
    """Create deterministic zeros without relying on the CUDA broadcast path."""

    return jt.array(np.zeros(tuple(shape), dtype=np.dtype(str(dtype))))


def _silu(x: jt.Var) -> jt.Var:
    return x * jt.sigmoid(x)


def _gelu(x: jt.Var) -> jt.Var:
    return 0.5 * x * (1.0 + jt.erf(x / np.sqrt(2.0)))


def _quick_gelu(x: jt.Var) -> jt.Var:
    return x * jt.sigmoid(1.702 * x)


def _assign_conv(module: nn.Conv, store: SafeTensorStore, prefix: str) -> None:
    assign_param(module.weight, store.get(prefix + ".weight"))
    assign_param(module.bias, store.get(prefix + ".bias"))


class Conv(nn.Conv):
    def execute(self, x: jt.Var) -> jt.Var:
        if not jt.flags.use_cuda:
            return super().execute(x)
        y = jt.cudnn.ops.cudnn_conv(
            x, self.weight, *self.stride, *self.padding, *self.dilation,
            self.groups, "abcd", "oihw", "abcd",
        )
        if self.bias is not None:
            y = y + self.bias.reshape(1, -1, 1, 1)
        return y


def _assign_linear(module: nn.Linear, store: SafeTensorStore, prefix: str) -> None:
    assign_param(module.weight, store.get(prefix + ".weight"))
    assign_param(module.bias, store.get(prefix + ".bias"))


def _assign_group_norm(module: nn.GroupNorm, store: SafeTensorStore, prefix: str) -> None:
    assign_param(module.weight, store.get(prefix + ".weight"))
    assign_param(module.bias, store.get(prefix + ".bias"))


def _assign_layer_norm(module: nn.LayerNorm, store: SafeTensorStore, prefix: str) -> None:
    assign_param(module.weight, store.get(prefix + ".weight"))
    assign_param(module.bias, store.get(prefix + ".bias"))


def _chunk_last(x: jt.Var, chunks: int):
    width = x.shape[-1]
    step = width // chunks
    return tuple(x[..., index * step:(index + 1) * step] for index in range(chunks))


def _chunk_channel(x: jt.Var, chunks: int):
    channels = x.shape[1]
    step = channels // chunks
    return tuple(x[:, index * step:(index + 1) * step] for index in range(chunks))


class ResnetBlock2D(nn.Module):
    """Diffusers-style VAE ResnetBlock2D without temporal embedding."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_group_norm(self.norm1, store, prefix + ".norm1")
        _assign_conv(self.conv1, store, prefix + ".conv1")
        _assign_group_norm(self.norm2, store, prefix + ".norm2")
        _assign_conv(self.conv2, store, prefix + ".conv2")
        if self.conv_shortcut is not None:
            _assign_conv(self.conv_shortcut, store, prefix + ".conv_shortcut")

    def execute(self, x: jt.Var) -> jt.Var:
        residual = x
        hidden = self.conv1(_silu(self.norm1(x)))
        hidden = self.conv2(_silu(self.norm2(hidden)))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return hidden + residual


class VAEAttentionBlock(nn.Module):
    """Single-head spatial self-attention used in SD1.5 VAE mid block."""

    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(groups, channels, eps=eps)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_group_norm(self.group_norm, store, prefix + ".group_norm")
        _assign_linear(self.to_q, store, prefix + ".to_q")
        _assign_linear(self.to_k, store, prefix + ".to_k")
        _assign_linear(self.to_v, store, prefix + ".to_v")
        _assign_linear(self.to_out, store, prefix + ".to_out.0")

    def execute(self, x: jt.Var) -> jt.Var:
        residual = x
        bsz, channels, height, width = x.shape
        hidden = self.group_norm(x)
        hidden = hidden.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)
        query = self.to_q(hidden)
        key = self.to_k(hidden)
        value = self.to_v(hidden)
        scale = channels ** -0.5
        scores = jt.matmul(query, key.permute(0, 2, 1)) * scale
        probs = scores.softmax(-1)
        hidden = jt.matmul(probs, value)
        hidden = self.to_out(hidden)
        hidden = hidden.reshape(bsz, height, width, channels).permute(0, 3, 1, 2)
        return hidden + residual


class VAEMidBlock2D(nn.Module):
    def __init__(self, channels: int = 512):
        super().__init__()
        self.resnet0 = ResnetBlock2D(channels, channels)
        self.attention0 = VAEAttentionBlock(channels)
        self.resnet1 = ResnetBlock2D(channels, channels)

    def load_from_store(self, store: SafeTensorStore, prefix: str = "decoder.mid_block") -> None:
        self.resnet0.load_from_store(store, prefix + ".resnets.0")
        self.attention0.load_from_store(store, prefix + ".attentions.0")
        self.resnet1.load_from_store(store, prefix + ".resnets.1")

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.resnet0(x)
        x = self.attention0(x)
        x = self.resnet1(x)
        return x


class VAEUpsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv(channels, channels, kernel_size=3, stride=1, padding=1)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_conv(self.conv, store, prefix + ".conv")

    def execute(self, x: jt.Var) -> jt.Var:
        height, width = x.shape[-2], x.shape[-1]
        x = nn.interpolate(x, size=(height * 2, width * 2), mode="nearest")
        return self.conv(x)


class VAEUpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_upsample: bool):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(in_channels, out_channels),
            ResnetBlock2D(out_channels, out_channels),
            ResnetBlock2D(out_channels, out_channels),
        ]
        self.upsampler = VAEUpsample2D(out_channels) if add_upsample else None

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        for index, resnet in enumerate(self.resnets):
            resnet.load_from_store(store, f"{prefix}.resnets.{index}")
        if self.upsampler is not None:
            self.upsampler.load_from_store(store, prefix + ".upsamplers.0")

    def execute(self, x: jt.Var) -> jt.Var:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsampler is not None:
            x = self.upsampler(x)
        return x


class SD15VAEDecoder(nn.Module):
    """Full SD1.5 VAE decoder, loadable from official safetensors."""

    def __init__(self):
        super().__init__()
        self.conv_in = Conv(4, 512, kernel_size=3, stride=1, padding=1)
        self.mid_block = VAEMidBlock2D(512)
        self.up_blocks = [
            VAEUpDecoderBlock2D(512, 512, add_upsample=True),
            VAEUpDecoderBlock2D(512, 512, add_upsample=True),
            VAEUpDecoderBlock2D(512, 256, add_upsample=True),
            VAEUpDecoderBlock2D(256, 128, add_upsample=False),
        ]
        self.conv_norm_out = nn.GroupNorm(32, 128, eps=1e-6)
        self.conv_out = Conv(128, 3, kernel_size=3, stride=1, padding=1)

    def load_from_store(self, store: SafeTensorStore) -> None:
        _assign_conv(self.conv_in, store, "decoder.conv_in")
        self.mid_block.load_from_store(store, "decoder.mid_block")
        for index, up_block in enumerate(self.up_blocks):
            up_block.load_from_store(store, f"decoder.up_blocks.{index}")
        _assign_group_norm(self.conv_norm_out, store, "decoder.conv_norm_out")
        _assign_conv(self.conv_out, store, "decoder.conv_out")

    def execute(self, z: jt.Var, split: bool = False) -> jt.Var:
        hidden = self.conv_in(z)
        hidden = self.mid_block(hidden)
        for up_block in self.up_blocks:
            hidden = up_block(hidden)
            if split:
                hidden = jt.array(hidden.numpy())
                jt.gc()
        hidden = self.conv_out(_silu(self.conv_norm_out(hidden)))
        return hidden


class VAEDownsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv(channels, channels, kernel_size=3, stride=2, padding=0)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_conv(self.conv, store, prefix + ".conv")

    def execute(self, x: jt.Var) -> jt.Var:
        batch, channels, height, width = x.shape
        x = jt.concat([x, _zeros((batch, channels, height, 1), x.dtype)], dim=3)
        x = jt.concat([x, _zeros((batch, channels, 1, width + 1), x.dtype)], dim=2)
        return self.conv(x)


class VAEDownBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_downsample: bool):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(in_channels, out_channels),
            ResnetBlock2D(out_channels, out_channels),
        ]
        self.downsampler = VAEDownsample2D(out_channels) if add_downsample else None

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        for index, resnet in enumerate(self.resnets):
            resnet.load_from_store(store, f"{prefix}.resnets.{index}")
        if self.downsampler is not None:
            self.downsampler.load_from_store(store, prefix + ".downsamplers.0")

    def execute(self, x: jt.Var) -> jt.Var:
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x


class SD15VAEEncoder(nn.Module):
    """SD1.5 VAE encoder through the posterior moments output."""

    def __init__(self):
        super().__init__()
        self.conv_in = Conv(3, 128, kernel_size=3, stride=1, padding=1)
        self.down_blocks = [
            VAEDownBlock2D(128, 128, add_downsample=True),
            VAEDownBlock2D(128, 256, add_downsample=True),
            VAEDownBlock2D(256, 512, add_downsample=True),
            VAEDownBlock2D(512, 512, add_downsample=False),
        ]
        self.mid_block = VAEMidBlock2D(512)
        self.conv_norm_out = nn.GroupNorm(32, 512, eps=1e-6)
        self.conv_out = Conv(512, 8, kernel_size=3, stride=1, padding=1)
        self.quant_conv = Conv(8, 8, kernel_size=1, stride=1, padding=0)

    def load_from_store(self, store: SafeTensorStore) -> None:
        _assign_conv(self.conv_in, store, "encoder.conv_in")
        for index, block in enumerate(self.down_blocks):
            block.load_from_store(store, f"encoder.down_blocks.{index}")
        self.mid_block.load_from_store(store, "encoder.mid_block")
        _assign_group_norm(self.conv_norm_out, store, "encoder.conv_norm_out")
        _assign_conv(self.conv_out, store, "encoder.conv_out")
        _assign_conv(self.quant_conv, store, "quant_conv")

    def execute(self, image: jt.Var) -> Tuple[jt.Var, jt.Var]:
        hidden = self.conv_in(image)
        for block in self.down_blocks:
            hidden = block(hidden)
        hidden = self.mid_block(hidden)
        moments = self.quant_conv(self.conv_out(_silu(self.conv_norm_out(hidden))))
        return _chunk_channel(moments, 2)


class SD15VAE(nn.Module):
    """Weight-compatible SD1.5 AutoencoderKL inference path."""

    def __init__(self):
        super().__init__()
        self.encoder = SD15VAEEncoder()
        self.post_quant_conv = Conv(4, 4, kernel_size=1, stride=1, padding=0)
        self.decoder = SD15VAEDecoder()

    def load_from_store(self, store: SafeTensorStore) -> None:
        self.encoder.load_from_store(store)
        _assign_conv(self.post_quant_conv, store, "post_quant_conv")
        self.decoder.load_from_store(store)

    def encode(self, image: jt.Var) -> jt.Var:
        mean, _ = self.encoder(image)
        return mean

    def decode(self, latent: jt.Var, split: bool = False) -> jt.Var:
        return self.decoder(self.post_quant_conv(latent), split=split)


class Timesteps(nn.Module):
    def __init__(self, channels: int = 320, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0.0):
        super().__init__()
        self.channels = channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def execute(self, timesteps: jt.Var) -> jt.Var:
        timesteps = timesteps.float32().reshape(-1)
        half_dim = self.channels // 2
        exponent = -np.log(10000.0) * jt.arange(half_dim).float32()
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = timesteps.unsqueeze(1) * jt.exp(exponent).unsqueeze(0)
        emb = jt.concat([jt.sin(emb), jt.cos(emb)], dim=-1)
        if self.flip_sin_to_cos:
            first, second = _chunk_last(emb, 2)
            emb = jt.concat([second, first], dim=-1)
        if self.channels % 2 == 1:
            emb = jt.concat([emb, _zeros((emb.shape[0], 1))], dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int = 320, time_embed_dim: int = 1280):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def load_from_store(self, store: SafeTensorStore, prefix: str = "time_embedding") -> None:
        _assign_linear(self.linear_1, store, prefix + ".linear_1")
        _assign_linear(self.linear_2, store, prefix + ".linear_2")

    def execute(self, sample: jt.Var) -> jt.Var:
        return self.linear_2(_silu(self.linear_1(sample)))


class UNetResnetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int = 1280, groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_group_norm(self.norm1, store, prefix + ".norm1")
        _assign_conv(self.conv1, store, prefix + ".conv1")
        _assign_linear(self.time_emb_proj, store, prefix + ".time_emb_proj")
        _assign_group_norm(self.norm2, store, prefix + ".norm2")
        _assign_conv(self.conv2, store, prefix + ".conv2")
        if self.conv_shortcut is not None:
            _assign_conv(self.conv_shortcut, store, prefix + ".conv_shortcut")

    def execute(self, x: jt.Var, temb: jt.Var) -> jt.Var:
        residual = x
        hidden = self.conv1(_silu(self.norm1(x)))
        time_hidden = self.time_emb_proj(_silu(temb)).unsqueeze(-1).unsqueeze(-1)
        hidden = hidden + time_hidden
        hidden = self.conv2(_silu(self.norm2(hidden)))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return hidden + residual


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: Optional[int] = None, heads: int = 8, dim_head: int = 40):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim if context_dim is not None else query_dim
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(self.context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(self.context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.editor = None
        self.place = ""

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        assign_param(self.to_q.weight, store.get(prefix + ".to_q.weight"))
        assign_param(self.to_k.weight, store.get(prefix + ".to_k.weight"))
        assign_param(self.to_v.weight, store.get(prefix + ".to_v.weight"))
        _assign_linear(self.to_out, store, prefix + ".to_out.0")

    def execute(self, hidden_states: jt.Var, encoder_hidden_states: Optional[jt.Var] = None) -> jt.Var:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        bsz, tokens, _ = hidden_states.shape
        query = self.to_q(hidden_states)
        key = self.to_k(context)
        value = self.to_v(context)

        def split_heads(x):
            return x.reshape(bsz, -1, self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(bsz * self.heads, -1, self.dim_head)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        scores = jt.matmul(query, key.permute(0, 2, 1)) * self.scale
        probs = scores.softmax(-1)
        if self.editor is None:
            hidden = jt.matmul(probs, value)
            hidden = hidden.reshape(bsz, self.heads, tokens, self.dim_head).permute(0, 2, 1, 3).reshape(
                bsz, tokens, self.heads * self.dim_head
            )
        else:
            hidden = self.editor(
                query,
                key,
                value,
                scores,
                probs,
                encoder_hidden_states is not None,
                self.place,
                self.heads,
                scale=self.scale,
            )
        return self.to_out(hidden)


class FeedForwardGEGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inner_dim = dim * 4
        self.proj_in = nn.Linear(dim, inner_dim * 2)
        self.proj_out = nn.Linear(inner_dim, dim)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_linear(self.proj_in, store, prefix + ".net.0.proj")
        _assign_linear(self.proj_out, store, prefix + ".net.2")

    def execute(self, hidden_states: jt.Var) -> jt.Var:
        hidden, gate = _chunk_last(self.proj_in(hidden_states), 2)
        return self.proj_out(hidden * _gelu(gate))


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, context_dim: int = 768, heads: int = 8, dim_head: int = 40):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, None, heads=heads, dim_head=dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, context_dim, heads=heads, dim_head=dim_head)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForwardGEGLU(dim)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_layer_norm(self.norm1, store, prefix + ".norm1")
        self.attn1.load_from_store(store, prefix + ".attn1")
        _assign_layer_norm(self.norm2, store, prefix + ".norm2")
        self.attn2.load_from_store(store, prefix + ".attn2")
        _assign_layer_norm(self.norm3, store, prefix + ".norm3")
        self.ff.load_from_store(store, prefix + ".ff")

    def execute(self, hidden_states: jt.Var, encoder_hidden_states: jt.Var) -> jt.Var:
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), encoder_hidden_states) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class Transformer2DModel(nn.Module):
    def __init__(self, channels: int, context_dim: int = 768, attention_head_dim: int = 8):
        super().__init__()
        self.channels = channels
        self.heads = attention_head_dim
        self.dim_head = channels // attention_head_dim
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.proj_in = Conv(channels, channels, kernel_size=1, stride=1, padding=0)
        self.transformer_block = BasicTransformerBlock(channels, context_dim, heads=self.heads, dim_head=self.dim_head)
        self.proj_out = Conv(channels, channels, kernel_size=1, stride=1, padding=0)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_group_norm(self.norm, store, prefix + ".norm")
        _assign_conv(self.proj_in, store, prefix + ".proj_in")
        self.transformer_block.load_from_store(store, prefix + ".transformer_blocks.0")
        _assign_conv(self.proj_out, store, prefix + ".proj_out")

    def execute(self, x: jt.Var, encoder_hidden_states: jt.Var) -> jt.Var:
        residual = x
        bsz, channels, height, width = x.shape
        hidden = self.proj_in(self.norm(x))
        hidden = hidden.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)
        hidden = self.transformer_block(hidden, encoder_hidden_states)
        hidden = hidden.reshape(bsz, height, width, channels).permute(0, 3, 1, 2)
        return self.proj_out(hidden) + residual


class UNetDownsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv(channels, channels, kernel_size=3, stride=2, padding=1)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_conv(self.conv, store, prefix + ".conv")

    def execute(self, x: jt.Var) -> jt.Var:
        return self.conv(x)


class UNetCrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, add_downsample: bool = True):
        super().__init__()
        self.resnets = [
            UNetResnetBlock2D(in_channels, out_channels),
            UNetResnetBlock2D(out_channels, out_channels),
        ]
        self.attentions = [
            Transformer2DModel(out_channels),
            Transformer2DModel(out_channels),
        ]
        self.downsampler = UNetDownsample2D(out_channels) if add_downsample else None

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        for index in range(2):
            self.resnets[index].load_from_store(store, f"{prefix}.resnets.{index}")
            self.attentions[index].load_from_store(store, f"{prefix}.attentions.{index}")
        if self.downsampler is not None:
            self.downsampler.load_from_store(store, prefix + ".downsamplers.0")

    def execute(self, x: jt.Var, temb: jt.Var, encoder_hidden_states: jt.Var, *, return_residuals: bool = False):
        residuals = []
        for resnet, attention in zip(self.resnets, self.attentions):
            x = resnet(x, temb)
            x = attention(x, encoder_hidden_states)
            residuals.append(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
            residuals.append(x)
        if return_residuals:
            return x, residuals
        return x


class UNetDownBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, add_downsample: bool = False):
        super().__init__()
        self.resnets = [
            UNetResnetBlock2D(in_channels, out_channels),
            UNetResnetBlock2D(out_channels, out_channels),
        ]
        self.downsampler = UNetDownsample2D(out_channels) if add_downsample else None

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        for index, resnet in enumerate(self.resnets):
            resnet.load_from_store(store, f"{prefix}.resnets.{index}")
        if self.downsampler is not None:
            self.downsampler.load_from_store(store, prefix + ".downsamplers.0")

    def execute(self, x: jt.Var, temb: jt.Var, *, return_residuals: bool = False):
        residuals = []
        for resnet in self.resnets:
            x = resnet(x, temb)
            residuals.append(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
            residuals.append(x)
        if return_residuals:
            return x, residuals
        return x


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self, channels: int = 1280):
        super().__init__()
        self.resnet0 = UNetResnetBlock2D(channels, channels)
        self.attention0 = Transformer2DModel(channels)
        self.resnet1 = UNetResnetBlock2D(channels, channels)

    def load_from_store(self, store: SafeTensorStore, prefix: str = "mid_block") -> None:
        self.resnet0.load_from_store(store, prefix + ".resnets.0")
        self.attention0.load_from_store(store, prefix + ".attentions.0")
        self.resnet1.load_from_store(store, prefix + ".resnets.1")

    def execute(self, x: jt.Var, temb: jt.Var, encoder_hidden_states: jt.Var) -> jt.Var:
        x = self.resnet0(x, temb)
        x = self.attention0(x, encoder_hidden_states)
        x = self.resnet1(x, temb)
        return x


class UNetUpsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv(channels, channels, kernel_size=3, stride=1, padding=1)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_conv(self.conv, store, prefix + ".conv")

    def execute(self, x: jt.Var) -> jt.Var:
        height, width = x.shape[-2], x.shape[-1]
        x = nn.interpolate(x, size=(height * 2, width * 2), mode="nearest")
        return self.conv(x)


def _match_spatial(residual: jt.Var, target: jt.Var) -> jt.Var:
    if residual.shape[-2] == target.shape[-2] and residual.shape[-1] == target.shape[-1]:
        return residual
    raise ValueError(
        "UNet skip shape mismatch: "
        f"residual={tuple(residual.shape)}, target={tuple(target.shape)}"
    )


class UNetUpBlock2D(nn.Module):
    def __init__(self, resnet_in_channels: List[int], out_channels: int, *, add_upsample: bool):
        super().__init__()
        self.resnets = [UNetResnetBlock2D(in_channels, out_channels) for in_channels in resnet_in_channels]
        self.upsampler = UNetUpsample2D(out_channels) if add_upsample else None

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        for index, resnet in enumerate(self.resnets):
            resnet.load_from_store(store, f"{prefix}.resnets.{index}")
        if self.upsampler is not None:
            self.upsampler.load_from_store(store, prefix + ".upsamplers.0")

    def execute(self, x: jt.Var, temb: jt.Var, residuals: List[jt.Var]) -> jt.Var:
        for resnet in self.resnets:
            residual = _match_spatial(residuals.pop(), x)
            x = jt.concat([x, residual], dim=1)
            x = resnet(x, temb)
        if self.upsampler is not None:
            x = self.upsampler(x)
        return x


class UNetCrossAttnUpBlock2D(nn.Module):
    def __init__(self, resnet_in_channels: List[int], out_channels: int, *, add_upsample: bool):
        super().__init__()
        self.resnets = [UNetResnetBlock2D(in_channels, out_channels) for in_channels in resnet_in_channels]
        self.attentions = [Transformer2DModel(out_channels) for _ in resnet_in_channels]
        self.upsampler = UNetUpsample2D(out_channels) if add_upsample else None

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        for index, resnet in enumerate(self.resnets):
            resnet.load_from_store(store, f"{prefix}.resnets.{index}")
            self.attentions[index].load_from_store(store, f"{prefix}.attentions.{index}")
        if self.upsampler is not None:
            self.upsampler.load_from_store(store, prefix + ".upsamplers.0")

    def execute(self, x: jt.Var, temb: jt.Var, encoder_hidden_states: jt.Var, residuals: List[jt.Var]) -> jt.Var:
        for resnet, attention in zip(self.resnets, self.attentions):
            residual = _match_spatial(residuals.pop(), x)
            x = jt.concat([x, residual], dim=1)
            x = resnet(x, temb)
            x = attention(x, encoder_hidden_states)
        if self.upsampler is not None:
            x = self.upsampler(x)
        return x


class UNetOutputHead(nn.Module):
    def __init__(self, in_channels: int = 320, out_channels: int = 4):
        super().__init__()
        self.conv_norm_out = nn.GroupNorm(32, in_channels, eps=1e-5)
        self.conv_out = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def load_from_store(self, store: SafeTensorStore) -> None:
        _assign_group_norm(self.conv_norm_out, store, "conv_norm_out")
        _assign_conv(self.conv_out, store, "conv_out")

    def execute(self, x: jt.Var) -> jt.Var:
        return self.conv_out(_silu(self.conv_norm_out(x)))


class SD15Scheduler:
    """Pure Jittor/NumPy scheduler state loaded from SD1.5 scheduler config.

    The SD1.5 config downloaded for this project identifies a PNDM scheduler.
    This class implements the deterministic alpha/beta state and a DDIM-form
    update step for migration smoke tests. It verifies no-torch latent scheduler
    math; full PNDM parity will be added with the complete UNet sampler.
    """

    def __init__(
        self,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        num_train_timesteps: int = 1000,
        steps_offset: int = 1,
        set_alpha_to_one: bool = False,
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.num_train_timesteps = num_train_timesteps
        self.steps_offset = steps_offset
        self.set_alpha_to_one = set_alpha_to_one
        if beta_schedule == "scaled_linear":
            betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32) ** 2
        elif beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        else:
            raise NotImplementedError(f"Unsupported beta_schedule: {beta_schedule}")
        alphas = 1.0 - betas
        self.betas_np = betas.astype("float32")
        self.alphas_cumprod_np = np.cumprod(alphas).astype("float32")
        final_alpha = 1.0 if set_alpha_to_one else self.alphas_cumprod_np[0]
        self.final_alpha_cumprod = np.array(final_alpha, dtype=np.float32)
        self.timesteps_np = np.array([], dtype=np.int32)

    @classmethod
    def from_pretrained(cls, root: str) -> "SD15Scheduler":
        config = load_component_config(root, "scheduler")
        return cls(
            beta_start=float(config.get("beta_start", 0.00085)),
            beta_end=float(config.get("beta_end", 0.012)),
            beta_schedule=str(config.get("beta_schedule", "scaled_linear")),
            num_train_timesteps=int(config.get("num_train_timesteps", 1000)),
            steps_offset=int(config.get("steps_offset", 1)),
            set_alpha_to_one=bool(config.get("set_alpha_to_one", False)),
        )

    def set_timesteps(self, num_inference_steps: int) -> jt.Var:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int32)
        if self.steps_offset:
            timesteps = np.clip(timesteps + self.steps_offset, 0, self.num_train_timesteps - 1)
        self.timesteps_np = timesteps
        return jt.array(timesteps)

    def scale_model_input(self, sample: jt.Var, timestep: int) -> jt.Var:
        return sample

    def step(self, model_output: jt.Var, timestep: int, sample: jt.Var) -> jt.Var:
        if self.timesteps_np.size == 0:
            raise RuntimeError("call set_timesteps before step")
        matches = np.where(self.timesteps_np == int(timestep))[0]
        index = int(matches[0]) if len(matches) else 0
        prev_timestep = int(self.timesteps_np[index + 1]) if index + 1 < len(self.timesteps_np) else -1

        alpha_prod_t = float(self.alphas_cumprod_np[int(timestep)])
        alpha_prod_prev = float(self.alphas_cumprod_np[prev_timestep]) if prev_timestep >= 0 else float(self.final_alpha_cumprod)
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_prev = 1.0 - alpha_prod_prev

        pred_original_sample = (sample - (beta_prod_t**0.5) * model_output) / (alpha_prod_t**0.5)
        pred_sample_direction = (beta_prod_prev**0.5) * model_output
        return (alpha_prod_prev**0.5) * pred_original_sample + pred_sample_direction

    def next_step(self, model_output: jt.Var, timestep: int, sample: jt.Var) -> jt.Var:
        if self.timesteps_np.size == 0:
            raise RuntimeError("call set_timesteps before next_step")
        step_ratio = self.num_train_timesteps // len(self.timesteps_np)
        next_timestep = int(timestep)
        current_timestep = min(next_timestep - step_ratio, self.num_train_timesteps - 1)
        alpha_current = (
            float(self.alphas_cumprod_np[current_timestep])
            if current_timestep >= 0
            else float(self.final_alpha_cumprod)
        )
        alpha_next = float(self.alphas_cumprod_np[next_timestep])
        beta_current = 1.0 - alpha_current
        pred_original = (sample - (beta_current**0.5) * model_output) / (alpha_current**0.5)
        direction = ((1.0 - alpha_next) ** 0.5) * model_output
        return (alpha_next**0.5) * pred_original + direction


class UNetInputStem(nn.Module):
    """The SD1.5 UNet input convolution, weight-compatible with official checkpoints."""

    weight_map: Mapping[str, str] = {
        "conv_in.weight": "conv_in.weight",
        "conv_in.bias": "conv_in.bias",
    }

    def __init__(self, in_channels: int = 4, out_channels: int = 320):
        super().__init__()
        self.conv_in = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def load_from_store(self, store: SafeTensorStore) -> None:
        assign_param(self.conv_in.weight, store.get("conv_in.weight"))
        assign_param(self.conv_in.bias, store.get("conv_in.bias"))

    def execute(self, latents: jt.Var) -> jt.Var:
        return self.conv_in(latents)


class VAEDecoderOutputHead(nn.Module):
    """The SD1.5 VAE decoder output convolution."""

    weight_map: Mapping[str, str] = {
        "decoder.conv_out.weight": "decoder.conv_out.weight",
        "decoder.conv_out.bias": "decoder.conv_out.bias",
    }

    def __init__(self, in_channels: int = 128, out_channels: int = 3):
        super().__init__()
        self.conv_out = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def load_from_store(self, store: SafeTensorStore) -> None:
        assign_param(self.conv_out.weight, store.get("decoder.conv_out.weight"))
        assign_param(self.conv_out.bias, store.get("decoder.conv_out.bias"))

    def execute(self, features: jt.Var) -> jt.Var:
        return self.conv_out(features)


class CLIPTokenEmbedding(nn.Module):
    """CLIP token embedding table from SD1.5 text_encoder weights."""

    weight_key = "text_model.embeddings.token_embedding.weight"

    def __init__(self, vocab_size: int = 49408, hidden_size: int = 768):
        super().__init__()
        self.weight = _zeros((vocab_size, hidden_size))

    def load_from_store(self, store: SafeTensorStore) -> None:
        assign_param(self.weight, store.get(self.weight_key))

    def execute(self, input_ids: jt.Var) -> jt.Var:
        # Jittor advanced indexing covers the embedding lookup and avoids a
        # dependency on a specific nn.Embedding attribute layout.
        return self.weight[input_ids]


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, vocab_size: int = 49408, max_position_embeddings: int = 77, hidden_size: int = 768):
        super().__init__()
        self.token_embedding = CLIPTokenEmbedding(vocab_size, hidden_size)
        self.position_embedding = _zeros((max_position_embeddings, hidden_size))

    def load_from_store(self, store: SafeTensorStore, prefix: str = "text_model.embeddings") -> None:
        self.token_embedding.load_from_store(store)
        assign_param(self.position_embedding, store.get(prefix + ".position_embedding.weight"))

    def execute(self, input_ids: jt.Var) -> jt.Var:
        seq_len = input_ids.shape[-1]
        position_ids = jt.arange(seq_len).int32().unsqueeze(0)
        return self.token_embedding(input_ids) + self.position_embedding[position_ids]


class CLIPSelfAttention(nn.Module):
    def __init__(self, hidden_size: int = 768, num_heads: int = 12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_linear(self.q_proj, store, prefix + ".q_proj")
        _assign_linear(self.k_proj, store, prefix + ".k_proj")
        _assign_linear(self.v_proj, store, prefix + ".v_proj")
        _assign_linear(self.out_proj, store, prefix + ".out_proj")

    def _split_heads(self, x: jt.Var) -> jt.Var:
        bsz, seq_len, _ = x.shape
        return x.reshape(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(
            bsz * self.num_heads, seq_len, self.head_dim
        )

    def execute(self, hidden_states: jt.Var, causal_mask: jt.Var) -> jt.Var:
        bsz, seq_len, _ = hidden_states.shape
        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))
        scores = jt.matmul(query, key.permute(0, 2, 1)) * self.scale
        scores = scores + causal_mask
        probs = scores.softmax(-1)
        hidden = jt.matmul(probs, value)
        hidden = hidden.reshape(bsz, self.num_heads, seq_len, self.head_dim).permute(0, 2, 1, 3).reshape(
            bsz, seq_len, self.hidden_size
        )
        return self.out_proj(hidden)


class CLIPMLP(nn.Module):
    def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_linear(self.fc1, store, prefix + ".fc1")
        _assign_linear(self.fc2, store, prefix + ".fc2")

    def execute(self, hidden_states: jt.Var) -> jt.Var:
        return self.fc2(_quick_gelu(self.fc1(hidden_states)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int = 768, num_heads: int = 12, intermediate_size: int = 3072):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = CLIPSelfAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.mlp = CLIPMLP(hidden_size, intermediate_size)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_layer_norm(self.layer_norm1, store, prefix + ".layer_norm1")
        self.self_attn.load_from_store(store, prefix + ".self_attn")
        _assign_layer_norm(self.layer_norm2, store, prefix + ".layer_norm2")
        self.mlp.load_from_store(store, prefix + ".mlp")

    def execute(self, hidden_states: jt.Var, causal_mask: jt.Var) -> jt.Var:
        hidden_states = self.self_attn(self.layer_norm1(hidden_states), causal_mask) + hidden_states
        hidden_states = self.mlp(self.layer_norm2(hidden_states)) + hidden_states
        return hidden_states


class CLIPTextModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 49408,
        max_position_embeddings: int = 77,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.embeddings = CLIPTextEmbeddings(vocab_size, max_position_embeddings, hidden_size)
        self.layers = [
            CLIPEncoderLayer(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_hidden_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(hidden_size)

    def load_from_store(self, store: SafeTensorStore) -> None:
        self.embeddings.load_from_store(store, "text_model.embeddings")
        for index, layer in enumerate(self.layers):
            layer.load_from_store(store, f"text_model.encoder.layers.{index}")
        _assign_layer_norm(self.final_layer_norm, store, "text_model.final_layer_norm")

    def _causal_mask(self, seq_len: int) -> jt.Var:
        mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1) * -10000.0
        return jt.array(mask).reshape(1, seq_len, seq_len)

    def execute(self, input_ids: jt.Var) -> jt.Var:
        hidden_states = self.embeddings(input_ids)
        causal_mask = self._causal_mask(input_ids.shape[-1])
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask)
        return self.final_layer_norm(hidden_states)


class StableDiffusion15Jittor(nn.Module):
    """Executable no-torch SD1.5 component bundle used by migration tests."""

    def __init__(self):
        super().__init__()
        enable_cuda()
        self.unet_input = UNetInputStem()
        self.time_proj = Timesteps(320, flip_sin_to_cos=True, downscale_freq_shift=0.0)
        self.time_embedding = TimestepEmbedding(320, 1280)
        self.unet_down_blocks = [
            UNetCrossAttnDownBlock2D(320, 320, add_downsample=True),
            UNetCrossAttnDownBlock2D(320, 640, add_downsample=True),
            UNetCrossAttnDownBlock2D(640, 1280, add_downsample=True),
            UNetDownBlock2D(1280, 1280, add_downsample=False),
        ]
        self.unet_mid_block = UNetMidBlock2DCrossAttn(1280)
        self.unet_up_blocks = [
            UNetUpBlock2D([2560, 2560, 2560], 1280, add_upsample=True),
            UNetCrossAttnUpBlock2D([2560, 2560, 1920], 1280, add_upsample=True),
            UNetCrossAttnUpBlock2D([1920, 1280, 960], 640, add_upsample=True),
            UNetCrossAttnUpBlock2D([960, 640, 640], 320, add_upsample=False),
        ]
        self.unet_output = UNetOutputHead(320, 4)
        self.vae = SD15VAE()
        self.vae_decoder_output = VAEDecoderOutputHead()
        self.vae_decoder = self.vae.decoder
        self.text_token_embedding = CLIPTokenEmbedding()
        self.text_encoder = CLIPTextModel()
        self.scheduler = None
        self.root = ""

    @classmethod
    def from_pretrained(cls, root: str) -> "StableDiffusion15Jittor":
        model = cls()
        model.load_pretrained(root)
        return model

    def load_pretrained(self, root: str) -> None:
        paths = SD15ComponentPaths.from_pretrained_dir(root)
        paths.validate()
        unet_store = SafeTensorStore(paths.unet)
        self.unet_input.load_from_store(unet_store)
        self.time_embedding.load_from_store(unet_store)
        for index, down_block in enumerate(self.unet_down_blocks):
            down_block.load_from_store(unet_store, f"down_blocks.{index}")
        self.unet_mid_block.load_from_store(unet_store, "mid_block")
        for index, up_block in enumerate(self.unet_up_blocks):
            up_block.load_from_store(unet_store, f"up_blocks.{index}")
        self.unet_output.load_from_store(unet_store)
        vae_store = SafeTensorStore(paths.vae)
        self.vae_decoder_output.load_from_store(vae_store)
        self.vae.load_from_store(vae_store)
        text_store = SafeTensorStore(paths.text_encoder)
        self.text_token_embedding.load_from_store(text_store)
        self.text_encoder.load_from_store(text_store)
        self.scheduler = SD15Scheduler.from_pretrained(root)
        self.root = root

    def predict_noise(
        self,
        latents: jt.Var,
        timestep: int,
        encoder_hidden_states: jt.Var,
        trace: Optional[Dict[str, jt.Var]] = None,
        down_residuals: Optional[List[jt.Var]] = None,
        mid_residual: Optional[jt.Var] = None,
    ) -> jt.Var:
        timestep_tensor = jt.array([int(timestep)]).int32()
        if latents.shape[0] > 1:
            timestep_tensor = timestep_tensor.repeat(latents.shape[0])
        temb = self.time_embedding(self.time_proj(timestep_tensor))
        hidden = self.unet_input(latents)
        if trace is not None:
            trace["time"] = temb
            trace["input"] = hidden
        residuals = [hidden]
        for index, block in enumerate(self.unet_down_blocks):
            if isinstance(block, UNetCrossAttnDownBlock2D):
                hidden, block_residuals = block(
                    hidden, temb, encoder_hidden_states, return_residuals=True
                )
            else:
                hidden, block_residuals = block(hidden, temb, return_residuals=True)
            residuals.extend(block_residuals)
            if trace is not None:
                trace[f"down{index}"] = hidden
        if down_residuals is not None:
            if len(down_residuals) != len(residuals):
                raise ValueError(f"ControlNet residual count mismatch: {len(down_residuals)} != {len(residuals)}")
            residuals = [left + right for left, right in zip(residuals, down_residuals)]
        hidden = self.unet_mid_block(hidden, temb, encoder_hidden_states)
        if mid_residual is not None:
            hidden = hidden + mid_residual
        if trace is not None:
            trace["mid"] = hidden
        for index, block in enumerate(self.unet_up_blocks):
            if isinstance(block, UNetCrossAttnUpBlock2D):
                hidden = block(hidden, temb, encoder_hidden_states, residuals)
            else:
                hidden = block(hidden, temb, residuals)
            if trace is not None:
                trace[f"up{index}"] = hidden
        if residuals:
            raise RuntimeError(f"UNet residual stack is not empty: {len(residuals)}")
        output = self.unet_output(hidden)
        if trace is not None:
            trace["output"] = output
        return output

    def execute(self, latents: jt.Var, timestep: int, encoder_hidden_states: jt.Var) -> jt.Var:
        return self.predict_noise(latents, timestep, encoder_hidden_states)

    def encode_image(self, image: jt.Var) -> jt.Var:
        return self.vae.encode(image) * 0.18215

    def decode_latent(self, latents: jt.Var) -> jt.Var:
        return self.vae.decode(latents / 0.18215)

    def attention_modules(self):
        modules = []

        def add_transformer(transformer, place):
            block = transformer.transformer_block
            block.attn1.place = place
            block.attn2.place = place
            modules.extend([block.attn1, block.attn2])

        for block in self.unet_down_blocks:
            for transformer in getattr(block, "attentions", []):
                add_transformer(transformer, "down")
        add_transformer(self.unet_mid_block.attention0, "mid")
        for block in self.unet_up_blocks:
            for transformer in getattr(block, "attentions", []):
                add_transformer(transformer, "up")
        return modules

    def smoke_forward(self, *, include_full_vae_decoder: bool = True) -> Dict[str, object]:
        latents = jt.randn((1, 4, 64, 64))
        unet_features = self.unet_input(latents)
        unet_features.sync()
        timestep_embedding = self.time_embedding(self.time_proj(jt.array([999]).int32()))
        timestep_embedding.sync()
        small_latent_for_unet = jt.randn((1, 4, 8, 8))
        small_unet_features = self.unet_input(small_latent_for_unet)
        encoder_hidden_states = _zeros((1, 1, 768))
        down_residuals = [small_unet_features]
        down_shapes = []
        down_means = []
        down_hidden = small_unet_features
        for down_block in self.unet_down_blocks:
            if isinstance(down_block, UNetCrossAttnDownBlock2D):
                down_hidden, block_residuals = down_block(
                    down_hidden, timestep_embedding, encoder_hidden_states, return_residuals=True
                )
            else:
                down_hidden, block_residuals = down_block(down_hidden, timestep_embedding, return_residuals=True)
            down_residuals.extend(block_residuals)
            down_hidden.sync()
            down_shapes.append(list(down_hidden.shape))
            down_means.append(float(down_hidden.mean().numpy()))
        mid_hidden = self.unet_mid_block(down_hidden, timestep_embedding, encoder_hidden_states)
        mid_hidden.sync()
        up_residuals = list(down_residuals)
        up_shapes = []
        up_means = []
        up_hidden = mid_hidden
        for up_block in self.unet_up_blocks:
            if isinstance(up_block, UNetCrossAttnUpBlock2D):
                up_hidden = up_block(up_hidden, timestep_embedding, encoder_hidden_states, up_residuals)
            else:
                up_hidden = up_block(up_hidden, timestep_embedding, up_residuals)
            up_hidden.sync()
            up_shapes.append(list(up_hidden.shape))
            up_means.append(float(up_hidden.mean().numpy()))
        unet_noise_pred = self.unet_output(up_hidden)
        unet_noise_pred.sync()

        decoder_features = jt.randn((1, 128, 64, 64))
        decoded = self.vae_decoder_output(decoder_features)
        decoded.sync()

        input_ids = jt.array([[0, 1, 2, 3]]).int32()
        token_embeddings = self.text_token_embedding(input_ids)
        token_embeddings.sync()
        text_input_ids = jt.array([list(range(77))]).int32()
        text_hidden_states = self.text_encoder(text_input_ids)
        text_hidden_states.sync()

        stats = {
            "unet.conv_in.out_shape": list(unet_features.shape),
            "unet.time_embedding.out_shape": list(timestep_embedding.shape),
            "unet.down_path.out_shapes": down_shapes,
            "unet.mid_block.out_shape": list(mid_hidden.shape),
            "unet.up_path.out_shapes": up_shapes,
            "unet.output.out_shape": list(unet_noise_pred.shape),
            "vae.decoder.conv_out.out_shape": list(decoded.shape),
            "text.token_embedding.out_shape": list(token_embeddings.shape),
            "text.token_embedding.table_shape": list(self.text_token_embedding.weight.shape),
            "text.encoder.last_hidden_state_shape": list(text_hidden_states.shape),
            "text.token_embedding.slice_mean": float(token_embeddings.mean().numpy()),
            "text.encoder.last_hidden_state_mean": float(text_hidden_states.mean().numpy()),
            "unet.conv_in.out_mean": float(unet_features.mean().numpy()),
            "unet.time_embedding.out_mean": float(timestep_embedding.mean().numpy()),
            "unet.down_path.out_means": down_means,
            "unet.mid_block.out_mean": float(mid_hidden.mean().numpy()),
            "unet.up_path.out_means": up_means,
            "unet.output.out_mean": float(unet_noise_pred.mean().numpy()),
            "vae.decoder.conv_out.out_mean": float(decoded.mean().numpy()),
        }
        if include_full_vae_decoder:
            small_latent = jt.randn((1, 4, 8, 8))
            small_image = self.vae.decode(small_latent)
            small_image.sync()
            stats["vae.decoder.full.out_shape"] = list(small_image.shape)
            stats["vae.decoder.full.out_mean"] = float(small_image.mean().numpy())
            if self.scheduler is not None:
                timesteps = self.scheduler.set_timesteps(4)
                timestep = int(timesteps.numpy()[0])
                noise_pred = small_latent * 0.0
                prev_latent = self.scheduler.step(noise_pred, timestep, small_latent)
                prev_latent.sync()
                stats["scheduler.timesteps"] = self.scheduler.timesteps_np.tolist()
                stats["scheduler.prev_latent_shape"] = list(prev_latent.shape)
                stats["scheduler.prev_latent_mean"] = float(prev_latent.mean().numpy())
        return stats

    def component_inference_smoke(self, *, latent_size: int = 8, num_inference_steps: int = 4) -> jt.Var:
        if self.scheduler is None:
            raise RuntimeError("load_pretrained must be called before component_inference_smoke")
        latents = jt.randn((1, 4, latent_size, latent_size))
        timesteps = self.scheduler.set_timesteps(num_inference_steps)
        for raw_timestep in timesteps.numpy().tolist():
            timestep = int(raw_timestep)
            stem_features = self.unet_input(self.scheduler.scale_model_input(latents, timestep))
            stem_features.sync()
            text = _zeros((1, 77, 768))
            noise_pred = self.predict_noise(latents, timestep, text)
            latents = self.scheduler.step(noise_pred, timestep, latents)
        decoded = self.decode_latent(latents)
        image = jt.clamp(decoded / 2 + 0.5, 0, 1)
        image.sync()
        return image


def load_component_config(root: str, component: str) -> Dict[str, object]:
    config_name = "scheduler_config.json" if component == "scheduler" else "config.json"
    config_path = os.path.join(root, component, config_name)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def list_checkpoint_keys(path: str, *, prefix: Optional[str] = None) -> List[str]:
    keys = SafeTensorStore(path).keys()
    if prefix is not None:
        keys = [key for key in keys if key.startswith(prefix)]
    return keys


def assert_legacy_backend_not_loaded() -> None:
    import sys

    loaded = [name for name in sys.modules if name == "torch" or name.startswith("torch.")]
    if loaded:
        raise RuntimeError("legacy backend modules were loaded unexpectedly: " + ", ".join(sorted(loaded)[:8]))
