"""Native Jittor Zero123 inference model.

The UNet and VAE share the verified SD1.5 implementation.  Zero123 changes the
UNet input to eight channels and replaces text conditioning with a CLIP image
embedding plus four camera parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Dict, Optional

import jittor as jt
import numpy as np
from jittor import nn

from masactrl.jittor_compat import enable_cuda
from masactrl.sd15_jittor import (
    SafeTensorStore,
    SD15Scheduler,
    SD15VAE,
    StableDiffusion15Jittor,
    Timesteps,
    TimestepEmbedding,
    UNetCrossAttnDownBlock2D,
    UNetCrossAttnUpBlock2D,
    UNetDownBlock2D,
    UNetInputStem,
    UNetMidBlock2DCrossAttn,
    UNetOutputHead,
    UNetUpBlock2D,
    _assign_conv,
    _assign_layer_norm,
    _assign_linear,
    _quick_gelu,
    assign_param,
)


@dataclass(frozen=True)
class Zero123Paths:
    root: str
    unet: str
    vae: str
    clip: str
    projection: str

    @classmethod
    def from_dir(cls, root: str) -> "Zero123Paths":
        return cls(
            root=root,
            unet=os.path.join(root, "unet.safetensors"),
            vae=os.path.join(root, "vae.safetensors"),
            clip=os.path.join(root, "clip.safetensors"),
            projection=os.path.join(root, "projection.safetensors"),
        )

    def validate(self) -> None:
        missing = [path for path in self.__dict__.values() if path != self.root and not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError("Missing Zero123 tensor stores: " + ", ".join(missing))


class VisionAttention(nn.Module):
    def __init__(self, width: int = 1024, heads: int = 16):
        super().__init__()
        self.width = width
        self.heads = heads
        self.dim = width // heads
        self.scale = self.dim**-0.5
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        weight = store.get(prefix + ".in_proj_weight")
        bias = store.get(prefix + ".in_proj_bias")
        step = weight.shape[0] // 3
        for index, module in enumerate((self.q, self.k, self.v)):
            assign_param(module.weight, weight[index * step:(index + 1) * step])
            assign_param(module.bias, bias[index * step:(index + 1) * step])
        _assign_linear(self.out, store, prefix + ".out_proj")

    def execute(self, value: jt.Var) -> jt.Var:
        batch, tokens, _ = value.shape

        def split(tensor: jt.Var) -> jt.Var:
            return tensor.reshape(batch, tokens, self.heads, self.dim).permute(0, 2, 1, 3)

        query = split(self.q(value))
        key = split(self.k(value))
        hidden = split(self.v(value))
        scores = jt.matmul(query, key.permute(0, 1, 3, 2)) * self.scale
        hidden = jt.matmul(scores.softmax(-1), hidden)
        hidden = hidden.permute(0, 2, 1, 3).reshape(batch, tokens, self.width)
        return self.out(hidden)


class VisionBlock(nn.Module):
    def __init__(self, width: int = 1024, heads: int = 16):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = VisionAttention(width, heads)
        self.ln2 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)

    def load_from_store(self, store: SafeTensorStore, prefix: str) -> None:
        _assign_layer_norm(self.ln1, store, prefix + ".ln_1")
        self.attn.load_from_store(store, prefix + ".attn")
        _assign_layer_norm(self.ln2, store, prefix + ".ln_2")
        _assign_linear(self.fc1, store, prefix + ".mlp.c_fc")
        _assign_linear(self.fc2, store, prefix + ".mlp.c_proj")

    def execute(self, value: jt.Var) -> jt.Var:
        value = value + self.attn(self.ln1(value))
        return value + self.fc2(_quick_gelu(self.fc1(self.ln2(value))))


class ClipVision(nn.Module):
    def __init__(self, width: int = 1024, layers: int = 24, heads: int = 16, output_dim: int = 768):
        super().__init__()
        self.width = width
        self.conv = nn.Conv(3, width, kernel_size=14, stride=14, padding=0, bias=False)
        self.class_embedding = jt.zeros((width,))
        self.position_embedding = jt.zeros((257, width))
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = [VisionBlock(width, heads) for _ in range(layers)]
        self.ln_post = nn.LayerNorm(width)
        self.proj = jt.zeros((width, output_dim))
        self.mean = jt.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1)
        self.std = jt.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)

    def load_from_store(self, store: SafeTensorStore) -> None:
        assign_param(self.conv.weight, store.get("conv1.weight"))
        assign_param(self.class_embedding, store.get("class_embedding"))
        assign_param(self.position_embedding, store.get("positional_embedding"))
        _assign_layer_norm(self.ln_pre, store, "ln_pre")
        for index, block in enumerate(self.blocks):
            block.load_from_store(store, f"transformer.resblocks.{index}")
        _assign_layer_norm(self.ln_post, store, "ln_post")
        assign_param(self.proj, store.get("proj"))

    def preprocess(self, image: jt.Var) -> jt.Var:
        image = nn.interpolate(image, size=(224, 224), mode="bicubic", align_corners=True)
        image = (image + 1.0) / 2.0
        return (image - self.mean) / self.std

    def execute(self, image: jt.Var) -> jt.Var:
        hidden = self.conv(self.preprocess(image))
        batch, channels, height, width = hidden.shape
        hidden = hidden.reshape(batch, channels, height * width).permute(0, 2, 1)
        cls = self.class_embedding.reshape(1, 1, channels).repeat(batch, 1, 1)
        hidden = jt.concat([cls, hidden], dim=1) + self.position_embedding.unsqueeze(0)
        hidden = self.ln_pre(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return jt.matmul(self.ln_post(hidden[:, 0]), self.proj)


class Zero123Jittor(StableDiffusion15Jittor):
    """Official Zero123 inference path without a PyTorch runtime dependency."""

    def __init__(self):
        nn.Module.__init__(self)
        enable_cuda()
        jt.cudnn.set_max_workspace_ratio(0.05)
        self.unet_input = UNetInputStem(8, 320)
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
        self.vae_decoder_output = self.vae.decoder.conv_out
        self.vae_decoder = self.vae.decoder
        self.clip = ClipVision()
        self.cc_projection = nn.Linear(772, 768)
        self.scheduler = SD15Scheduler(steps_offset=1, set_alpha_to_one=False)
        self.root = ""

    @classmethod
    def from_pretrained(cls, root: str) -> "Zero123Jittor":
        model = cls()
        model.load_pretrained(root)
        return model

    def load_pretrained(self, root: str) -> None:
        paths = Zero123Paths.from_dir(root)
        paths.validate()
        unet = SafeTensorStore(paths.unet)
        self.unet_input.load_from_store(unet)
        self.time_embedding.load_from_store(unet)
        for index, block in enumerate(self.unet_down_blocks):
            block.load_from_store(unet, f"down_blocks.{index}")
        self.unet_mid_block.load_from_store(unet, "mid_block")
        for index, block in enumerate(self.unet_up_blocks):
            block.load_from_store(unet, f"up_blocks.{index}")
        self.unet_output.load_from_store(unet)
        self.vae.load_from_store(SafeTensorStore(paths.vae))
        self.clip.load_from_store(SafeTensorStore(paths.clip))
        projection = SafeTensorStore(paths.projection)
        assign_param(self.cc_projection.weight, projection.get("weight"))
        assign_param(self.cc_projection.bias, projection.get("bias"))
        self.root = root

    def condition(self, image: jt.Var, polar: float, azimuth: float, radius: float) -> Dict[str, jt.Var]:
        batch = image.shape[0]
        clip = self.clip(image).unsqueeze(1)
        camera = np.array(
            [math.radians(polar), math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth)), radius],
            dtype=np.float32,
        )
        camera = jt.array(camera).reshape(1, 1, 4).repeat(batch, 1, 1)
        context = self.cc_projection(jt.concat([clip, camera], dim=-1))
        return {"context": context, "image": self.vae.encode(image)}

    def predict_view(
        self,
        latents: jt.Var,
        timestep: int,
        condition: Dict[str, jt.Var],
        trace: Optional[Dict[str, jt.Var]] = None,
    ) -> jt.Var:
        model_input = jt.concat([latents, condition["image"]], dim=1)
        return self.predict_noise(model_input, timestep, condition["context"], trace)

    def sample_view(
        self,
        latents: jt.Var,
        condition: Dict[str, jt.Var],
        steps: int = 50,
        scale: float = 3.0,
    ) -> jt.Var:
        """Match the official Zero123 deterministic DDIM inference loop."""

        stride = self.scheduler.num_train_timesteps // steps
        # The upstream sampler applies ``timesteps[:-1]`` at its default
        # ``t_start=-1``. Preserve that behavior for exact output parity.
        timesteps = (np.arange(0, self.scheduler.num_train_timesteps, stride) + 1)[:-1][::-1]
        self.scheduler.timesteps_np = timesteps.astype(np.int32)
        uncond = {
            "context": jt.zeros_like(condition["context"]),
            "image": jt.zeros_like(condition["image"]),
        }
        sample = latents
        for timestep in self.scheduler.timesteps_np:
            sample_in = jt.concat([sample, sample], dim=0)
            joined = {
                "context": jt.concat([uncond["context"], condition["context"]], dim=0),
                "image": jt.concat([uncond["image"], condition["image"]], dim=0),
            }
            noise = self.predict_view(sample_in, int(timestep), joined)
            uncond_noise, cond_noise = noise[:1], noise[1:]
            guided = uncond_noise + scale * (cond_noise - uncond_noise)
            sample = self.scheduler.step(guided, int(timestep), sample)
            # Jittor is lazy: force each denoising step and detach its tiny
            # latent so a 50-step inference does not retain 50 UNet graphs.
            sample.sync()
            sample = jt.array(sample.numpy())
            jt.clean()
        return sample
