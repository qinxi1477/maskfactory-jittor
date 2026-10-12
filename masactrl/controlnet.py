from dataclasses import dataclass
import os
from typing import List, Sequence

import jittor as jt
from jittor import nn
import numpy as np
from PIL import Image

from .jittor_compat import enable_cuda
from .sd15_jittor import (
    SafeTensorStore,
    StableDiffusion15Jittor,
    Timesteps,
    TimestepEmbedding,
    UNetCrossAttnDownBlock2D,
    UNetDownBlock2D,
    UNetMidBlock2DCrossAttn,
    _assign_conv,
    Conv,
    _silu,
)


def _weight(root: str) -> str:
    for name in ("diffusion_pytorch_model.fp16.safetensors", "diffusion_pytorch_model.safetensors"):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"ControlNet safetensors not found: {root}")


class CondEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = Conv(3, 16, 3, padding=1)
        self.blocks = [
            Conv(16, 16, 3, padding=1),
            Conv(16, 32, 3, stride=2, padding=1),
            Conv(32, 32, 3, padding=1),
            Conv(32, 96, 3, stride=2, padding=1),
            Conv(96, 96, 3, padding=1),
            Conv(96, 256, 3, stride=2, padding=1),
        ]
        self.conv_out = Conv(256, 320, 3, padding=1)

    def load_from_store(self, store: SafeTensorStore):
        prefix = "controlnet_cond_embedding"
        _assign_conv(self.conv_in, store, prefix + ".conv_in")
        for index, block in enumerate(self.blocks):
            _assign_conv(block, store, f"{prefix}.blocks.{index}")
        _assign_conv(self.conv_out, store, prefix + ".conv_out")

    def execute(self, image: jt.Var) -> jt.Var:
        hidden = _silu(self.conv_in(image))
        for block in self.blocks:
            hidden = _silu(block(hidden))
        return self.conv_out(hidden)


class ControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        enable_cuda()
        self.conv_in = Conv(4, 320, 3, padding=1)
        self.time_proj = Timesteps(320, flip_sin_to_cos=True, downscale_freq_shift=0.0)
        self.time_embedding = TimestepEmbedding(320, 1280)
        self.cond_embedding = CondEmbed()
        self.down_blocks = [
            UNetCrossAttnDownBlock2D(320, 320, add_downsample=True),
            UNetCrossAttnDownBlock2D(320, 640, add_downsample=True),
            UNetCrossAttnDownBlock2D(640, 1280, add_downsample=True),
            UNetDownBlock2D(1280, 1280, add_downsample=False),
        ]
        self.mid_block = UNetMidBlock2DCrossAttn(1280)
        channels = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
        self.control_blocks = [Conv(channels[i], channels[i], 1) for i in range(len(channels))]
        self.control_mid = Conv(1280, 1280, 1)

    @classmethod
    def from_pretrained(cls, root: str):
        model = cls()
        model.load_pretrained(root)
        return model

    def load_pretrained(self, root: str):
        store = SafeTensorStore(_weight(root))
        _assign_conv(self.conv_in, store, "conv_in")
        self.time_embedding.load_from_store(store)
        self.cond_embedding.load_from_store(store)
        for index, block in enumerate(self.down_blocks):
            block.load_from_store(store, f"down_blocks.{index}")
        self.mid_block.load_from_store(store, "mid_block")
        for index, block in enumerate(self.control_blocks):
            _assign_conv(block, store, f"controlnet_down_blocks.{index}")
        _assign_conv(self.control_mid, store, "controlnet_mid_block")

    def execute(
        self,
        sample: jt.Var,
        timestep: int,
        encoder_hidden_states: jt.Var,
        condition: jt.Var,
        scale: float = 1.0,
    ):
        timesteps = jt.array([int(timestep)]).int32().repeat(sample.shape[0])
        temb = self.time_embedding(self.time_proj(timesteps))
        hidden = self.conv_in(sample) + self.cond_embedding(condition)
        residuals = [hidden]
        for block in self.down_blocks:
            if isinstance(block, UNetCrossAttnDownBlock2D):
                hidden, values = block(hidden, temb, encoder_hidden_states, return_residuals=True)
            else:
                hidden, values = block(hidden, temb, return_residuals=True)
            residuals.extend(values)
        hidden = self.mid_block(hidden, temb, encoder_hidden_states)
        down = [block(value) * scale for block, value in zip(self.control_blocks, residuals)]
        return down, self.control_mid(hidden) * scale


@dataclass
class Output:
    images: List[Image.Image]


class MultiControl(nn.Module):
    def __init__(self, base: StableDiffusion15Jittor, controls: Sequence[ControlNet], tokenizer):
        super().__init__()
        self.base = base
        self.controls = list(controls)
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, base_path: str, control_paths: Sequence[str]):
        from transformers import CLIPTokenizer

        base = StableDiffusion15Jittor.from_pretrained(base_path)
        jt.cudnn.set_max_workspace_ratio(0.05)
        controls = [ControlNet.from_pretrained(path) for path in control_paths]
        tokenizer = CLIPTokenizer.from_pretrained(os.path.join(base_path, "tokenizer"), local_files_only=True)
        return cls(base, controls, tokenizer)

    def _text(self, prompt: str, negative_prompt: str) -> jt.Var:
        tokens = self.tokenizer(
            [negative_prompt, prompt],
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="np",
        )
        return self.base.text_encoder(jt.array(tokens.input_ids).int32())

    @staticmethod
    def _image(image, width: int, height: int) -> jt.Var:
        image = image if isinstance(image, Image.Image) else Image.open(image)
        image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return jt.array(array)

    def execute(
        self,
        prompt: str,
        image: Sequence[Image.Image],
        negative_prompt: str = "",
        seed: int = 0,
        num_inference_steps: int = 50,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 7.5,
    ) -> Output:
        if len(image) != len(self.controls):
            raise ValueError(f"Expected {len(self.controls)} control images, received {len(image)}")
        text = self._text(prompt, negative_prompt)
        conditions = [self._image(value, width, height).repeat(2, 1, 1, 1) for value in image]
        noise = np.random.RandomState(seed).standard_normal((1, 4, height // 8, width // 8)).astype("float32")
        latents = jt.array(noise)
        timesteps = self.base.scheduler.set_timesteps(num_inference_steps).numpy().tolist()
        for timestep in timesteps:
            model_input = jt.concat([latents, latents], dim=0)
            down_sum = None
            mid_sum = None
            for control, condition in zip(self.controls, conditions):
                down, mid = control(model_input, int(timestep), text, condition)
                down = [jt.array(value.numpy()) for value in down]
                mid = jt.array(mid.numpy())
                down_sum = down if down_sum is None else [left + right for left, right in zip(down_sum, down)]
                mid_sum = mid if mid_sum is None else mid_sum + mid
            noise_pred = self.base.predict_noise(
                model_input,
                int(timestep),
                text,
                down_residuals=down_sum,
                mid_residual=mid_sum,
            )
            noise_uncond, noise_text = noise_pred[:1], noise_pred[1:]
            guided = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            latents = self.base.scheduler.step(guided, int(timestep), latents)
            latents.sync()
            latents = jt.array(latents.numpy())
            del model_input, down_sum, mid_sum, down, mid, noise_pred, noise_uncond, noise_text, guided
            jt.gc()
        decoded = self.base.decode_latent(latents)
        pixels = ((decoded.numpy().transpose(0, 2, 3, 1) / 2.0 + 0.5).clip(0, 1) * 255).astype("uint8")
        return Output([Image.fromarray(value) for value in pixels])
