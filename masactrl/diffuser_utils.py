"""Jittor implementation of the MaskFactory diffusion pipeline."""

from __future__ import annotations

import os
import json
from typing import List, Optional, Union

import jittor as jt
from jittor import nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from .jittor_compat import enable_cuda
from .sd15_jittor import StableDiffusion15Jittor


Prompt = Union[str, List[str]]


# 引入生成器和判别器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 定义生成器网络结构
        self.main = nn.Sequential(
            nn.Conv(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm(256),
            nn.ReLU(),
            nn.ConvTranspose(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.ConvTranspose(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.ConvTranspose(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def execute(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 定义判别器网络结构
        self.main = nn.Sequential(
            nn.Conv(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm(128),
            nn.LeakyReLU(0.2),
            nn.Conv(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm(256),
            nn.LeakyReLU(0.2),
            nn.Conv(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def execute(self, x):
        return self.main(x)


class MasaCtrlPipeline(nn.Module):
    """Stable Diffusion 1.5 and image-generator inference in Jittor."""

    def __init__(self, backend=None, generator=None, discriminator=None, tokenizer=None):
        super().__init__()
        enable_cuda()
        self.backend = backend
        self.generator = generator if generator is not None else Generator()
        self.discriminator = discriminator if discriminator is not None else Discriminator()
        self.optimizer_G = jt.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = jt.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion_GAN = nn.BCELoss()
        self.criterion_content = nn.L1Loss()
        self.criterion_structure = nn.MSELoss()
        self.tokenizer = tokenizer
        self.scheduler = backend.scheduler if backend is not None else None
        self.unet = backend

    @classmethod
    def from_pretrained(cls, model_path, *args, **kwargs):
        unet_config = os.path.join(model_path, "unet", "config.json")
        if os.path.isfile(unet_config):
            with open(unet_config, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            if config.get("cross_attention_dim") != 768 or config.get("sample_size") != 64:
                raise NotImplementedError("This Jittor backend currently supports Stable Diffusion 1.x weights")
            backend = StableDiffusion15Jittor.from_pretrained(model_path)
            tokenizer = cls._load_tokenizer(model_path)
            return cls(backend=backend, tokenizer=tokenizer)

        pipeline = cls()
        path = cls._generator_path(model_path)
        if path is None:
            raise FileNotFoundError(model_path)
        pipeline.generator.load(path)
        return pipeline

    @staticmethod
    def _generator_path(model_path: str) -> Optional[str]:
        if os.path.isfile(model_path):
            return model_path
        if not os.path.isdir(model_path):
            return None
        names = ["generator.pkl", "generator.pth", "model.pkl", "model.pth"]
        return next(
            (os.path.join(model_path, name) for name in names if os.path.isfile(os.path.join(model_path, name))),
            None,
        )

    @staticmethod
    def _load_tokenizer(model_path: str):
        from transformers import CLIPTokenizer

        return CLIPTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"), local_files_only=True)

    def _encode_text(self, prompt: Prompt) -> jt.Var:
        if self.tokenizer is None or self.backend is None:
            raise RuntimeError("A Stable Diffusion model directory is required for prompt inference")
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )
        return self.backend.text_encoder(jt.array(tokens.input_ids).int32())

    def next_step(self, model_output, timestep, x, eta=0.0, verbose=False):
        if verbose:
            print("timestep:", timestep)
        x_next = self.scheduler.next_step(model_output, int(timestep), x)
        step_ratio = self.scheduler.num_train_timesteps // len(self.scheduler.timesteps_np)
        current = min(int(timestep) - step_ratio, self.scheduler.num_train_timesteps - 1)
        alpha = self.scheduler.alphas_cumprod_np[current] if current >= 0 else self.scheduler.final_alpha_cumprod
        beta = 1.0 - float(alpha)
        pred_x0 = (x - beta**0.5 * model_output) / float(alpha) ** 0.5
        return x_next, pred_x0

    def step(self, model_output, timestep, x, eta=0.0, verbose=False):
        if verbose:
            print("timestep:", timestep)
        x_prev = self.scheduler.step(model_output, int(timestep), x)
        alpha = float(self.scheduler.alphas_cumprod_np[int(timestep)])
        beta = 1.0 - alpha
        pred_x0 = (x - beta**0.5 * model_output) / alpha**0.5
        return x_prev, pred_x0

    def image2latent(self, image):
        if self.backend is None:
            return self._image_tensor(image)
        return self.backend.encode_image(self._image_tensor(image))

    @staticmethod
    def _image_tensor(image):
        if isinstance(image, Image.Image):
            array = np.array(image.convert("RGB"), dtype=np.float32)
            image = jt.array(array).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        return image

    def latent2image(self, latents, return_type="np"):
        image = jt.clamp(self.latent2image_grad(latents) / 2 + 0.5, 0, 1)
        if return_type == "pt":
            return image
        array = image.numpy()[0].transpose(1, 2, 0)
        return (array * 255).astype(np.uint8)

    def latent2image_grad(self, latents):
        if self.backend is None:
            return latents
        return self.backend.decode_latent(latents)

    def execute(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        **kwargs,
    ):
        if not isinstance(prompt, (str, list)):
            return self.generator(prompt)
        if self.backend is None:
            raise RuntimeError("Prompt inference requires an SD1.5 model directory")

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if isinstance(prompt, str) and batch_size > 1:
            prompts *= batch_size
        batch_size = len(prompts)
        text_embeddings = self._encode_text(prompts)
        if kwargs.get("dir") is not None:
            text_embeddings = self._apply_direction(text_embeddings, float(kwargs["dir"]))

        shape = (batch_size, 4, height // 8, width // 8)
        if latents is None:
            latents = jt.randn(shape)
        elif tuple(latents.shape) != shape:
            raise ValueError(f"latent shape {tuple(latents.shape)} != {shape}")

        if guidance_scale > 1.0:
            negative = neg_prompt if neg_prompt is not None else ""
            negative = [negative] * batch_size if isinstance(negative, str) else negative
            text_embeddings = jt.concat([self._encode_text(negative), text_embeddings], dim=0)

        timesteps = self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents]
        pred_x0_list = [latents]
        for index, raw_timestep in enumerate(tqdm(timesteps.numpy().tolist(), desc="DDIM Sampler")):
            timestep = int(raw_timestep)
            if ref_intermediate_latents is not None:
                current = latents[latents.shape[0] // 2:]
                latents = jt.concat([ref_intermediate_latents[-1 - index], current], dim=0)
            model_input = jt.concat([latents, latents], dim=0) if guidance_scale > 1.0 else latents
            step_text = text_embeddings
            if isinstance(unconditioning, list):
                conditional = text_embeddings[text_embeddings.shape[0] // 2:]
                step_text = jt.concat([unconditioning[index], conditional], dim=0)
            noise_pred = self.backend.predict_noise(model_input, timestep, step_text)
            if guidance_scale > 1.0:
                split = noise_pred.shape[0] // 2
                noise_uncond, noise_text = noise_pred[:split], noise_pred[split:]
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            latents, pred_x0 = self.step(noise_pred, timestep, latents, eta)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            pred_images = [self.latent2image(value, return_type="pt") for value in pred_x0_list]
            latent_images = [self.latent2image(value, return_type="pt") for value in latents_list]
            return image, pred_images, latent_images
        return image

    @staticmethod
    def _apply_direction(text_embeddings: jt.Var, scale: float) -> jt.Var:
        values = text_embeddings.numpy()
        if values.shape[0] < 2:
            return text_embeddings
        direction = values[-2] - values[-1]
        matrix = direction.T
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
        _, _, right = np.linalg.svd(matrix, full_matrices=False)
        values[-1] = values[-1] + scale * right[0][:, None]
        return jt.array(values)

    def invert(
        self,
        image,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwargs,
    ):
        if self.backend is None:
            raise RuntimeError("DDIM inversion requires an SD1.5 model directory")
        image = self._image_tensor(image)
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if image.shape[0] == 1 and len(prompts) > 1:
            image = image.repeat(len(prompts), 1, 1, 1)
        text_embeddings = self._encode_text(prompts)
        latents = self.image2latent(image)
        start_latents = latents
        if guidance_scale > 1.0:
            text_embeddings = jt.concat([self._encode_text([""] * len(prompts)), text_embeddings], dim=0)

        timesteps = self.scheduler.set_timesteps(num_inference_steps).numpy().tolist()
        latents_list = [latents]
        for raw_timestep in tqdm(reversed(timesteps), desc="DDIM Inversion"):
            timestep = int(raw_timestep)
            model_input = jt.concat([latents, latents], dim=0) if guidance_scale > 1.0 else latents
            noise_pred = self.backend.predict_noise(model_input, timestep, text_embeddings)
            if guidance_scale > 1.0:
                split = noise_pred.shape[0] // 2
                noise_uncond, noise_text = noise_pred[:split], noise_pred[split:]
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            latents, _ = self.next_step(noise_pred, timestep, latents, eta)
            latents_list.append(latents)
        if return_intermediates:
            return latents, latents_list
        return latents, start_latents
