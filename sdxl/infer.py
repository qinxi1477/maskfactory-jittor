"""Text-to-image inference with native Jittor SDXL components."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Union

import jittor as jt
import numpy as np
from PIL import Image
from transformers import CLIPTokenizer

from masactrl.sd15_jittor import SafeTensorStore, SD15Scheduler
from .model import SDXLParts
from .unet import SDXLUNet


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("output")
    parser.add_argument("--negative", default="")
    parser.add_argument("--model_dir", default="/data/L202500448/maskfactory/models/stable-diffusion-xl-base-1.0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def token_ids(root: str, folder: str, text: str) -> np.ndarray:
    tokenizer = CLIPTokenizer.from_pretrained(root, subfolder=folder, local_files_only=True)
    encoded = tokenizer(
        text,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="np",
    )
    return encoded.input_ids.astype("int32")


def prompt_embed(parts: SDXLParts, root: str, text: str):
    ids1 = jt.array(token_ids(root, "tokenizer", text)).int32()
    ids2 = jt.array(token_ids(root, "tokenizer_2", text)).int32()
    pen1, _, _ = parts.text1(ids1)
    pen2, _, pooled = parts.text2(ids2)
    return jt.concat([pen1, pen2], dim=-1), pooled


def save_image(image: jt.Var, output: str) -> None:
    value = jt.clamp((image[0] + 1.0) / 2.0, 0.0, 1.0).numpy()
    value = np.rint(value.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value).save(path)


class SDXLPipeline:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.parts = SDXLParts.from_pretrained(model_dir)
        jt.cudnn.set_max_workspace_ratio(0.02)
        self.unet = SDXLUNet()
        weight_path = os.path.join(model_dir, "unet", "diffusion_pytorch_model.fp16.safetensors")
        self.unet.load_from_store(SafeTensorStore(weight_path))

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "SDXLPipeline":
        return cls(model_dir)

    def attention_modules(self):
        return self.unet.attention_modules()

    def __call__(
        self,
        prompt: Union[str, List[str]],
        latents: jt.Var | None = None,
        negative_prompt: Union[str, List[str]] = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: int = 42,
    ) -> jt.Var:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        negatives = [negative_prompt] if isinstance(negative_prompt, str) else list(negative_prompt)
        if len(negatives) == 1:
            negatives *= len(prompts)
        if len(negatives) != len(prompts):
            raise ValueError("negative_prompt must contain one item or match prompt")

        schedule = SD15Scheduler(steps_offset=0, set_alpha_to_one=False)
        schedule.set_timesteps(num_inference_steps)
        positive, pos_pool = prompt_embed(self.parts, self.model_dir, prompts)
        negative, neg_pool = prompt_embed(self.parts, self.model_dir, negatives)
        context = jt.concat([negative, positive], dim=0)
        pooled = jt.concat([neg_pool, pos_pool], dim=0)
        batch = len(prompts)
        size = np.array(
            [[height, width, 0, 0, height, width]] * (batch * 2),
            dtype=np.float32,
        )
        time_ids = jt.array(size)
        if latents is None:
            rng = np.random.default_rng(seed)
            shape = (batch, 4, height // 8, width // 8)
            sample = jt.array(rng.standard_normal(shape, dtype=np.float32))
        else:
            sample = latents

        for timestep in schedule.timesteps_np:
            model_input = sample
            model_input = jt.concat([model_input, model_input], dim=0)
            step_time = jt.array(np.full((batch * 2,), timestep, dtype=np.float32))
            noise = self.unet(model_input, step_time, context, pooled, time_ids)
            uncond, cond = noise[:batch], noise[batch:]
            guided = uncond + guidance_scale * (cond - uncond)
            sample = schedule.step(guided, int(timestep), sample)
            sample.sync()
            sample = jt.array(sample.numpy())
            del model_input, step_time, noise, uncond, cond, guided
            jt.gc()
        if max(height, width) < 1024:
            return self.parts.decode_latent(sample)
        images = []
        for index in range(batch):
            image = self.parts.decode_latent(sample[index:index + 1])
            images.append(jt.array(image.numpy()))
            del image
            jt.gc()
        return jt.concat(images, dim=0)


def main() -> None:
    args = arguments()
    if args.height % 8 or args.width % 8:
        raise ValueError("height and width must be divisible by 8")
    jt.flags.use_cuda = 1
    pipeline = SDXLPipeline.from_pretrained(args.model_dir)
    with jt.no_grad():
        image = pipeline(
            args.prompt,
            negative_prompt=args.negative,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.scale,
            seed=args.seed,
        )
        save_image(image, args.output)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
