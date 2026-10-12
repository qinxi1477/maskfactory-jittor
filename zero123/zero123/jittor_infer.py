"""Generate a Zero123 novel view with the native Jittor runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import jittor as jt
import numpy as np
from PIL import Image

from .jittor_model import Zero123Jittor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--model_dir", default="/data/L202500448/maskfactory/models/zero123/jittor_ema")
    parser.add_argument("--polar", type=float, default=0.0)
    parser.add_argument("--azimuth", type=float, default=30.0)
    parser.add_argument("--radius", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_image(path: str) -> jt.Var:
    image = Image.open(path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0
    return jt.array(array[None])


def save_image(value: jt.Var, path: str) -> None:
    array = jt.clamp((value[0] + 1.0) / 2.0, 0.0, 1.0).numpy()
    array = np.rint(array.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output)


def main() -> None:
    args = parse_args()
    jt.flags.use_cuda = 1
    model = Zero123Jittor.from_pretrained(args.model_dir)
    image = load_image(args.input)
    with jt.no_grad():
        condition = model.condition(image, args.polar, args.azimuth, args.radius)
        rng = np.random.default_rng(args.seed)
        latent = jt.array(rng.standard_normal((1, 4, 32, 32), dtype=np.float32))
        sample = model.sample_view(latent, condition, args.steps, args.scale)
        save_image(model.decode_latent(sample), args.output)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
