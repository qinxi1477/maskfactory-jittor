"""Run a short SDXL generation with MaskFactory attention control."""

from __future__ import annotations

import argparse
import numpy as np
import jittor as jt

from masactrl.masactrl import MutualSelfAttentionControl
from masactrl.masactrl_utils import register_attention_editor
from sdxl.infer import SDXLPipeline, save_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", default="/data/L202500448/maskfactory/logs/sdxl_editor.png")
    args = parser.parse_args()
    root = "/data/L202500448/maskfactory/models/stable-diffusion-xl-base-1.0"
    jt.flags.use_cuda = 1
    model = SDXLPipeline.from_pretrained(root)
    editor = MutualSelfAttentionControl(
        start_step=0, start_layer=0, total_steps=args.steps, model_type="SDXL"
    )
    register_attention_editor(model, editor)
    source = np.random.default_rng(42).standard_normal((1, 4, args.size // 8, args.size // 8), dtype=np.float32)
    latents = jt.array(np.repeat(source, 2, axis=0))
    with jt.no_grad():
        result = model(
            ["an old man", "an old man smiling"],
            latents=latents,
            height=args.size,
            width=args.size,
            num_inference_steps=args.steps,
            guidance_scale=5.0,
        )
        value = result.numpy()
        assert value.shape == (2, 3, args.size, args.size), value.shape
        assert np.isfinite(value).all()
        assert editor.cur_step == args.steps, editor.cur_step
        save_image(result[1:2], args.output)
    print(f"attention_layers={editor.num_att_layers} steps={editor.cur_step} output={args.output}")


if __name__ == "__main__":
    main()
