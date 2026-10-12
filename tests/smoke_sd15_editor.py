"""Run MaskFactory SD1.5 inversion and attention editing on a small image."""

from __future__ import annotations

import numpy as np
import jittor as jt

from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl import MutualSelfAttentionControl
from masactrl.masactrl_utils import register_attention_editor


def main() -> None:
    model_dir = "/data/L202500448/maskfactory/models/stable-diffusion-v1-5"
    jt.flags.use_cuda = 1
    model = MasaCtrlPipeline.from_pretrained(model_dir)
    axis = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis)
    image = jt.array(np.stack([grid_x, grid_y, grid_x * grid_y])[None])
    with jt.no_grad():
        noise, history = model.invert(
            image, "", num_inference_steps=2, guidance_scale=7.5,
            return_intermediates=True,
        )
        assert len(history) == 3, len(history)
        assert tuple(noise.shape) == (1, 4, 8, 8), noise.shape
        editor = MutualSelfAttentionControl(
            start_step=0, start_layer=0, total_steps=2, model_type="SD"
        )
        register_attention_editor(model, editor)
        result = model(
            ["a red cube", "a blue cube"],
            latents=noise.repeat(2, 1, 1, 1),
            height=64,
            width=64,
            num_inference_steps=2,
            guidance_scale=7.5,
        )
        value = result.numpy()
        assert value.shape == (2, 3, 64, 64), value.shape
        assert np.isfinite(value).all()
        assert editor.cur_step == 2, editor.cur_step
    print(f"attention_layers={editor.num_att_layers} steps={editor.cur_step}")


if __name__ == "__main__":
    main()
