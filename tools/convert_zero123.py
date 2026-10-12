"""Convert the official Zero123 checkpoint into runtime-only tensor stores.

The converter may use PyTorch because it is an offline migration tool.  The
generated Jittor runtime does not import PyTorch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
    create_unet_diffusers_config,
    create_vae_diffusers_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("config")
    parser.add_argument("output")
    parser.add_argument("--ema", action="store_true")
    return parser.parse_args()


def save_store(state: dict[str, torch.Tensor], path: Path) -> None:
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    save_file(tensors, str(path))


def extract_clip(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("cond_stage_model.model.visual.", "cond_stage_model.visual.")
    for prefix in prefixes:
        clip = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if clip:
            return clip
    raise KeyError("Zero123 CLIP visual weights were not found")


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    unet_config = create_unet_diffusers_config(config, image_size=256)
    vae_config = create_vae_diffusers_config(config, image_size=256)
    unet = convert_ldm_unet_checkpoint(
        dict(state),
        unet_config,
        path=args.checkpoint,
        extract_ema=args.ema,
    )
    vae = convert_ldm_vae_checkpoint(dict(state), vae_config)
    clip = extract_clip(state)
    projection = {
        key.removeprefix("cc_projection."): value
        for key, value in state.items()
        if key.startswith("cc_projection.")
    }
    if set(projection) != {"weight", "bias"}:
        raise KeyError("Zero123 cc_projection weights were not found")

    save_store(unet, output / "unet.safetensors")
    save_store(vae, output / "vae.safetensors")
    save_store(clip, output / "clip.safetensors")
    save_store(projection, output / "projection.safetensors")
    print(f"converted={output}")
    print(f"unet={len(unet)} vae={len(vae)} clip={len(clip)} projection={len(projection)}")


if __name__ == "__main__":
    main()
