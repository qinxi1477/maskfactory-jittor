"""PyTorch/Jittor parity for SDXL text encoders and VAE."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


ORDER = [
    "text1", "final1", "pool1", "text2", "final2", "pool2",
    "vae_in", "vae_down0", "vae_down1", "vae_down2", "vae_down3",
    "vae_mid", "vae_out", "encode", "decode",
]
LIMITS = {
    "text1": ("rel_rmse", 2e-3),
    "final1": ("rel_rmse", 2e-3),
    "pool1": ("rel_rmse", 2e-3),
    "text2": ("rel_rmse", 2e-3),
    "final2": ("rel_rmse", 2e-3),
    "pool2": ("rel_rmse", 2e-3),
    "vae_in": ("rel_rmse", 2e-3),
    "vae_down0": ("rel_rmse", 2e-3),
    "vae_down1": ("rel_rmse", 2e-3),
    "vae_down2": ("rel_rmse", 2e-3),
    "vae_down3": ("rel_rmse", 2e-3),
    "vae_mid": ("rel_rmse", 2e-3),
    "vae_out": ("rel_rmse", 6e-3),
    "encode": ("rel_rmse", 6e-3),
    "decode": ("max_abs", 2e-2),
}


def inputs(path: str) -> dict[str, np.ndarray]:
    if os.path.isfile(path):
        with np.load(path) as handle:
            return {key: handle[key] for key in handle.files}
    rng = np.random.default_rng(20260917)
    ids1 = rng.integers(3, 1000, size=(1, 77), dtype=np.int32)
    ids2 = ids1.copy()
    ids1[:, 0], ids1[:, -1] = 49406, 49407
    ids2[:, 0], ids2[:, -1] = 49406, 49407
    axis = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis)
    image = np.stack(
        [grid_x, grid_y, np.sin(np.pi * grid_x) * np.cos(np.pi * grid_y)],
        axis=0,
    )[None]
    data = {
        "ids1": ids1,
        "ids2": ids2,
        "image": image.astype("float32"),
        "latent": rng.normal(0, 1, size=(1, 4, 32, 32)).astype("float32"),
    }
    np.savez(path, **data)
    return data


def torch_run(root: str, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import torch
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTextModelWithProjection

    device = torch.device("cuda")
    ids1 = torch.from_numpy(data["ids1"]).to(device)
    ids2 = torch.from_numpy(data["ids2"]).to(device)
    image = torch.from_numpy(data["image"]).to(device)
    latent = torch.from_numpy(data["latent"]).to(device)
    with torch.inference_mode():
        text1 = CLIPTextModel.from_pretrained(
            root, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float32, local_files_only=True
        ).to(device)
        out1 = text1(ids1, output_hidden_states=True)
        pen1 = out1.hidden_states[-2]
        final1 = out1.last_hidden_state
        pool1 = out1.pooler_output
        del text1
        text2 = CLIPTextModelWithProjection.from_pretrained(
            root, subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float32, local_files_only=True
        ).to(device)
        out2 = text2(ids2, output_hidden_states=True)
        pen2 = out2.hidden_states[-2]
        final2 = out2.last_hidden_state
        pool2 = out2.text_embeds
        del text2
        vae = AutoencoderKL.from_pretrained(
            root, subfolder="vae", variant="fp16", torch_dtype=torch.float32, local_files_only=True
        ).to(device)
        hidden = vae.encoder.conv_in(image)
        vae_trace = {"vae_in": hidden.cpu().numpy()}
        for index, block in enumerate(vae.encoder.down_blocks):
            hidden = block(hidden)
            vae_trace[f"vae_down{index}"] = hidden.cpu().numpy()
        hidden = vae.encoder.mid_block(hidden)
        vae_trace["vae_mid"] = hidden.cpu().numpy()
        hidden = vae.encoder.conv_out(vae.encoder.conv_act(vae.encoder.conv_norm_out(hidden)))
        vae_trace["vae_out"] = hidden.cpu().numpy()
        moments = vae.quant_conv(hidden)
        encoded = moments.chunk(2, dim=1)[0] * vae.config.scaling_factor
        decoded = vae.decode(latent / vae.config.scaling_factor).sample
    result = {
        "text1": pen1.cpu().numpy(), "final1": final1.cpu().numpy(), "pool1": pool1.cpu().numpy(),
        "text2": pen2.cpu().numpy(), "final2": final2.cpu().numpy(), "pool2": pool2.cpu().numpy(),
        "encode": encoded.cpu().numpy(), "decode": decoded.cpu().numpy(),
    }
    result.update(vae_trace)
    return result


def jittor_run(root: str, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import jittor as jt
    from masactrl.sd15_jittor import _silu
    from sdxl import SDXLParts
    from sdxl.model import stable_norm

    jt.flags.use_cuda = 1
    model = SDXLParts.from_pretrained(root)
    with jt.no_grad():
        pen1, final1, pool1 = model.text1(jt.array(data["ids1"]).int32())
        pen2, final2, pool2 = model.text2(jt.array(data["ids2"]).int32())
        hidden = model.vae.encoder.conv_in(jt.array(data["image"]))
        vae_trace = {"vae_in": hidden.numpy()}
        for index, block in enumerate(model.vae.encoder.down_blocks):
            hidden = block(hidden)
            vae_trace[f"vae_down{index}"] = hidden.numpy()
        hidden = model.vae.encoder.mid_block(hidden)
        vae_trace["vae_mid"] = hidden.numpy()
        hidden = model.vae.encoder.conv_out(
            _silu(stable_norm(hidden, model.vae.encoder.conv_norm_out))
        )
        vae_trace["vae_out"] = hidden.numpy()
        moments = model.vae.encoder.quant_conv(hidden)
        encoded = moments[:, :4] * model.scale
        decoded = model.decode_latent(jt.array(data["latent"]))
    output = {
        "text1": pen1.numpy(), "final1": final1.numpy(), "pool1": pool1.numpy(),
        "text2": pen2.numpy(), "final2": final2.numpy(), "pool2": pool2.numpy(),
        "encode": encoded.numpy(), "decode": decoded.numpy(),
    }
    output.update(vae_trace)
    return output


def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = left.astype("float64").reshape(-1)
    right = right.astype("float64").reshape(-1)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return {"max_abs": float("inf"), "mean_abs": float("inf"), "rel_rmse": float("inf"), "cosine": 0.0}
    diff = np.abs(left - right)
    norm = np.linalg.norm(left)
    denom = norm * np.linalg.norm(right)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_rmse": float(np.linalg.norm(left - right) / norm) if norm else 0.0,
        "cosine": float(np.dot(left, right) / denom) if denom else 1.0,
    }


def compare(source_path: str, target_path: str, report_path: str) -> dict:
    with np.load(source_path) as source, np.load(target_path) as target:
        results = {}
        first = None
        for key in ORDER:
            result = metrics(source[key], target[key])
            metric, limit = LIMITS[key]
            result.update(metric=metric, limit=limit, passed=result[metric] <= limit)
            results[key] = result
            if first is None and not result["passed"]:
                first = key
    report = {"passed": first is None, "first_divergence": first, "results": results}
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["torch", "jittor", "compare"])
    parser.add_argument("--model_dir", default="/data/L202500448/maskfactory/models/stable-diffusion-xl-base-1.0")
    parser.add_argument("--output_dir", default="/data/L202500448/maskfactory/logs/parity_sdxl")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    data = inputs(os.path.join(args.output_dir, "inputs.npz"))
    if args.backend == "torch":
        np.savez(os.path.join(args.output_dir, "torch.npz"), **torch_run(args.model_dir, data))
    elif args.backend == "jittor":
        np.savez(os.path.join(args.output_dir, "jittor.npz"), **jittor_run(args.model_dir, data))
    else:
        report = compare(
            os.path.join(args.output_dir, "torch.npz"), os.path.join(args.output_dir, "jittor.npz"),
            os.path.join(args.output_dir, "report_parts.json"),
        )
        raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
