"""Separate-process parity test for the native Jittor SDXL UNet."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


ORDER = ["down0", "down1", "down2", "mid", "up0", "up1", "up2", "sample"]


def load_inputs(path: str) -> dict[str, np.ndarray]:
    if os.path.isfile(path):
        with np.load(path) as handle:
            return {key: handle[key] for key in handle.files}
    rng = np.random.default_rng(20260917)
    data = {
        "sample": rng.normal(0, 1, (1, 4, 16, 16)).astype("float32"),
        "timestep": np.array([500.0], dtype="float32"),
        "context": rng.normal(0, 0.2, (1, 77, 2048)).astype("float32"),
        "pooled": rng.normal(0, 0.2, (1, 1280)).astype("float32"),
        "time_ids": np.array([[256, 256, 0, 0, 256, 256]], dtype="float32"),
    }
    np.savez(path, **data)
    return data


def torch_run(root: str, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import torch
    from diffusers import UNet2DConditionModel

    device = torch.device("cuda")
    model = UNet2DConditionModel.from_pretrained(
        root, subfolder="unet", variant="fp16", torch_dtype=torch.float32, local_files_only=True
    ).to(device)
    trace: dict[str, np.ndarray] = {}
    hooks = []
    blocks = list(model.down_blocks) + [model.mid_block] + list(model.up_blocks)
    names = ORDER[:-1]
    for name, block in zip(names, blocks):
        def capture(_, __, output, key=name):
            value = output[0] if isinstance(output, tuple) else output
            trace[key] = value.detach().cpu().numpy()
        hooks.append(block.register_forward_hook(capture))
    tensors = {key: torch.from_numpy(value).to(device) for key, value in data.items()}
    with torch.inference_mode():
        output = model(
            tensors["sample"],
            tensors["timestep"],
            encoder_hidden_states=tensors["context"],
            added_cond_kwargs={"text_embeds": tensors["pooled"], "time_ids": tensors["time_ids"]},
        ).sample
    for hook in hooks:
        hook.remove()
    trace["sample"] = output.cpu().numpy()
    return trace


def jittor_run(root: str, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import jittor as jt
    from masactrl.sd15_jittor import SafeTensorStore
    from sdxl import SDXLUNet

    jt.flags.use_cuda = 1
    model = SDXLUNet()
    model.load_from_store(SafeTensorStore(os.path.join(root, "unet", "diffusion_pytorch_model.fp16.safetensors")))
    values = {key: jt.array(value) for key, value in data.items()}
    trace = {}
    with jt.no_grad():
        output = model(
            values["sample"], values["timestep"], values["context"],
            values["pooled"], values["time_ids"], trace,
        )
    result = {key: value.numpy() for key, value in trace.items()}
    result["sample"] = output.numpy()
    return result


def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = left.astype("float64").reshape(-1)
    right = right.astype("float64").reshape(-1)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return {"max_abs": float("inf"), "rel_rmse": float("inf"), "cosine": 0.0}
    norm = np.linalg.norm(left)
    denom = norm * np.linalg.norm(right)
    return {
        "max_abs": float(np.abs(left - right).max()),
        "rel_rmse": float(np.linalg.norm(left - right) / norm),
        "cosine": float(np.dot(left, right) / denom),
    }


def compare(source_path: str, target_path: str, report_path: str) -> dict:
    results = {}
    first = None
    with np.load(source_path) as source, np.load(target_path) as target:
        for key in ORDER:
            result = metrics(source[key], target[key])
            result["limit"] = 3e-3
            result["passed"] = result["rel_rmse"] <= result["limit"]
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
    parser.add_argument("--output_dir", default="/data/L202500448/maskfactory/logs/parity_sdxl_unet")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    data = load_inputs(os.path.join(args.output_dir, "inputs.npz"))
    if args.backend == "torch":
        output = torch_run(args.model_dir, data)
        np.savez(os.path.join(args.output_dir, "torch.npz"), **output)
    elif args.backend == "jittor":
        output = jittor_run(args.model_dir, data)
        np.savez(os.path.join(args.output_dir, "jittor.npz"), **output)
    else:
        report = compare(
            os.path.join(args.output_dir, "torch.npz"),
            os.path.join(args.output_dir, "jittor.npz"),
            os.path.join(args.output_dir, "report.json"),
        )
        raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
