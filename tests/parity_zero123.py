"""Layerwise PyTorch/Jittor parity for the official Zero123 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np


ORDER = ["clip", "context", "encode", "noise", "decode", "sample", "result"]
TRACE = ["time", "input", "down0", "down1", "down2", "down3", "mid", "up0", "up1", "up2", "up3", "output"]
LIMITS = {
    "clip": ("max_abs", 3e-3),
    "context": ("max_abs", 3e-3),
    # The unscaled VAE posterior spans roughly [-19, 14]. Relative RMSE is
    # stable across activation scale and is also used for intermediate traces.
    "encode": ("rel_rmse", 2e-3),
    "noise": ("max_abs", 1e-2),
    "decode": ("max_abs", 1e-2),
    "sample": ("max_abs", 3e-2),
    "result": ("max_abs", 5e-2),
}
TRACE_LIMIT = 3e-3


def make_inputs(path: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260916)
    data = {
        "image": rng.normal(0.0, 0.3, size=(1, 3, 256, 256)).clip(-1, 1).astype("float32"),
        "latent": rng.normal(0.0, 1.0, size=(1, 4, 32, 32)).astype("float32"),
        "camera": np.array([15.0, 30.0, 0.1], dtype="float32"),
        "timestep": np.array([751], dtype="int64"),
    }
    np.savez(path, **data)
    return data


def load_inputs(path: str) -> dict[str, np.ndarray]:
    if not os.path.isfile(path):
        return make_inputs(path)
    with np.load(path) as handle:
        return {key: handle[key] for key in handle.files}


def camera_tensor(torch, values, device):
    polar, azimuth, radius = (float(value) for value in values)
    return torch.tensor(
        [math.radians(polar), math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth)), radius],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 4)


def torch_run(checkpoint: str, config_path: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import torch
    from omegaconf import OmegaConf

    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm.util import instantiate_from_config

    print("[torch] loading official Zero123", flush=True)
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload.get("state_dict", payload), strict=False)
    if missing or unexpected:
        print(f"[torch] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model = model.cuda().eval()
    device = torch.device("cuda")
    image = torch.from_numpy(inputs["image"]).to(device)
    latent = torch.from_numpy(inputs["latent"]).to(device)
    timestep = torch.from_numpy(inputs["timestep"]).to(device)
    camera = camera_tensor(torch, inputs["camera"], device)
    output: dict[str, np.ndarray] = {}
    trace: dict[str, np.ndarray] = {}
    diffusion = model.model.diffusion_model

    def save(name):
        def hook(_, __, value):
            value = value[0] if isinstance(value, tuple) else value
            trace[name] = value.detach().float().cpu().numpy()
        return hook

    points = [
        (diffusion.time_embed, "time"),
        (diffusion.input_blocks[0], "input"),
        (diffusion.input_blocks[3], "down0"),
        (diffusion.input_blocks[6], "down1"),
        (diffusion.input_blocks[9], "down2"),
        (diffusion.input_blocks[11], "down3"),
        (diffusion.middle_block, "mid"),
        (diffusion.output_blocks[2], "up0"),
        (diffusion.output_blocks[5], "up1"),
        (diffusion.output_blocks[8], "up2"),
        (diffusion.output_blocks[11], "up3"),
        (diffusion.out, "output"),
    ]
    hooks = [module.register_forward_hook(save(name)) for module, name in points]
    with torch.inference_mode(), model.ema_scope("parity"):
        clip = model.get_learned_conditioning(image)
        context = model.cc_projection(torch.cat([clip, camera], dim=-1))
        encoded = model.encode_first_stage(image).mode()
        cond = {"c_crossattn": [context], "c_concat": [encoded]}
        noise = model.apply_model(latent, timestep, cond)
        for hook in hooks:
            hook.remove()
        decoded = model.decode_first_stage(latent)
        uncond = {
            "c_crossattn": [torch.zeros_like(context)],
            "c_concat": [torch.zeros_like(encoded)],
        }
        sample, _ = DDIMSampler(model).sample(
            S=4,
            conditioning=cond,
            batch_size=1,
            shape=[4, 32, 32],
            verbose=False,
            unconditional_guidance_scale=3.0,
            unconditional_conditioning=uncond,
            eta=0.0,
            x_T=latent,
        )
        result = model.decode_first_stage(sample)
    output.update(
        clip=clip.float().cpu().numpy(),
        context=context.float().cpu().numpy(),
        encode=encoded.float().cpu().numpy(),
        noise=noise.float().cpu().numpy(),
        decode=decoded.float().cpu().numpy(),
        sample=sample.float().cpu().numpy(),
        result=result.float().cpu().numpy(),
    )
    output.update({"trace_" + key: value for key, value in trace.items()})
    print("[torch] complete", flush=True)
    return output


def jittor_run(model_dir: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import jittor as jt

    from zero123 import Zero123Jittor

    print("[jittor] loading Zero123", flush=True)
    jt.flags.use_cuda = 1
    model = Zero123Jittor.from_pretrained(model_dir)
    image = jt.array(inputs["image"])
    latent = jt.array(inputs["latent"])
    polar, azimuth, radius = (float(value) for value in inputs["camera"])
    clip = model.clip(image).unsqueeze(1)
    condition = model.condition(image, polar, azimuth, radius)
    trace = {}
    noise = model.predict_view(latent, int(inputs["timestep"][0]), condition, trace)
    sample = model.sample_view(latent, condition, steps=4, scale=3.0)
    output = {
        "clip": clip.numpy(),
        "context": condition["context"].numpy(),
        "encode": condition["image"].numpy(),
        "noise": noise.numpy(),
        "decode": model.decode_latent(latent).numpy(),
        "sample": sample.numpy(),
        "result": model.decode_latent(sample).numpy(),
    }
    output.update({"trace_" + key: value.numpy() for key, value in trace.items()})
    print("[jittor] complete", flush=True)
    return output


def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = left.astype("float64").reshape(-1)
    right = right.astype("float64").reshape(-1)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return {"max_abs": float("inf"), "mean_abs": float("inf"), "rel_rmse": float("inf"), "cosine": 0.0}
    diff = np.abs(left - right)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    norm = np.linalg.norm(left)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_rmse": float(np.linalg.norm(left - right) / norm) if norm else 0.0,
        "cosine": float(np.dot(left, right) / denom) if denom else 1.0,
    }


def compare(torch_path: str, jittor_path: str, report_path: str) -> dict:
    with np.load(torch_path) as source, np.load(jittor_path) as target:
        results = {}
        first = None
        for key in ORDER:
            result = metrics(source[key], target[key])
            metric, limit = LIMITS[key]
            result["metric"] = metric
            result["limit"] = limit
            result["passed"] = result[metric] <= limit
            results[key] = result
            if first is None and not result["passed"]:
                first = key
        traces = {}
        first_trace = None
        for key in TRACE:
            result = metrics(source["trace_" + key], target["trace_" + key])
            result["limit"] = TRACE_LIMIT
            result["passed"] = result["rel_rmse"] <= TRACE_LIMIT
            traces[key] = result
            if first_trace is None and not result["passed"]:
                first_trace = key
    report = {
        "passed": first is None and first_trace is None,
        "first_divergence": first,
        "first_trace": first_trace,
        "results": results,
        "trace": traces,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["torch", "jittor", "compare"])
    parser.add_argument("--checkpoint", default="/data/L202500448/maskfactory/models/zero123/105000.ckpt")
    parser.add_argument("--config", default="/data/L202500448/maskfactory/repo/MaskFactory/zero123/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml")
    parser.add_argument("--model_dir", default="/data/L202500448/maskfactory/models/zero123/jittor_ema")
    parser.add_argument("--output_dir", default="/data/L202500448/maskfactory/logs/parity_zero123")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    inputs = load_inputs(os.path.join(args.output_dir, "inputs.npz"))
    if args.backend == "torch":
        np.savez(os.path.join(args.output_dir, "torch.npz"), **torch_run(args.checkpoint, args.config, inputs))
    elif args.backend == "jittor":
        np.savez(os.path.join(args.output_dir, "jittor.npz"), **jittor_run(args.model_dir, inputs))
    else:
        report = compare(
            os.path.join(args.output_dir, "torch.npz"),
            os.path.join(args.output_dir, "jittor.npz"),
            os.path.join(args.output_dir, "report.json"),
        )
        raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
