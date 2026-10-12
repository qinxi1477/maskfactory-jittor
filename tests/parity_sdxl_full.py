"""Compare the original SDXL DDIM sampling path across PyTorch and Jittor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROMPT = "a red cube on a white background"
NEGATIVE = ""
SIZE = 128
STEPS = 2
SCALE = 5.0
LIMITS = {
    "context": 0.002,
    "pooled": 0.002,
    "initial": 0.00001,
    "noise0": 0.003,
    "sample0": 0.003,
    "noise1": 0.003,
    "sample1": 0.003,
    "decoded": 0.02,
}


def source_latent() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((1, 4, SIZE // 8, SIZE // 8), dtype=np.float32)


def torch_run(root: str) -> dict[str, np.ndarray]:
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    device = torch.device("cuda")
    with torch.inference_mode():
        ids = []
        for folder in ("tokenizer", "tokenizer_2"):
            tokenizer = CLIPTokenizer.from_pretrained(root, subfolder=folder, local_files_only=True)
            encoded = tokenizer([NEGATIVE, PROMPT], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
            ids.append(encoded.input_ids.to(device))
        text1 = CLIPTextModel.from_pretrained(root, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float32, local_files_only=True).to(device)
        output1 = text1(ids[0], output_hidden_states=True)
        pen1 = output1.hidden_states[-2]
        del text1, output1
        text2 = CLIPTextModelWithProjection.from_pretrained(root, subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float32, local_files_only=True).to(device)
        output2 = text2(ids[1], output_hidden_states=True)
        context = torch.cat([pen1, output2.hidden_states[-2]], dim=-1)
        pooled = output2.text_embeds
        del text2, output2, pen1
        trace = {"context": context.cpu().numpy(), "pooled": pooled.cpu().numpy()}
        unet = UNet2DConditionModel.from_pretrained(root, subfolder="unet", variant="fp16", torch_dtype=torch.float32, local_files_only=True).to(device)
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        scheduler.set_timesteps(STEPS)
        sample = torch.from_numpy(source_latent()).to(device)
        trace["initial"] = sample.cpu().numpy()
        time_ids = torch.tensor([[SIZE, SIZE, 0, 0, SIZE, SIZE]] * 2, dtype=torch.float32, device=device)
        for index, timestep in enumerate(scheduler.timesteps):
            timestep = int(timestep)
            model_input = torch.cat([sample, sample], dim=0)
            time = torch.full((2,), timestep, dtype=torch.float32, device=device)
            noise = unet(model_input, time, encoder_hidden_states=context, added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
            uncond, cond = noise.chunk(2, dim=0)
            guided = uncond + SCALE * (cond - uncond)
            trace[f"noise{index}"] = noise.cpu().numpy()
            sample = scheduler.step(guided, timestep, sample, eta=0.0).prev_sample
            trace[f"sample{index}"] = sample.cpu().numpy()
        del unet
        vae = AutoencoderKL.from_pretrained(root, subfolder="vae", variant="fp16", torch_dtype=torch.float32, local_files_only=True).to(device)
        trace["decoded"] = vae.decode(sample / vae.config.scaling_factor).sample.cpu().numpy()
    return trace


def jittor_run(root: str) -> dict[str, np.ndarray]:
    import jittor as jt
    from masactrl.sd15_jittor import SD15Scheduler
    from sdxl.infer import SDXLPipeline, prompt_embed

    jt.flags.use_cuda = 1
    model = SDXLPipeline.from_pretrained(root)
    with jt.no_grad():
        negative, neg_pool = prompt_embed(model.parts, root, NEGATIVE)
        positive, pos_pool = prompt_embed(model.parts, root, PROMPT)
        context = jt.concat([negative, positive], dim=0)
        pooled = jt.concat([neg_pool, pos_pool], dim=0)
        trace = {"context": context.numpy(), "pooled": pooled.numpy()}
        scheduler = SD15Scheduler(steps_offset=0, set_alpha_to_one=False)
        scheduler.set_timesteps(STEPS)
        sample = jt.array(source_latent())
        trace["initial"] = sample.numpy()
        time_ids = jt.array(np.array([[SIZE, SIZE, 0, 0, SIZE, SIZE]] * 2, dtype=np.float32))
        for index, timestep in enumerate(scheduler.timesteps_np):
            model_input = jt.concat([sample, sample], dim=0)
            time = jt.array(np.full((2,), timestep, dtype=np.float32))
            noise = model.unet(model_input, time, context, pooled, time_ids)
            uncond, cond = noise[:1], noise[1:]
            guided = uncond + SCALE * (cond - uncond)
            trace[f"noise{index}"] = noise.numpy()
            sample = scheduler.step(guided, int(timestep), sample)
            trace[f"sample{index}"] = sample.numpy()
        trace["decoded"] = model.parts.decode_latent(sample).numpy()
    return trace


def cross_run(root: str, folder: Path) -> np.ndarray:
    import jittor as jt
    from masactrl.sd15_jittor import SafeTensorStore, SD15Scheduler
    from sdxl import SDXLUNet

    jt.flags.use_cuda = 1
    jt.cudnn.set_max_workspace_ratio(0.02)
    model = SDXLUNet()
    model.load_from_store(SafeTensorStore(str(Path(root) / "unet" / "diffusion_pytorch_model.fp16.safetensors")))
    with np.load(folder / "torch.npz") as source, jt.no_grad():
        scheduler = SD15Scheduler(steps_offset=0, set_alpha_to_one=False)
        scheduler.set_timesteps(STEPS)
        sample = jt.array(source["sample0"])
        model_input = jt.concat([sample, sample], dim=0)
        time = jt.array(np.full((2,), scheduler.timesteps_np[1], dtype=np.float32))
        time_ids = jt.array(np.array([[SIZE, SIZE, 0, 0, SIZE, SIZE]] * 2, dtype=np.float32))
        noise = model(model_input, time, jt.array(source["context"]), jt.array(source["pooled"]), time_ids)
        return noise.numpy()


def compare(folder: Path) -> bool:
    result = {}
    with np.load(folder / "torch.npz") as source, np.load(folder / "jittor.npz") as target:
        for key, limit in LIMITS.items():
            left = source[key].astype(np.float64)
            right = target[key].astype(np.float64)
            valid = left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
            if valid:
                delta = left - right
                rel = float(np.linalg.norm(delta) / max(np.linalg.norm(left), 1e-12))
                absolute = float(np.max(np.abs(delta)))
            else:
                rel, absolute = float("inf"), float("inf")
            result[key] = {"rel_rmse": rel, "max_abs": absolute, "limit": limit, "passed": rel <= limit}
    if (folder / "cross.npz").exists():
        with np.load(folder / "torch.npz") as source, np.load(folder / "cross.npz") as cross:
            delta = source["noise1"].astype(np.float64) - cross["noise1"].astype(np.float64)
            result["cross_noise1"] = {
                "rel_rmse": float(np.linalg.norm(delta) / np.linalg.norm(source["noise1"])),
                "max_abs": float(np.abs(delta).max()),
                "limit": LIMITS["noise1"],
            }
            result["cross_noise1"]["passed"] = result["cross_noise1"]["rel_rmse"] <= LIMITS["noise1"]
    strict = all(result[key]["passed"] for key in LIMITS)
    first = next((key for key in LIMITS if not result[key]["passed"]), None)
    report = {
        "passed": strict,
        "output_passed": result["sample1"]["passed"] and result["decoded"]["passed"],
        "first_divergence": first,
        "results": result,
    }
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report["passed"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("torch", "jittor", "cross", "compare"))
    parser.add_argument("--model_dir", default="/data/L202500448/maskfactory/models/stable-diffusion-xl-base-1.0")
    parser.add_argument("--output_dir", default="/data/L202500448/maskfactory/logs/parity_sdxl_ddim")
    args = parser.parse_args()
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    if args.backend == "compare":
        raise SystemExit(0 if compare(folder) else 1)
    if args.backend == "cross":
        np.savez(folder / "cross.npz", noise1=cross_run(args.model_dir, folder))
        return
    run = torch_run if args.backend == "torch" else jittor_run
    np.savez(folder / f"{args.backend}.npz", **run(args.model_dir))


if __name__ == "__main__":
    main()
