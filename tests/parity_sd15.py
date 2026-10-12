"""PyTorch/Jittor component and one-step parity for Stable Diffusion 1.5."""

import argparse
import json
import os

import numpy as np


ORDER = ["text", "encode", "noise", "step", "decode", "image", "sample", "result", "inversion"]
TRACE = ["time", "input", "down0", "down1", "down2", "down3", "mid", "up0", "up1", "up2", "up3", "output"]
LIMITS = {
    "text": 2e-3,
    "encode": 3e-3,
    "noise": 1e-2,
    "step": 2e-2,
    "decode": 1e-2,
    "image": 2e-2,
    "sample": 5e-2,
    "result": 5e-2,
    "inversion": 5e-2,
}
TRACE_LIMIT = 2e-3


def make_inputs(path):
    rng = np.random.default_rng(20260915)
    data = {
        "ids": rng.integers(0, 1000, size=(1, 77), dtype=np.int32),
        "image": rng.normal(0, 0.3, size=(1, 3, 64, 64)).clip(-1, 1).astype("float32"),
        "latent": rng.normal(0, 1, size=(1, 4, 8, 8)).astype("float32"),
    }
    np.savez(path, **data)
    return data


def load_inputs(path):
    if not os.path.isfile(path):
        return make_inputs(path)
    with np.load(path) as handle:
        return {key: handle[key] for key in handle.files}


def torch_run(model_dir, inputs):
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel

    print("[torch] loading SD1.5", flush=True)
    device = torch.device("cuda")
    output = {}
    with torch.inference_mode():
        text_model = CLIPTextModel.from_pretrained(
            model_dir, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float32, local_files_only=True
        ).to(device)
        ids = torch.from_numpy(inputs["ids"]).to(device)
        text = text_model(ids).last_hidden_state
        output["text"] = text.cpu().numpy()
        del text_model

        unet = UNet2DConditionModel.from_pretrained(
            model_dir, subfolder="unet", variant="fp16", torch_dtype=torch.float32, local_files_only=True
        ).to(device)
        trace = {}

        def save(name):
            def hook(_, __, value):
                value = value[0] if isinstance(value, tuple) else value
                trace[name] = value.detach().cpu().numpy()
            return hook

        modules = [unet.time_embedding, unet.conv_in, *unet.down_blocks, unet.mid_block, *unet.up_blocks, unet.conv_out]
        hooks = [module.register_forward_hook(save(name)) for module, name in zip(modules, TRACE)]
        latent = torch.from_numpy(inputs["latent"]).to(device)
        print("[torch] tracing UNet", flush=True)
        noise = unet(latent, 751, encoder_hidden_states=text).sample
        for hook in hooks:
            hook.remove()
        output.update({"trace_" + key: value for key, value in trace.items()})
        output["noise"] = noise.cpu().numpy()
        scheduler = DDIMScheduler.from_pretrained(model_dir, subfolder="scheduler", local_files_only=True)
        scheduler.set_timesteps(4, device=device)
        step = scheduler.step(noise, 751, latent, eta=0.0).prev_sample
        output["step"] = step.cpu().numpy()

        sample = latent.clone()
        for sample_timestep in scheduler.timesteps:
            sample_noise = unet(sample, sample_timestep, encoder_hidden_states=text).sample
            sample = scheduler.step(sample_noise, sample_timestep, sample, eta=0.0).prev_sample
        output["sample"] = sample.cpu().numpy()
        inverted = latent.clone()
        ratio = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        for raw_timestep in reversed(scheduler.timesteps):
            timestep = int(raw_timestep)
            inverse_noise = unet(inverted, timestep, encoder_hidden_states=text).sample
            previous = timestep - ratio
            alpha = scheduler.alphas_cumprod[previous] if previous >= 0 else scheduler.final_alpha_cumprod
            alpha_next = scheduler.alphas_cumprod[timestep]
            predicted = (inverted - (1 - alpha) ** 0.5 * inverse_noise) / alpha ** 0.5
            inverted = alpha_next ** 0.5 * predicted + (1 - alpha_next) ** 0.5 * inverse_noise
        output["inversion"] = inverted.cpu().numpy()
        del unet

        vae = AutoencoderKL.from_pretrained(
            model_dir, subfolder="vae", variant="fp16", torch_dtype=torch.float32, local_files_only=True
        ).to(device)
        image = torch.from_numpy(inputs["image"]).to(device)
        encoded = vae.encode(image).latent_dist.mean * 0.18215
        decoded = vae.decode(latent / 0.18215).sample
        final_image = vae.decode(step / 0.18215).sample.clamp(-1, 1)
        output["encode"] = encoded.cpu().numpy()
        output["decode"] = decoded.cpu().numpy()
        output["image"] = final_image.cpu().numpy()
        output["result"] = vae.decode(sample / 0.18215).sample.clamp(-1, 1).cpu().numpy()
    print("[torch] complete", flush=True)
    return output


def jittor_run(model_dir, inputs):
    import jittor as jt

    from masactrl.sd15_jittor import StableDiffusion15Jittor

    print("[jittor] loading SD1.5", flush=True)
    jt.flags.use_cuda = 1
    model = StableDiffusion15Jittor.from_pretrained(model_dir)
    ids = jt.array(inputs["ids"]).int32()
    image = jt.array(inputs["image"])
    latent = jt.array(inputs["latent"])
    print("[jittor] tracing text and UNet", flush=True)
    text = model.text_encoder(ids)
    trace = {}
    noise = model.predict_noise(latent, 751, text, trace=trace)
    model.scheduler.set_timesteps(4)
    step = model.scheduler.step(noise, 751, latent)

    sample = latent
    for sample_timestep in model.scheduler.timesteps_np:
        sample_noise = model.predict_noise(sample, int(sample_timestep), text)
        sample = model.scheduler.step(sample_noise, int(sample_timestep), sample)
    inverted = latent
    for timestep in reversed(model.scheduler.timesteps_np):
        inverse_noise = model.predict_noise(inverted, int(timestep), text)
        inverted = model.scheduler.next_step(inverse_noise, int(timestep), inverted)
    print("[jittor] tracing VAE", flush=True)
    output = {
        "text": text.numpy(),
        "encode": model.encode_image(image).numpy(),
        "noise": noise.numpy(),
        "step": step.numpy(),
        "decode": model.decode_latent(latent).numpy(),
        "image": jt.clamp(model.decode_latent(step), -1, 1).numpy(),
        "sample": sample.numpy(),
        "inversion": inverted.numpy(),
        "result": jt.clamp(model.decode_latent(sample), -1, 1).numpy(),
    }
    output.update({"trace_" + key: value.numpy() for key, value in trace.items()})
    print("[jittor] complete", flush=True)
    return output


def metrics(left, right):
    left = left.astype("float64").reshape(-1)
    right = right.astype("float64").reshape(-1)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return {
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "rel_rmse": float("inf"),
            "cosine": 0.0,
        }
    diff = np.abs(left - right)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    cosine = float(np.dot(left, right) / denom) if denom else 1.0
    left_norm = np.linalg.norm(left)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_rmse": float(np.linalg.norm(left - right) / left_norm) if left_norm else 0.0,
        "cosine": cosine,
    }


def compare(torch_path, jittor_path, report_path):
    with np.load(torch_path) as torch_data, np.load(jittor_path) as jittor_data:
        results = {}
        first = None
        for key in ORDER:
            result = metrics(torch_data[key], jittor_data[key])
            metric = "rel_rmse" if key == "encode" else "max_abs"
            result["metric"] = metric
            result["limit"] = LIMITS[key]
            result["passed"] = result[metric] <= LIMITS[key]
            results[key] = result
            if first is None and not result["passed"]:
                first = key
        trace = {}
        first_trace = None
        for key in TRACE:
            result = metrics(torch_data["trace_" + key], jittor_data["trace_" + key])
            result["limit"] = TRACE_LIMIT
            result["passed"] = result["rel_rmse"] <= TRACE_LIMIT
            trace[key] = result
            if first_trace is None and not result["passed"]:
                first_trace = key
    report = {
        "passed": first is None,
        "first_divergence": first,
        "first_trace": first_trace,
        "results": results,
        "trace": trace,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["torch", "jittor", "compare"])
    parser.add_argument("--model_dir", default="/data/L202500448/maskfactory/models/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", default="/data/L202500448/maskfactory/logs/parity_sd15")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    input_path = os.path.join(args.output_dir, "inputs.npz")
    inputs = load_inputs(input_path)
    if args.backend == "torch":
        np.savez(os.path.join(args.output_dir, "torch.npz"), **torch_run(args.model_dir, inputs))
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
