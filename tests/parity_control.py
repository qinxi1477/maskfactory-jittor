import argparse
import json
from pathlib import Path

import numpy as np


def inputs(path: Path):
    if path.exists():
        return dict(np.load(path))
    rng = np.random.RandomState(17)
    values = {
        "latent": rng.standard_normal((2, 4, 8, 8)).astype("float32"),
        "text": rng.standard_normal((2, 77, 768)).astype("float32"),
        "image": rng.standard_normal((2, 3, 64, 64)).astype("float32"),
    }
    np.savez(path, **values)
    return values


def torch_run(model: str, work: Path):
    import torch
    from diffusers import ControlNetModel

    values = inputs(work / "inputs.npz")
    net = ControlNetModel.from_pretrained(model, torch_dtype=torch.float32, local_files_only=True).cuda().eval()
    with torch.no_grad():
        down, mid = net(
            torch.from_numpy(values["latent"]).cuda(),
            500,
            encoder_hidden_states=torch.from_numpy(values["text"]).cuda(),
            controlnet_cond=torch.from_numpy(values["image"]).cuda(),
            return_dict=False,
        )
    output = {f"down{index:02d}": value.float().cpu().numpy() for index, value in enumerate(down)}
    output["mid"] = mid.float().cpu().numpy()
    np.savez(work / "torch.npz", **output)


def jittor_run(model: str, work: Path):
    import jittor as jt
    from masactrl.controlnet import ControlNet

    jt.flags.use_cuda = 1
    values = inputs(work / "inputs.npz")
    net = ControlNet.from_pretrained(model)
    down, mid = net(
        jt.array(values["latent"]),
        500,
        jt.array(values["text"]),
        jt.array(values["image"]),
    )
    output = {f"down{index:02d}": value.numpy() for index, value in enumerate(down)}
    output["mid"] = mid.numpy()
    np.savez(work / "jittor.npz", **output)


def compare(work: Path):
    left = dict(np.load(work / "torch.npz"))
    right = dict(np.load(work / "jittor.npz"))
    report = {}
    passed = True
    for name in left:
        delta = np.abs(left[name] - right[name])
        relative = float(np.linalg.norm(delta) / max(np.linalg.norm(left[name]), 1e-12))
        cosine = float(
            np.dot(left[name].reshape(-1), right[name].reshape(-1))
            / max(np.linalg.norm(left[name]) * np.linalg.norm(right[name]), 1e-12)
        )
        report[name] = {"max": float(delta.max()), "relative": relative, "cosine": cosine}
        passed = passed and relative < 5e-3 and cosine > 0.9999
    report["passed"] = passed
    (work / "control.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["torch", "jittor", "compare"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--work", type=Path, default=Path("tests/output/control"))
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.mode == "torch":
        torch_run(args.model, args.work)
    elif args.mode == "jittor":
        jittor_run(args.model, args.work)
    else:
        compare(args.work)


if __name__ == "__main__":
    main()
