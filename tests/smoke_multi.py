import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from masactrl.controlnet import MultiControl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--canny", required=True)
    parser.add_argument("--seg", required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/output/multi.png"))
    args = parser.parse_args()

    axis = np.linspace(0, 255, 64, dtype=np.uint8)
    grid = np.tile(axis, (64, 1))
    canny = Image.fromarray(np.stack([grid, grid, grid], axis=-1))
    seg = Image.fromarray(np.stack([grid.T, grid, 255 - grid], axis=-1))
    pipe = MultiControl.from_pretrained(args.base, [args.canny, args.seg])
    image = pipe(
        "a red cube on a white table",
        image=[canny, seg],
        negative_prompt="low quality",
        seed=0,
        num_inference_steps=1,
        width=64,
        height=64,
    ).images[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"multi_control_passed={args.output}")


if __name__ == "__main__":
    main()
