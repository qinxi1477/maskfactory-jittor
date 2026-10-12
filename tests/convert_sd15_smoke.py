import argparse
import os

from masactrl.jittor_compat import enable_cuda
from masactrl.sd15_jittor import StableDiffusion15Jittor, assert_legacy_backend_not_loaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sd15_dir",
        default="/data/L202500448/maskfactory/models/stable-diffusion-v1-5",
    )
    parser.add_argument(
        "--out",
        default="/data/L202500448/maskfactory/logs/reproduce_stage1/sd15_jittor_weight_smoke.txt",
    )
    args = parser.parse_args()

    enable_cuda()
    assert_legacy_backend_not_loaded()
    model = StableDiffusion15Jittor.from_pretrained(args.sd15_dir)
    stats = model.smoke_forward()
    assert_legacy_backend_not_loaded()

    lines = [
        "sd15_jittor_weight_smoke passed",
        f"unet.conv_in.out_shape={stats['unet.conv_in.out_shape']}",
        f"unet.time_embedding.out_shape={stats['unet.time_embedding.out_shape']}",
        f"unet.down_path.out_shapes={stats['unet.down_path.out_shapes']}",
        f"unet.mid_block.out_shape={stats['unet.mid_block.out_shape']}",
        f"unet.up_path.out_shapes={stats['unet.up_path.out_shapes']}",
        f"unet.output.out_shape={stats['unet.output.out_shape']}",
        f"vae.decoder.conv_out.out_shape={stats['vae.decoder.conv_out.out_shape']}",
        f"text.token_embedding.out_shape={stats['text.token_embedding.out_shape']}",
        f"text.token_embedding.table_shape={stats['text.token_embedding.table_shape']}",
        f"text.encoder.last_hidden_state_shape={stats['text.encoder.last_hidden_state_shape']}",
        f"vae.decoder.full.out_shape={stats['vae.decoder.full.out_shape']}",
        f"scheduler.timesteps={stats['scheduler.timesteps']}",
        f"scheduler.prev_latent_shape={stats['scheduler.prev_latent_shape']}",
        f"text.token_embedding.slice_mean={stats['text.token_embedding.slice_mean']:.8f}",
        f"text.encoder.last_hidden_state_mean={stats['text.encoder.last_hidden_state_mean']:.8f}",
        f"unet.conv_in.out_mean={stats['unet.conv_in.out_mean']:.8f}",
        f"unet.time_embedding.out_mean={stats['unet.time_embedding.out_mean']:.8f}",
        f"unet.down_path.out_means={[round(value, 8) for value in stats['unet.down_path.out_means']]}",
        f"unet.mid_block.out_mean={stats['unet.mid_block.out_mean']:.8f}",
        f"unet.up_path.out_means={[round(value, 8) for value in stats['unet.up_path.out_means']]}",
        f"unet.output.out_mean={stats['unet.output.out_mean']:.8f}",
        f"vae.decoder.conv_out.out_mean={stats['vae.decoder.conv_out.out_mean']:.8f}",
        f"vae.decoder.full.out_mean={stats['vae.decoder.full.out_mean']:.8f}",
        f"scheduler.prev_latent_mean={stats['scheduler.prev_latent_mean']:.8f}",
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
