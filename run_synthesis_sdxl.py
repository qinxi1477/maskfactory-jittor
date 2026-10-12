import os
import jittor as jt

from masactrl.masactrl_utils import AttentionBase
from masactrl.masactrl_utils import register_attention_editor
from masactrl.masactrl import MutualSelfAttentionControl
from masactrl.jittor_compat import enable_cuda, set_seed
from sdxl.infer import SDXLPipeline, save_image

# model_path = "stabilityai/stable-diffusion-xl-base-1.0"
model_path = os.path.join(
    os.environ.get("MASKFACTORY_MODELS", "/data/L202500448/maskfactory/models"),
    "stable-diffusion-xl-base-1.0",
)
# model_path = "Linaqruf/animagine-xl"
enable_cuda()
model = SDXLPipeline.from_pretrained(model_path)


def consistent_synthesis():
    seed = 42
    set_seed(seed)

    out_dir_ori = os.path.join(os.environ.get("MASKFACTORY_ROOT", "/data/L202500448/maskfactory"), "results", "sdxl")
    os.makedirs(out_dir_ori, exist_ok=True)

    prompts = [
        "A portrait of an old man, facing camera, best quality",
        "A portrait of an old man, facing camera, smiling, best quality",
    ]

    # inference the synthesized image with MasaCtrl
    # TODO: note that the hyper paramerter of MasaCtrl for SDXL may be not optimal
    STEP = 4
    LAYER_LIST = [44, 54, 64]  # run the synthesis with MasaCtrl at three different layer configs

    # initialize the noise map
    start_code = jt.randn([1, 4, 128, 128])
    # start_code = None
    start_code = start_code.repeat(len(prompts), 1, 1, 1)

    # inference the synthesized image without MasaCtrl
    editor = AttentionBase()
    register_attention_editor(model, editor)
    with jt.no_grad():
        image_ori = model(prompts, latents=start_code, guidance_scale=7.5, height=1024, width=1024)

    for LAYER in LAYER_LIST:
        # hijack the attention module
        editor = MutualSelfAttentionControl(STEP, LAYER, model_type="SDXL")
        register_attention_editor(model, editor)

        # inference the synthesized image
        with jt.no_grad():
            image_masactrl = model(prompts, latents=start_code, guidance_scale=7.5, height=1024, width=1024)

        sample_count = len(os.listdir(out_dir_ori))
        out_dir = os.path.join(out_dir_ori, f"sample_{sample_count}")
        os.makedirs(out_dir, exist_ok=True)
        save_image(image_ori[0:1], os.path.join(out_dir, f"source_step{STEP}_layer{LAYER}.png"))
        save_image(image_ori[1:2], os.path.join(out_dir, f"without_step{STEP}_layer{LAYER}.png"))
        save_image(image_masactrl[-1:], os.path.join(out_dir, f"masactrl_step{STEP}_layer{LAYER}.png"))
        with open(os.path.join(out_dir, f"prompts.txt"), "w") as f:
            for p in prompts:
                f.write(p + "\n")
            f.write(f"seed: {seed}\n")
        print("Syntheiszed images are saved in", out_dir)


if __name__ == "__main__":
    consistent_synthesis()
