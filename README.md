# MaskFactory-Jittor

本仓库是 NeurIPS 2024 论文 **MaskFactory: Towards High-quality Synthetic Data Generation for Dichotomous Image Segmentation** 代码实现的 Jittor 版本，由社区迁移维护。

[原始代码](https://github.com/ydchen0806/MaskFactory) · [论文](https://arxiv.org/abs/2412.19080) · [English](README_en.md)

保留原项目入口和目录结构，提供 SD1.5、MasaCtrl、ControlNet、Multi-ControlNet、Zero123 和 SDXL 的 Jittor 推理实现。对齐脚本集中在 `tests/`；模型权重不随代码发布。

![MaskFactory 方法概览](figure/1.png)
![MaskFactory 原论文示例](figure/2.png)

上图来自原项目，展示原方法，并非本次迁移重新生成的实验结果。

## 安装

```bash
git clone https://github.com/qinxi1477/maskfactory-jittor.git
cd maskfactory-jittor
python -m pip install -r requirements.txt
python -m tests.verify_jittor_cuda
```

已有验证环境为 Python 3.10、Jittor 1.3.11、NumPy 1.26.4、CUDA 12.2 / cuDNN 8.9.7、H100 MIG。Jittor 首次运行会编译算子，需要可用的 C++ 编译器和 CUDA 环境。`env_maskfactory_jittor.sh` 是原验证服务器的环境脚本，其他机器请按自己的安装路径配置，不要直接套用其中的绝对路径。

## 准备权重

SD1.5、SDXL 和 ControlNet 使用原始 Diffusers 目录及 `safetensors` 权重；Zero123 需要先离线转换。权重请从对应模型的官方发布渠道获取，并遵循其使用条款。

```text
models/
├── stable-diffusion-v1-5/
├── stable-diffusion-xl-base-1.0/
├── sd-controlnet-canny/
├── sd-controlnet-seg/
└── zero123/
    ├── 105000.ckpt
    └── jittor_ema/
```

在仓库根目录设置模型与输出目录：

```bash
export MASKFACTORY_ROOT="$PWD"
export MASKFACTORY_MODELS="$PWD/models"
```

Zero123 离线转换使用 PyTorch、Diffusers 和 OmegaConf；转换后的 Jittor 推理不需要导入 PyTorch。用于数值对齐的源端环境也需要 PyTorch，相关版本见 [验证说明](docs/MIGRATION_STATUS.md)。

```bash
python tools/convert_zero123.py \
  models/zero123/105000.ckpt \
  zero123/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml \
  models/zero123/jittor_ema --ema
```

## 运行

### SD1.5 / MasaCtrl 图像编辑

```bash
python parrellel_diffusion.py \
  --world_size 1 \
  --model_path "$MASKFACTORY_MODELS/stable-diffusion-v1-5" \
  --data_list '/path/to/images/*.png' \
  --out_dir_base output_images
```

### SDXL 合成

```bash
python run_synthesis_sdxl.py
```

该入口使用真正的 SDXL 模型，按原脚本配置进行 1024×1024 合成，结果写入 `results/sdxl/`。

### Zero123 新视角生成

```bash
python -m zero123.zero123.jittor_infer input.png output_images/view.png \
  --model_dir "$MASKFACTORY_MODELS/zero123/jittor_ema" \
  --azimuth 30 --steps 50
```

### Multi-ControlNet 运行检查

```bash
python -m tests.smoke_multi \
  --base "$MASKFACTORY_MODELS/stable-diffusion-v1-5" \
  --canny "$MASKFACTORY_MODELS/sd-controlnet-canny" \
  --seg "$MASKFACTORY_MODELS/sd-controlnet-seg" \
  --output tests/output/multi.png
```

这个检查使用合成条件图和一步采样，验证双 ControlNet 路径可以执行，不代表数据集指标复现。

## 数值验证

源端和目标端分别执行，使用相同测试条件，然后比较保存的张量。下面以 SD1.5 为例；若使用独立环境，请分别用对应的 Python 执行 `torch` 和 `jittor` 步骤。

```bash
python -m tests.parity_sd15 torch \
  --model_dir "$MASKFACTORY_MODELS/stable-diffusion-v1-5" --output_dir tests/output/sd15
python -m tests.parity_sd15 jittor \
  --model_dir "$MASKFACTORY_MODELS/stable-diffusion-v1-5" --output_dir tests/output/sd15
python -m tests.parity_sd15 compare \
  --model_dir "$MASKFACTORY_MODELS/stable-diffusion-v1-5" --output_dir tests/output/sd15
```

| 路径 | 已保存的验证结果 |
| --- | --- |
| SD1.5 | 固定输入下的组件、采样及中间层对齐通过 |
| Zero123 | 固定输入下的组件、采样及中间层对齐通过 |
| ControlNet | 保存了条件残差的数值比较结果 |
| Multi-ControlNet | 双条件执行检查通过 |
| SDXL | 最终输出按报告容差通过；第二步 `noise1` 未通过严格中间检查 |

历史验证报告保存在 [docs/validation/](docs/validation/)，具体范围和复测命令见 [验证说明](docs/MIGRATION_STATUS.md)。这些报告来自迁移期间的服务器测试，本次发布未重新运行全部 GPU 推理。

## 代码结构

```text
masactrl/        SD1.5、MasaCtrl、ControlNet 和框架适配
sdxl/            SDXL 模型与推理
zero123/         Zero123 原始资源与 Jittor 推理入口
tests/           结构核对、执行检查和双端数值验证
tools/           离线权重转换
docs/            验证范围与已有报告
```

`zero123/` 保留了上游辅助代码，其中部分辅助训练和三维重建路径仍使用 PyTorch；已迁移的新视角推理入口为 `zero123.zero123.jittor_infer`。

## 当前范围

- 上游 `train.py` 的数据集仍为占位符；本仓库保留其训练结构，未复现完整真实数据训练。
- `run_8_node.sh` 保留多卡入口，尚未验证完整八卡运行。
- SDXL 严格中间层对齐尚未全部通过，不能把最终输出通过表述为全链路严格一致。

## 引用与来源

MaskFactory 的论文、方法和原始实现归原作者所有。本仓库提供框架迁移版本。感谢 [MaskFactory](https://github.com/ydchen0806/MaskFactory)、[MasaCtrl](https://github.com/TencentARC/MasaCtrl)、[Zero123](https://github.com/cvlab-columbia/zero123) 及 Stable Diffusion 项目。

```bibtex
@inproceedings{qianmaskfactory,
  title={MaskFactory: Towards High-quality Synthetic Data Generation for Dichotomous Image Segmentation},
  author={Qian, Haotian and Chen, Yinda and Lou, Shengtao and Khan, Fahad and Jin, Xiaogang and Fan, Deng-Ping},
  booktitle={NeurIPS},
  year={2024}
}
```

第三方代码保留原有版权和许可证，包括 `zero123/` 中的许可证文件。MaskFactory 上游副本未包含顶层许可证，因此本次发布不为整个组合仓库另行指定统一许可证；使用和再分发须按各上游授权执行。
