# Migration Status

This document records the Jittor migration state and the scope of the archived validation evidence.

## Scope

The migration keeps the original MaskFactory repository structure and ports the main PyTorch execution paths to Jittor:

- SD1.5 / MasaCtrl inference
- ControlNet
- Multi-ControlNet
- Zero123 novel-view inference
- SDXL synthesis
- Structure and parity checks

## Verified

| Area | Status | Evidence |
| --- | --- | --- |
| Structure audit | Passed | 152 source files checked; no missing files; no missing top-level symbols |
| Python compile | Passed | `python -m compileall -q .` |
| SD1.5 parity | Passed under the recorded tolerances | [sd15.json](validation/sd15.json) |
| Zero123 parity | Passed under the recorded tolerances | [zero123.json](validation/zero123.json) |
| ControlNet parity | Passed under the recorded tolerances | [controlnet.json](validation/controlnet.json) |
| Multi-ControlNet smoke | Passed | `logs/multi_final.png` |
| SDXL 1024 synthesis | Passed | `results/sdxl/sample_0..2` |

Publication checks: the released tree retains all 151 source files after excluding editor swap files, with no missing top-level symbols. All 123 Python files pass syntax parsing; all four archived JSON reports parse successfully. The original server audit counted 152 source files including an editor swap file. Model implementation files are unchanged from the retrieved Jittor snapshot; documentation, ignore rules and the structural-audit allowlist were adjusted for publication.

## Known Boundaries

- The upstream `train.py` uses a placeholder dataset, so full real-data training is not claimed.
- `run_8_node.sh` has been path-adjusted, but true 8-GPU execution was not validated on the current H100 MIG environment.
- The [SDXL report](validation/sdxl.json) has `output_passed=true` and `passed=false`. The first failed intermediate is `noise1`: relative RMSE 0.020902 versus the 0.003 limit. Final decoded relative RMSE is 0.008736 versus the 0.02 limit. Final-output acceptance does not imply strict intermediate acceptance.
- These JSON files are copied from existing server reports, without changing their values. They are historical measurements, not fresh GPU reruns performed for publication.
- Original Zero123 auxiliary code is retained; Jittor validation applies to the migrated inference entry and tested components, not every upstream auxiliary workflow.

## Last Validated Paths

On the validation server:

```text
/data/L202500448/maskfactory/repo/MaskFactory
/data/L202500448/maskfactory/repo/MaskFactory-jittor
/data/L202500448/maskfactory/logs
/data/L202500448/maskfactory/results/sdxl
```

## Reference Environment

Jittor inference dependencies are pinned in `requirements.txt`. Reference comparisons additionally use PyTorch, Diffusers and OmegaConf. Weight conversion is an offline PyTorch step; converted Zero123 inference runs in Jittor. Upstream license notices remain in the vendored Zero123 tree.

The original validation server reports PyTorch 2.6.0+cu124, Diffusers 0.39.0, OmegaConf 2.3.1 and Transformers 5.13.0. These are source/reference dependencies, separate from the Jittor runtime requirements.

## Reproduce on a New Machine

All commands below run from the repository root. For separate source and target environments, replace `python` with the corresponding interpreter in each step.

```bash
# Zero123: repeat for torch, jittor and compare.
python -m tests.parity_zero123 torch \
  --checkpoint models/zero123/105000.ckpt \
  --config zero123/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml \
  --model_dir models/zero123/jittor_ema --output_dir tests/output/zero123

# ControlNet: repeat for torch, jittor and compare.
python -m tests.parity_control torch \
  --model models/sd-controlnet-canny --work tests/output/control

# SDXL: run torch, jittor and cross, followed by compare.
python -m tests.parity_sdxl_full torch \
  --model_dir models/stable-diffusion-xl-base-1.0 --output_dir tests/output/sdxl

# Structural comparison requires the original PyTorch source tree.
python tests/audit_structure.py /path/to/original/MaskFactory .
```

The structure audit checks source-file presence and top-level symbol presence. It does not establish semantic equivalence or full training reproduction.
