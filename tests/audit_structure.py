import argparse
import ast
import difflib
from pathlib import Path


CHANGED = {
    "README.md",
    "infer.py",
    "masactrl/diffuser_utils.py",
    "masactrl/masactrl.py",
    "masactrl/masactrl_utils.py",
    "multicontrol_model_parrel.py",
    "parrellel_diffusion.py",
    "run_8_node.sh",
    "run_synthesis_sdxl.py",
    "train.py",
}

ADDED = {
    "README_en.md",
    "docs/validation/sd15.json",
    "docs/validation/zero123.json",
    "docs/validation/controlnet.json",
    "docs/validation/sdxl.json",
    ".gitignore",
    "docs/MIGRATION_STATUS.md",
    "env_jittor.sh",
    "env_maskfactory_jittor.sh",
    "requirements.txt",
    "masactrl/controlnet.py",
    "masactrl/jittor_compat.py",
    "masactrl/sd15_jittor.py",
    "sdxl/__init__.py",
    "sdxl/infer.py",
    "sdxl/model.py",
    "sdxl/unet.py",
    "tests/__init__.py",
    "tests/audit_structure.py",
    "tests/convert_sd15_smoke.py",
    "tests/parity_sd15.py",
    "tests/parity_control.py",
    "tests/parity_sdxl.py",
    "tests/parity_sdxl_full.py",
    "tests/parity_sdxl_unet.py",
    "tests/parity_zero123.py",
    "tests/run_parity.sh",
    "tests/smoke_jittor.py",
    "tests/smoke_multi.py",
    "tests/smoke_sd15_editor.py",
    "tests/smoke_sdxl_editor.py",
    "tests/verify_jittor_cuda.py",
    "tools/convert_zero123.py",
    "zero123/__init__.py",
    "zero123/zero123/__init__.py",
    "zero123/zero123/jittor_infer.py",
    "zero123/zero123/jittor_model.py",
}


def files(root):
    return {
        str(path.relative_to(root)): path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.suffix != ".swp"
        and not str(path.relative_to(root)).startswith("tests/output/")
    }


def symbols(path):
    tree = ast.parse(path.read_text(errors="ignore"))
    return {
        (type(node).__name__, node.name)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()

    source = files(args.source)
    target = files(args.target)
    missing = sorted(source.keys() - target.keys())
    added = sorted(target.keys() - source.keys())
    changed = sorted(
        name for name in source.keys() & target.keys()
        if source[name].read_bytes() != target[name].read_bytes()
    )

    unexpected_added = sorted(set(added) - ADDED)
    unexpected_changed = sorted(set(changed) - CHANGED)
    missing_symbols = {}
    for name in changed:
        if name.endswith(".py"):
            absent = sorted(symbols(source[name]) - symbols(target[name]))
            if absent:
                missing_symbols[name] = absent
    if missing or unexpected_added or unexpected_changed or missing_symbols:
        raise SystemExit(
            f"missing={missing}\n"
            f"unexpected_added={unexpected_added}\n"
            f"unexpected_changed={unexpected_changed}\n"
            f"missing_symbols={missing_symbols}"
        )

    print(f"source_files={len(source)}")
    print(f"target_files={len(target)}")
    print("missing=0")
    print("missing_symbols=0")
    for name in changed:
        try:
            left = source[name].read_text(errors="ignore").splitlines()
            right = target[name].read_text(errors="ignore").splitlines()
            ratio = difflib.SequenceMatcher(None, left, right).ratio()
            print(f"changed {name} similarity={ratio:.3f}")
        except OSError:
            print(f"changed {name}")


if __name__ == "__main__":
    main()
