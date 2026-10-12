#!/usr/bin/env bash

# Unified PyTorch/Jittor environment for MaskFactory on qx-slai3.
# Runtime and Jittor caches stay under /data so every entry uses the same stack.

export MASKFACTORY_ROOT="/data/L202500448/maskfactory"
export MASKFACTORY_VENV="/data/L202500448/qx/miniconda3/envs/qx"
export MASKFACTORY_MODELS="${MASKFACTORY_ROOT}/models"
export JITTOR_HOME="/data/L202500448/qx/.cache/jittor_maskfactory"
export JITTOR_CUDA_HOME="${JITTOR_HOME}/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux"
export nvcc_path="${JITTOR_CUDA_HOME}/bin/nvcc"

# cutlass self-test can hang on this H100 image; disabling it keeps Jittor GPU ops usable.
export use_cutlass=0

export PATH="${MASKFACTORY_VENV}/bin:${JITTOR_CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${JITTOR_CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1

mkdir -p "${JITTOR_HOME}" "${MASKFACTORY_ROOT}/logs"

if [[ -f "${nvcc_path}" && ! -x "${nvcc_path}" ]]; then
    find "${JITTOR_CUDA_HOME}/bin" "${JITTOR_CUDA_HOME}/nvvm/bin" -type f -exec chmod u+x {} +
fi

if [[ ! -x "${nvcc_path}" ]]; then
    echo "Missing Jittor CUDA bundle: ${nvcc_path}" >&2
    echo "Run: JITTOR_HOME=${JITTOR_HOME} JTCUDA_VERSION=12.2 ${MASKFACTORY_VENV}/bin/python -m jittor_utils.install_cuda" >&2
    return 1 2>/dev/null || exit 1
fi
