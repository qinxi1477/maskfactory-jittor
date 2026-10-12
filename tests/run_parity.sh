#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/env_maskfactory_jittor.sh"

cd "${ROOT}"
python -m tests.parity_sd15 torch "$@"
python -m tests.parity_sd15 jittor "$@"
python -m tests.parity_sd15 compare "$@"
