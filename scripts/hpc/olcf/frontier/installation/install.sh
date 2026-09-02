#!/usr/bin/env bash
# Copyright (c) 2026, Oak Ridge National Laboratory
# SPDX-License-Identifier: BSD-3-Clause

# Canonical, single-entry installer for Frontier.  The ROCm-specific files in
# this directory are implementation profiles; users should invoke this script.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYDRAGNN_SRC="${HYDRAGNN_SRC:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
FRONTIER_ROCM_VERSION="${FRONTIER_ROCM_VERSION:-7.1}"
FAIRCHEM_CORE_VERSION="${FAIRCHEM_CORE_VERSION:-2.22.0}"

case "${FRONTIER_ROCM_VERSION}" in
  6.4) PROFILE="install-rocm64.sh" ;;
  7.1) PROFILE="install-rocm71.sh" ;;
  7.2) PROFILE="install-rocm72.sh" ;;
  7.13) PROFILE="install-rocm713.sh" ;;
  *)
    echo "Unsupported FRONTIER_ROCM_VERSION=${FRONTIER_ROCM_VERSION}." >&2
    echo "Supported values: 6.4, 7.1, 7.2, 7.13" >&2
    exit 2
    ;;
esac

INSTALL_ROOT="${INSTALL_ROOT:-${PWD}/HydraGNN-Installation-Frontier}"
VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/hydragnn_venv}"
export INSTALL_ROOT VENV_PATH HYDRAGNN_SRC

bash "${SCRIPT_DIR}/${PROFILE}"
source "${VENV_PATH}/bin/activate"

# Install HydraGNN before dependencies that are needed only by optional model
# backbones. --no-deps preserves the accelerator-specific PyTorch/PyG stack
# installed by the selected Frontier profile.
python -m pip install --no-deps -e "${HYDRAGNN_SRC}"
python -m pip install -r "${HYDRAGNN_SRC}/requirements-specific-models.txt"
python -m pip install "fairchem-core==${FAIRCHEM_CORE_VERSION}"

python - <<'PY'
import fairchem.core
import hydragnn
import torch

print("HydraGNN:", hydragnn.__file__)
print("FAIR-Chem:", fairchem.core.__file__)
print("PyTorch:", torch.__version__)
PY

echo "Frontier environment ready: ${VENV_PATH}"
echo "For gated UMA checkpoints, authenticate with huggingface-cli or set HF_TOKEN."
