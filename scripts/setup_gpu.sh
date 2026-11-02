#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH
export HF_HOME="$PWD/data/hf"
nvidia-smi
python3.10 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install --upgrade pip
.venv-gpu/bin/pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
.venv-gpu/bin/pip install -e '.[dev,research]'
.venv-gpu/bin/pip check
.venv-gpu/bin/va-lab doctor --require-gpu
