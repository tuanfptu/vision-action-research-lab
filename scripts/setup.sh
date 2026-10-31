#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/data/hf"
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e '.[dev,research]'
.venv/bin/pip check
.venv/bin/va-lab doctor
