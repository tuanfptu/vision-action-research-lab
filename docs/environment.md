# GPU environment and reproducibility

Use `.venv-gpu` for actual training. It is fresh and independent from the preserved CPU `.venv`.
Official wheels: torch 2.7.1+cu118 / torchvision 0.22.1+cu118, CUDA 11.8 runtime; Python 3.10.
The installed NVIDIA driver is not changed. `nvidia-smi` and CUDA device operations must run
with permission to access NVIDIA device files. The earlier sandbox-only failure did not prove
a driver failure. `va-lab doctor --require-gpu` executes a real CUDA forward/backward check.

```bash
bash scripts/setup_gpu.sh
source .venv-gpu/bin/activate
unset PYTHONPATH
export HF_HOME="$PWD/data/hf"
export MUJOCO_GL=egl
va-lab doctor --require-gpu
```

GTX 1660 Ti 6GB: default FP32, two 32×32 views, batch 8, accumulation 2, width 64, chunk 4.
No BF16 assumption and no automatic mixed precision. GPU memory figures must come from
executed runs; `peak_gpu_memory_mib` is allocated tensor memory, not full nvidia-smi process
memory. Low-VRAM preset is the default. Cloud increases batch/width/epochs explicitly.

CPU setup remains `bash scripts/setup.sh` and CPU tests use `make test PY=.venv/bin/python`.
CPU smoke config is explicit; GPU-configured runs never silently switch devices.
ROS PYTHONPATH is cleared only in the current process/shell, preserving the ROS installation.
No dependency on OpenArm ROS nodes, prior local code, pretrained weights or OS changes.

Exact tested GPU snapshot: requirements-gpu.lock.txt. For a clean Linux/Python 3.10 environment,
install the official CUDA torch pair first, then `pip install -r requirements-gpu.lock.txt`.
The lock has no local absolute source path, prior environment link or foundation model download.
