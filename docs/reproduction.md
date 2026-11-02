# Tái lập migration OpenArm trên GPU

```bash
cd ~/vision-action-research-lab
bash scripts/setup_gpu.sh
source .venv-gpu/bin/activate
unset PYTHONPATH
export HF_HOME="$PWD/data/hf"
export MUJOCO_GL=egl
va-lab doctor --require-gpu
va-lab verify-lerobot
va-lab download --episodes 30 --cameras ceiling wrist_right --size 32
va-lab inspect
va-lab train --model bc --output runs/my_openarm_bc
va-lab train --model state --output runs/my_openarm_state
va-lab train --model act --output runs/my_openarm_act
va-lab train --model official_act --output runs/my_openarm_official_act
va-lab train --model diffusion --output runs/my_openarm_diffusion
va-lab train --model flow --output runs/my_openarm_flow
va-lab train --model video --output runs/my_openarm_video
va-lab video --checkpoint runs/my_openarm_video/best.pt
python scripts/ablate.py --output runs/my_openarm_ablation --grid configs/models/act_smoke_ablation.yaml
python scripts/transfer.py --output runs/my_openarm_transfer
va-lab assets
va-lab sim-smoke --render
va-lab rollout --checkpoint runs/my_openarm_bc/best.pt --allow-unverified-scene
va-lab record --output data/my_openarm_collected.npz
VA_LAB_INTEGRATION=1 make test
make lint
```

Skip download if data/openarm.npz already exists. All exports and experiment directories
must be new; no silent overwrite. Standard training uses validation only; the untouched
final episode set is accessed only by explicit `evaluate --split test` after model selection.
GPU logs identify the actual device, CUDA runtime and peak allocated VRAM. CPU tests can run
separately with `make test PY=.venv/bin/python`; this is not a substitute for CUDA training.

All five original video views are acquired for each requested episode. To export five-view
or higher-resolution training caches, choose a new output file, all desired camera names and
--size. Match cameras/image_size in the training YAML. Named camera selection may reduce a
larger cache to a subset without changing episode splits or action coordinates.
