# Vision–Action Research Lab

An independent, research-oriented Vision–Action learning laboratory covering Behavioral
Cloning, ACT, Diffusion/Flow Policies, Action Expert Pretraining, Video–Action World Models,
and Hierarchical Robot Control.

Built from scratch for learning through executable experiments. No personal robotics
repository, dataset, environment, checkpoint, simulator, or robot-specific integration is
required. No pretrained weights are downloaded automatically. Original project code is MIT.
Public dataset and upstream library licenses remain their own.

## Start here

Validated target: Python 3.10, PyTorch 2.7.1 / torchvision 0.22.1, LeRobot 0.3.3,
MuJoCo 3.3.2. The older stable LeRobot release deliberately supports Ubuntu 22.04's
Python 3.10 and the pre-2.8 PyTorch line. Latest LeRobot documentation describes a newer
API; use the versioned references in [docs/references.md](docs/references.md).

```bash
bash scripts/setup.sh
source .venv/bin/activate
unset PYTHONPATH  # only this shell; prevents ROS Python packages entering this environment
va-lab doctor
va-lab generate --output data/synthetic.npz
va-lab inspect --data data/synthetic.npz
va-lab train --data data/synthetic.npz --output runs/bc --model bc
va-lab train --data data/synthetic.npz --output runs/state --model state
va-lab train --data data/synthetic.npz --output runs/act --model act
va-lab evaluate --checkpoint runs/act/best.pt --data data/synthetic.npz
```

Synthetic curved reaching is an offline teaching task, **not PushT**. The consistent public
comparison benchmark is PushT image, with absolute agent-position targets in simulator
pixel coordinates. Start with six episodes to check the pipeline, then increase the data
budget and train for longer before studying success rates.

```bash
va-lab download --episodes 6 --output data/pusht.npz
va-lab inspect --data data/pusht.npz --output runs/pusht.png
va-lab train --data data/pusht.npz --output runs/pusht_bc --model bc
va-lab train --data data/pusht.npz --output runs/pusht_act --model act
va-lab rollout --checkpoint runs/pusht_act/best.pt --horizon 1 --ensemble
va-lab verify-lerobot
va-lab train --data data/pusht.npz --output runs/official_act --model official_act
```

The public parquet export intentionally handles the versioned v2.1 image dataset without
silently migrating it. The separate official LeRobot ACT smoke path and trainable adapter verify the pinned
API. Educational ACT uses a small pooled CNN and simplified conditional memory; it is
not architecture-equivalent to official ACT. Dataset manifests record the exact Hub SHA.
Images are resized to 32×32 to fit development hardware; report this when comparing results.

## Curriculum and implementation scope

| Lab | Executable implementation | Limits |
|---|---|---|
| [00](labs/lab_00_setup/README.md) | Fresh venv, dependency checks, seeds, JSONL logs, CPU CI | GPU validation depends on driver |
| [01](labs/lab_01_va_fundamentals/README.md) | State-only/vision BC, training, best checkpoint, offline and PushT evaluation | Short runs are smoke experiments |
| [02](labs/lab_02_act/README.md) | CVAE Transformer, padding mask, KL, chunks, temporal ensemble, official API check | Educational architecture |
| [03](labs/lab_03_act_ablation/README.md) | Automated chunk/latent/KL and rollout horizon/ensemble sweeps; reports | PARTIAL: history/encoder/data-budget sweeps and multi-seed study remain |
| [04](labs/lab_04_diffusion_flow/README.md) | Noise prediction, DDIM sampling, rectified-flow velocity and Euler sampling | Small MLP action generator |
| [05](labs/lab_05_apt/README.md) | Source pretraining; random/frozen/partial/full encoder transfer | APT-inspired; no official model reproduction |
| [06](labs/lab_06_lingbot_va/README.md) | Low-resolution future frames + actions; visualization | PARTIAL: no latent dual-stream diffusion or long autoregressive evaluation |
| [07](labs/lab_07_helix02/README.md) | Learned VA proposals, 100 Hz bounded controller, timeout, MuJoCo reacher | PARTIAL: original two-slide model, no Panda integration; not Helix reproduction |

Each lab includes a Vietnamese tutorial. See [validation](docs/validation.md) for executed
checks and measurements; [limitations](docs/limitations.md) for precise outstanding work.
No short-run ranking is claimed as a research conclusion.

## Engineering and experiments

`src/va_lab/` contains datasets, models, training, evaluation, simulation, controllers and
utilities. `configs/` holds CPU, low-VRAM GPU, cloud, model, dataset and evaluation settings.
`runs/` and `data/` are ignored; never commit downloaded data or weights. Experiments create
a fresh output directory, config, exact episode split, per-epoch metrics, best checkpoint,
training-only normalization statistics and dataset hash. Evaluation rejects a changed dataset.
The held-out split is **validation**, not an untouched final test set; model selection uses it.
Use a separate final episode set before reporting publication-quality generalization results.

```bash
python scripts/ablate.py --data data/pusht.npz --output runs/ablation --rollouts
va-lab train --data data/synthetic.npz --output runs/flow --model flow
va-lab train --data data/synthetic.npz --output runs/diffusion --model diffusion
va-lab train --data data/synthetic.npz --output runs/video --model video
va-lab video --checkpoint runs/video/best.pt --data data/synthetic.npz
va-lab record --output data/reacher.npz
va-lab hierarchy --checkpoint runs/bc/best.pt
python scripts/report.py runs
make test lint
```

CPU is the verified starting point. `configs/training/low_vram.yaml` uses batch 8 and hidden
width 64; `cloud.yaml` raises the budget. These GPU presets are estimates, not measured
VRAM guarantees. For GTX 1660 Ti use FP32 and a supported PyTorch 2.7.1 CUDA wheel such as
cu118 **only after checking the installed driver**; never modify drivers through setup.
The checked-in setup installs CPU wheels. See [environment](docs/environment.md).

## Research integrity

Offline metrics measure normalized action error, chunk variation and synchronized inference
latency. PushT rollouts separately record environment success, reward, action variation and
latency. Training loss values across CVAE, diffusion and flow objectives are not comparable.
Batch latency is not single-action latency. Report seeds, data size, image resolution, action
schema, hardware, horizons and sampling steps. No large foundation model download is needed.

[Official papers, code and licensing](docs/references.md) · [Vietnamese lab index](labs/README.md)
