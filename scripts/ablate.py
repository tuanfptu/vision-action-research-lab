"""OpenArm paired ablations; validation/test episodes remain fixed for every job."""

import argparse
import itertools
from pathlib import Path

import yaml

from va_lab.simulation.benchmark import rollout
from va_lab.training.engine import train
from va_lab.utils.runtime import load_config, write_json

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/openarm.npz")
p.add_argument("--output", required=True)
p.add_argument("--config", default="configs/training/low_vram.yaml")
p.add_argument("--grid", default="configs/models/act_ablation.yaml")
p.add_argument("--rollouts", action="store_true")
p.add_argument("--allow-unverified-scene", action="store_true")
a = p.parse_args()
grid = yaml.safe_load(Path(a.grid).read_text())
results = []
for chunk, latent, beta, cameras, budget in itertools.product(
    grid["chunks"], grid["latents"], grid["betas"], grid["camera_sets"], grid["train_fractions"]
):
    cfg = load_config(a.config)
    cfg.update(
        model="act", chunk=chunk, latent=latent, beta=beta, cameras=cameras, train_fraction=budget
    )
    out = Path(a.output) / f"k{chunk}_z{latent}_b{beta}_v{len(cameras)}_n{budget}"
    m = train(cfg, a.data, str(out))
    results.append({"config": cfg, "offline": m})
    if a.rollouts:
        for horizon, ensemble in itertools.product(
            grid["execution_horizons"], grid["temporal_ensemble"]
        ):
            if horizon > chunk or (ensemble and horizon != 1):
                continue
            rollout(
                str(out / "best.pt"),
                str(out / f"h{horizon}_e{ensemble}.json"),
                horizon=horizon,
                ensemble=ensemble,
                allow_unverified_scene=a.allow_unverified_scene,
            )
write_json(
    Path(a.output) / "ablation.json",
    {"results": results, "scope": "validation-only; final test untouched"},
)
