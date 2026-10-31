"""Controlled ACT ablation: real training jobs and optional closed-loop sweeps."""

import argparse
import itertools
from pathlib import Path

import yaml

from va_lab.simulation.benchmark import pusht_rollout
from va_lab.training.engine import train
from va_lab.utils.runtime import load_config, write_json

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--output", required=True)
p.add_argument("--rollouts", action="store_true")
a = p.parse_args()
grid = yaml.safe_load(Path("configs/models/act_ablation.yaml").read_text())
results = []
for chunk, latent, beta in itertools.product(grid["chunks"], grid["latents"], grid["betas"]):
    cfg = load_config("configs/training/cpu.yaml")
    cfg.update(model="act", chunk=chunk, latent=latent, beta=beta)
    out = Path(a.output) / f"chunk{chunk}_latent{latent}_beta{beta}"
    metrics = train(cfg, a.data, str(out))
    results.append({"config": cfg, "offline": metrics})
    if a.rollouts:
        for horizon, ensemble in itertools.product(
            grid["execution_horizons"], grid["temporal_ensemble"]
        ):
            if horizon > chunk or (ensemble and horizon != 1):
                continue
            pusht_rollout(
                str(out / "best.pt"),
                str(out / f"rollout_h{horizon}_e{ensemble}.json"),
                horizon=horizon,
                ensemble=ensemble,
            )
write_json(Path(a.output) / "ablation.json", {"results": results})
