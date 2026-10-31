"""Controlled APT-inspired transfer, never an official APT reproduction.
Source and target must be separate datasets; target split remains identical across arms.
"""

import argparse
from pathlib import Path

from va_lab.training.engine import train
from va_lab.utils.runtime import load_config, write_json

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--target", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
if Path(a.source).resolve() == Path(a.target).resolve():
    raise ValueError("Pretraining source and target must differ")
cfg = load_config("configs/training/cpu.yaml")
source = Path(a.output) / "pretrain"
train(cfg, a.source, str(source))
results = {"random": train(cfg, a.target, str(Path(a.output) / "random"))}
for arm in ["frozen", "partial", "full"]:
    results[arm] = train(cfg, a.target, str(Path(a.output) / arm), str(source / "best.pt"), arm)
write_json(Path(a.output) / "transfer.json", results)
