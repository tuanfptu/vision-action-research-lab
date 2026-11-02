"""OpenArm initial VA prior experiment; source and target episodes cannot overlap.
One task only: not evidence for cross-task transfer or sample efficiency.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from va_lab.datasets.core import episode_split, file_hash, manifest, read_arrays
from va_lab.training.engine import train
from va_lab.utils.runtime import load_config, write_json

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/openarm.npz")
p.add_argument("--output", required=True)
p.add_argument("--config", default="configs/training/low_vram.yaml")
a = p.parse_args()
arrays = read_arrays(a.data)
cfg = load_config(a.config)
# Split the global training pool only; global validation/test episodes never enter source.

tr, va, te = episode_split(arrays["episode"], cfg.get("split_seed", 7))
if len(tr) < 6:
    raise ValueError("Need >=6 training episodes for the source/target protocol")
source_ids, target_ids = tr[::2], tr[1::2]
root = Path(a.output)
root.mkdir(parents=True, exist_ok=False)
metadata = json.loads(Path(a.data).with_suffix(".manifest.json").read_text())
for name, ids in [("source", source_ids), ("target", target_ids)]:
    mask = np.isin(arrays["episode"], ids)
    path = root / f"{name}.npz"
    np.savez_compressed(path, **{k: v[mask] for k, v in arrays.items()})
    manifest(
        path,
        dict(
            metadata,
            source_episode_ids=ids.tolist(),
            episodes=len(ids),
            frames=int(mask.sum()),
            parent_sha256=file_hash(a.data),
            scope="same-task disjoint episodes; not cross-task transfer",
        ),
    )
write_json(
    root / "protocol.json",
    {
        "source_ids": source_ids.tolist(),
        "target_ids": target_ids.tolist(),
        "reserved_global_validation": va.tolist(),
        "untouched_global_test": te.tolist(),
    },
)
train(cfg, str(root / "source.npz"), str(root / "pretrain"))
results = {"random": train(cfg, str(root / "target.npz"), str(root / "random"))}
for arm in ["frozen", "partial", "full"]:
    results[arm] = train(
        cfg, str(root / "target.npz"), str(root / arm), str(root / "pretrain/best.pt"), arm
    )
write_json(root / "transfer.json", results)
