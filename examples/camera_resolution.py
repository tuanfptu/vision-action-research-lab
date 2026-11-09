"""Exploratory validation-only camera/resolution controls after the main study."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from va_lab.datasets.core import file_hash, read_arrays
from va_lab.evaluation.research import freeze_json, measure_checkpoint, summarize
from va_lab.training.engine import train


def run(args):
    study, out = Path(args.study), Path(args.output)
    plan = json.loads((study / "protocol.json").read_text())
    selection = json.loads((study / "selection.json").read_text())
    # Equality of numeric trajectories preserves the paired episode comparison.
    a, b = read_arrays(args.low), read_arrays(args.high)
    for key in ["state", "action", "episode", "frame", "timestamp"]:
        if not np.array_equal(a[key], b[key]):
            raise ValueError("Camera caches have different numeric trajectories")
    high_meta = json.loads(Path(args.high).with_suffix(".manifest.json").read_text())
    low_meta = json.loads(Path(args.low).with_suffix(".manifest.json").read_text())
    if low_meta["revision"] != high_meta["revision"]:
        raise ValueError("Camera cache revision mismatch")
    del a, b
    cfg = plan["design"]
    cases = [
        ("two32", args.low, ["ceiling", "wrist_right"], 32),
        ("two64", args.high, ["ceiling", "wrist_right"], 64),
        ("five64", args.high, high_meta["cameras"], 64),
    ]
    out.mkdir(parents=True, exist_ok=False)
    freeze_json(
        out / "protocol.json",
        {
            "scope": "exploratory validation only; no final-test evaluation or model reselection",
            "selection_sha256": file_hash(study / "selection.json"),
            "runner_sha256": file_hash(__file__),
            "cases": [
                {"name": n, "data_sha256": file_hash(d), "cameras": v, "resolution": r}
                for n, d, v, r in cases
            ],
            "family": selection["selected_family"],
            "seeds": cfg["seeds"],
            "epochs": 3,
            "batch_size": 64,
            "gradient_accumulation": 1,
            "rationale": "fresh matched 3-epoch runs; batch64 for every case, including official ACT BatchNorm; main 12-epoch scores not reused as controls",
        },
    )
    rows, runs = [], []
    for name, data, cameras, size in cases:
        for seed in cfg["seeds"]:
            c = dict(
                cfg["training"],
                model=selection["selected_family"],
                seed=seed,
                epochs=3,
                batch_size=64,
                gradient_accumulation=1,
                cameras=cameras,
                image_size=size,
            )
            target = out / name / str(seed)
            train(c, data, str(target))
            measured = measure_checkpoint(str(target / "best.pt"), data, baselines=False)
            rows.extend([dict(r, model=name) for r in measured])
            runs.append(
                {
                    "case": name,
                    "seed": seed,
                    "checkpoint_sha256": file_hash(target / "best.pt"),
                    "cuda": str(torch.cuda.get_device_name()),
                }
            )
    freeze_json(
        out / "summary.json",
        {
            "summary": summarize(
                rows,
                [x[0] for x in cases],
                cfg["seeds"],
                sorted(plan["splits"]["validation"]),
                reference="two32",
            ),
            "records": rows,
            "runs": runs,
            "scope": "validation only; three fresh seeds per case, matched3epochs; no test scores",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--low", default="data/openarm.npz")
    p.add_argument("--high", default="data/openarm_fiveviews64.npz")
    run(p.parse_args())
