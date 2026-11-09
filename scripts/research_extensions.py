"""Validation-only ACT controls, sampler sensitivity and deterministic control faults."""

import argparse
import json
from pathlib import Path

import numpy as np

from va_lab.controllers.hierarchy import BoundedController
from va_lab.evaluation.research import freeze_json, measure_checkpoint, summarize
from va_lab.training.engine import train


def controller_faults():
    """Command-interface faults, not a hardware or IPC certification."""
    c = BoundedController(-np.ones(16), np.ones(16))
    q, target = np.zeros(16), np.ones(16) * 100
    cases = []
    for name, t, age in [
        ("fresh", target, 0),
        ("at_timeout", target, 0.25),
        ("stale", target, 0.251),
        ("nan_target", target * np.nan, 0),
        ("inf_target", target * np.inf, 0),
    ]:
        u = c.step(q, t, age, 0.001)
        hold = name in ["stale", "nan_target", "inf_target"]
        cases.append(
            {
                "case": name,
                "max_command_rad": float(np.abs(u).max()),
                "expected_hold": hold,
                "passed": bool(np.allclose(u, q) if hold else np.allclose(u, 0.002)),
            }
        )
    for name, pos, age, dt in [
        ("nan_clock", q, np.nan, 0.001),
        ("negative_dt", q, 0, -0.001),
        ("nan_state", q * np.nan, 0, 0.001),
        ("wrong_shape", np.zeros(15), 0, 0.001),
    ]:
        rejected = False
        try:
            c.step(pos, target, age, dt)
        except ValueError:
            rejected = True
        cases.append({"case": name, "passed": rejected})
    if not all(x["passed"] for x in cases):
        raise AssertionError("Controller fault check failed")
    return {
        "cases": cases,
        "scope": "deterministic controller interface; no real-time IPC or hardware certification",
    }


def run(args):
    root = Path(args.study)
    plan = json.loads((root / "protocol.json").read_text())
    cfg = plan["design"]
    seeds, ids = cfg["seeds"], sorted(plan["splits"]["validation"])
    rows = []
    for seed in seeds:
        base = json.loads((root / "act" / str(seed) / "episodes_validation.json").read_text())
        rows.extend(
            [
                dict(r, model="base", action_mse_rad2=r["horizon_mse_rad2"][0])
                for r in base
                if r["model"] == "act"
            ]
        )
    for name, changes in cfg.get("ablation_overrides", {}).items():
        for seed in seeds:
            c = dict(cfg["training"], model="act", seed=seed, **changes)
            out = root / "ablations" / name / str(seed)
            if not (out / "evaluation.json").exists():
                train(c, args.data, str(out))
            target = out / "episodes_validation.json"
            if not target.exists():
                freeze_json(
                    target, measure_checkpoint(str(out / "best.pt"), args.data, baselines=False)
                )
            raw = json.loads(target.read_text())
            rows.extend(
                [dict(r, model=name, action_mse_rad2=r["horizon_mse_rad2"][0]) for r in raw]
            )
    target = root / "ablations.json"
    if not target.exists():
        freeze_json(
            target,
            {
                "metric": "first-action raw MSE rad^2 for comparable K; validation only; exploratory paired CIs, no multiplicity correction",
                "summary": summarize(
                    rows, ["base"] + list(cfg["ablation_overrides"]), seeds, ids, reference="base"
                ),
                "records": rows,
            },
        )
    target = root / "sampler_sensitivity.json"
    if not target.exists():
        samplers = []
        for name in ["diffusion", "flow"]:
            for seed in seeds:
                for steps in [1, 5, 10, 20, 50]:
                    samplers.extend(
                        measure_checkpoint(
                            str(root / name / str(seed) / "best.pt"),
                            args.data,
                            baselines=False,
                            steps=steps,
                        )
                    )
        freeze_json(
            target,
            {
                "scope": "validation only; same checkpoints, no best-of-N selection; stochastic one draw per window per seed",
                "records": samplers,
            },
        )
    target = root / "controller_faults.json"
    if not target.exists():
        freeze_json(target, controller_faults())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--data", default="data/openarm.npz")
    run(p.parse_args())
