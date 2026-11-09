"""Secondary GPU experiments: same-task transfer and independent-scene control.

Run after the registered main study. These experiments never select on final-test labels.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from va_lab.datasets.core import file_hash
from va_lab.evaluation.research import freeze_json, measure_checkpoint, summarize
from va_lab.simulation.benchmark import rollout


def run(args):
    study, out = Path(args.study), Path(args.output)
    if not (study / "completed.json").exists():
        raise ValueError("Complete the registered study before followups")
    plan = json.loads((study / "protocol.json").read_text())
    selection = json.loads((study / "selection.json").read_text())
    cfg = plan["design"]
    selected = selection["selected_family"]
    out.mkdir(parents=True, exist_ok=False)
    freeze_json(
        out / "protocol.json",
        {
            "main_protocol_sha256": file_hash(study / "protocol.json"),
            "selection_sha256": file_hash(study / "selection.json"),
            "runner_sha256": file_hash(__file__),
            "data_sha256": file_hash(args.data),
            "transfer": "same-task disjoint global-train pools; seeds7/17/27; random/frozen/partial/full; not cross-task",
            "control": {
                "checkpoint_seed": 7,
                "families": ["bc", selected],
                "cases": ["bc30", "selected30", "selected10"],
                "episodes_per_case": 10,
                "ticks_per_episode": 300,
                "reset_seeds": list(range(10000, 10010)),
                "scene": "independent official scene, NOT dataset-equivalent",
            },
            "test_use": "none; no further access to held-out action labels",
        },
    )
    records = []
    for seed in cfg["seeds"]:
        c = dict(cfg["training"], model="bc", seed=seed)
        config = out / f"transfer_seed{seed}.yaml"
        config.write_text(yaml.safe_dump(c))
        root = out / "transfer" / str(seed)
        subprocess.run(
            [
                sys.executable,
                "scripts/transfer.py",
                "--data",
                args.data,
                "--config",
                str(config),
                "--output",
                str(root),
            ],
            check=True,
        )
        for arm in ["random", "frozen", "partial", "full"]:
            rows = measure_checkpoint(
                str(root / arm / "best.pt"), str(root / "target.npz"), baselines=False
            )
            records.extend([dict(r, model=arm) for r in rows])
    ids = sorted({r["episode"] for r in records})
    freeze_json(
        out / "transfer_summary.json",
        {
            "summary": summarize(
                records,
                ["random", "frozen", "partial", "full"],
                cfg["seeds"],
                ids,
                reference="random",
            ),
            "records": records,
            "scope": "same-task initialization; one target validation episode, seed uncertainty only; normalization-shift confound retained",
        },
    )
    cases = []
    for name, family, hz in [
        ("bc30", "bc", 30),
        ("selected30", selected, 30),
        ("selected10", selected, 10),
    ]:
        checkpoint = study / family / "7" / "best.pt"
        result = rollout(
            str(checkpoint),
            str(out / f"{name}.json"),
            episodes=10,
            max_steps=300,
            policy_hz=hz,
            allow_unverified_scene=True,
        )
        # Existing rollout resets the clipping counter at each reset.
        # Label its reported count accurately instead of treating it as a trial total.
        result["command_clipped_values_last_episode"] = result.pop("command_clipped_values")
        n, rate = result["episodes"], result["success_rate"]
        z = 1.959963984540054
        den = 1 + z * z / n
        center = (rate + z * z / (2 * n)) / den
        half = z * ((rate * (1 - rate) / n + z * z / (4 * n * n)) ** 0.5) / den
        result.update(
            case=name,
            model=family,
            checkpoint_seed=7,
            checkpoint_sha256=file_hash(checkpoint),
            success_ci95_wilson=[max(0.0, center - half), min(1.0, center + half)],
        )
        cases.append(result)
    freeze_json(
        out / "independent_control_summary.json",
        {
            "results": cases,
            "scope": "10 independent reset seeds per case; fixed checkpoint seed7; independent scene, not original dataset task evaluation",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data", default="data/openarm.npz")
    run(p.parse_args())
