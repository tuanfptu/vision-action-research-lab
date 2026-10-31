"""Build a Markdown table only from recorded offline evaluation files."""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("--output", default="runs/report.md")
a = p.parse_args()
rows = ["| Experiment | Normalized MSE | Batch latency (ms) |", "|---|---:|---:|"]
for path in sorted(Path(a.root).rglob("evaluation.json")):
    m = json.loads(path.read_text())
    rows.append(
        f"| {path.parent} | {m['normalized_action_mse']:.6f} | {m['batch_latency_ms_mean']:.3f} |"
    )
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(
    "# Offline validation (not simulator success)\n\n" + "\n".join(rows) + "\n"
)
