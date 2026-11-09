"""Run a locked GPU study, select on validation, then explicitly finalize test once."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from va_lab.datasets.core import file_hash
from va_lab.evaluation.research import freeze_json, measure_checkpoint, select_family, summarize
from va_lab.training.engine import prepare, resolve_splits, train


def source_hash():
    h = hashlib.sha256()
    for p in sorted(Path("src").rglob("*.py")) + sorted(Path("scripts").glob("*.py")):
        h.update(str(p).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def run(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    a, _, provenance, _ = prepare(cfg["training"], args.data)
    tr, va, te = resolve_splits(a, cfg["training"])
    plan = {
        "design": cfg,
        "data_sha256": file_hash(args.data),
        "dataset_revision": provenance.get("revision"),
        "source_sha256": source_hash(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "splits": {"train": tr.tolist(), "validation": va.tolist(), "test": te.tolist()},
        "primary_metric": "mean across seeds and episodes of raw action MSE rad^2; all joints and valid chunk targets",
        "checkpoint_selection": "existing trainer validation normalized action MSE + video MSE when applicable",
        "family_selection": "minimum validation primary metric across registered families; alphabetical tie break",
        "final_test": "all registered seeds of selected family and BC reference plus persistence/train-mean; no test tuning",
        "uncertainty": "10000 crossed seed/episode bootstrap draws, seed2026, paired differences; small-sample descriptive only",
        "scope": "single-task offline pilot, not a paper reproduction or closed-loop performance claim",
    }
    root = Path(args.output)
    if args.resume:
        saved = json.loads((root / "protocol.json").read_text())
        # A later publishing commit may differ; the executed source must not.
        saved.pop("base_commit")
        expected = dict(plan)
        expected.pop("base_commit")
        if saved != expected:
            raise ValueError("Resume protocol/data/source mismatch")
    else:
        root.mkdir(parents=True, exist_ok=False)
        freeze_json(root / "protocol.json", plan)
        freeze_json(
            root / "started.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "gpu": torch.cuda.get_device_name()
                if cfg["training"]["device"].startswith("cuda")
                else None,
            },
        )
    seeds, models = cfg["seeds"], cfg["models"]
    if len(seeds) != len(set(seeds)) or len(models) != len(set(models)):
        raise ValueError("Duplicate seeds/models")
    validation = []
    for name in models:
        for seed in seeds:
            c = dict(cfg["training"], model=name, seed=seed)
            c.update(cfg.get("model_overrides", {}).get(name, {}))
            out = root / name / str(seed)
            if out.exists():
                actual = json.loads((out / "config.json").read_text())
                if (
                    any(actual.get(k) != v for k, v in c.items())
                    or not (out / "evaluation.json").exists()
                ):
                    raise ValueError(
                        f"Incomplete/different run {out}; preserve it and use a fresh study directory"
                    )
            else:
                train(c, args.data, str(out))
            result = out / "episodes_validation.json"
            if not result.exists():
                freeze_json(
                    result,
                    measure_checkpoint(
                        str(out / "best.pt"), args.data, baselines=name == models[0]
                    ),
                )
            rows = json.loads(result.read_text())
            if any(r["checkpoint_sha256"] != file_hash(out / "best.pt") for r in rows):
                raise ValueError("Measured checkpoint changed")
            validation.extend(rows)
    all_models = models + ["persistence", "train_mean"]
    summary = summarize(validation, all_models, seeds, sorted(va.tolist()))
    selected = select_family(summary, models)
    selection = {
        "selected_family": selected,
        "validation_summary": summary,
        "protocol_sha256": file_hash(root / "protocol.json"),
        "validation_records_sha256": hashlib.sha256(
            json.dumps(validation, sort_keys=True).encode()
        ).hexdigest(),
        "selected_checkpoints": {
            str(s): file_hash(root / selected / str(s) / "best.pt") for s in seeds
        },
        "reference_checkpoints": {
            str(s): file_hash(root / "bc" / str(s) / "best.pt") for s in seeds
        },
    }
    selection_path = root / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError("Locked selection changed")
    else:
        freeze_json(selection_path, selection)
        freeze_json(root / "validation_records.json", validation)
    print(
        json.dumps({"selected_on_validation": selected, "finalize_test": args.finalize}), flush=True
    )
    if args.finalize:
        for name in dict.fromkeys([selected, "bc"]):
            for seed in seeds:
                out = root / name / str(seed)
                target = out / "episodes_test.json"
                if not target.exists():
                    freeze_json(
                        target,
                        {
                            "selection_sha256": file_hash(selection_path),
                            "records": measure_checkpoint(
                                str(out / "best.pt"),
                                args.data,
                                split="test",
                                baselines=name == selected,
                            ),
                        },
                    )
        records = []
        for name in dict.fromkeys([selected, "bc"]):
            for seed in seeds:
                result = json.loads((root / name / str(seed) / "episodes_test.json").read_text())
                if result["selection_sha256"] != file_hash(selection_path):
                    raise ValueError("Test selection lock changed")
                if any(
                    r["checkpoint_sha256"] != file_hash(root / name / str(seed) / "best.pt")
                    for r in result["records"]
                ):
                    raise ValueError("Final-test checkpoint changed")
                records.extend(result["records"])
        test_models = list(dict.fromkeys([selected, "bc", "persistence", "train_mean"]))
        test = {
            "selection_sha256": file_hash(selection_path),
            "summary": summarize(records, test_models, seeds, sorted(te.tolist())),
            "records": records,
        }
        target = root / "final_test.json"
        if target.exists():
            if json.loads(target.read_text()) != test:
                raise ValueError("Final test record changed")
        else:
            freeze_json(target, test)
        freeze_json(
            root / "completed.json", {"utc": datetime.now(timezone.utc).isoformat()}
        ) if not (root / "completed.json").exists() else None


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/research/openarm_gpu.yaml")
    p.add_argument("--data", default="data/openarm.npz")
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--finalize", action="store_true")
    run(p.parse_args())
