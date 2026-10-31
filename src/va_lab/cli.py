"""Explicit commands: no automatic large model or dataset downloads."""

import argparse
import json
from pathlib import Path

from va_lab.utils.runtime import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--output", default="data/synthetic.npz")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("download")
    p.add_argument("--output", default="data/pusht.npz")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--revision")
    p = sub.add_parser("inspect")
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="runs/observations.png")
    p = sub.add_parser("train")
    p.add_argument("--config", default="configs/training/cpu.yaml")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--model", choices=["bc", "state", "act", "official_act", "diffusion", "flow", "video"]
    )
    p.add_argument("--init")
    p.add_argument("--transfer", choices=["full", "partial", "frozen"], default="full")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="runs/evaluation.json")
    p = sub.add_parser("rollout")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="runs/rollout.json")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--ensemble", action="store_true")
    p = sub.add_parser("hierarchy")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="runs/hierarchy.json")
    p.add_argument("--policy-hz", type=int, default=10)
    p = sub.add_parser("video")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="runs/future.png")
    p = sub.add_parser("record")
    p.add_argument("--output", default="data/reacher.npz")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    sub.add_parser("doctor")
    sub.add_parser("verify-lerobot")
    args = parser.parse_args()
    c = args.command
    result = None
    if c == "generate":
        from va_lab.datasets.core import generate

        generate(args.output, args.episodes, seed=args.seed)
    elif c == "download":
        from va_lab.datasets.public import export_pusht

        export_pusht(args.output, args.episodes, args.revision)
    elif c == "inspect":
        from va_lab.evaluation.visualize import inspect

        result = inspect(args.data, args.output)
    elif c == "train":
        from va_lab.training.engine import train

        cfg = load_config(args.config)
        if args.model:
            cfg["model"] = args.model
        result = train(cfg, args.data, args.output, args.init, args.transfer)
    elif c == "evaluate":
        from va_lab.training.engine import evaluate_checkpoint

        result = evaluate_checkpoint(args.checkpoint, args.data, args.output)
    elif c == "rollout":
        from va_lab.simulation.benchmark import pusht_rollout

        result = pusht_rollout(
            args.checkpoint, args.output, args.episodes, args.horizon, args.ensemble
        )
    elif c == "hierarchy":
        from va_lab.simulation.benchmark import hierarchy

        result = hierarchy(args.checkpoint, args.output, policy_hz=args.policy_hz)
    elif c == "video":
        from va_lab.evaluation.visualize import video

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        video(args.checkpoint, args.data, args.output)
    elif c == "verify-lerobot":
        from va_lab.datasets.public import verify_lerobot

        result = verify_lerobot()
    elif c == "record":
        from va_lab.simulation.benchmark import record

        result = record(args.output, args.episodes, args.seed)
    elif c == "doctor":
        import platform

        import torch

        from va_lab.simulation.benchmark import smoke

        result = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "simulation": smoke(),
        }
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
