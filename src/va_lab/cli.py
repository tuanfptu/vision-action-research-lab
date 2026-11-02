"""OpenArm-only benchmark CLI; GPU is required by the default training config."""

import argparse
import json

from va_lab.utils.runtime import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download")
    p.add_argument("--output", default="data/openarm.npz")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--cameras", nargs="+", default=["ceiling", "wrist_right"])
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--root", default="data/openarm_raw")
    p = sub.add_parser("verify-lerobot")
    p.add_argument("--root", default="data/openarm_raw")
    p = sub.add_parser("inspect")
    p.add_argument("--data", default="data/openarm.npz")
    p.add_argument("--output", default="runs/openarm_observations.png")
    p = sub.add_parser("train")
    p.add_argument("--config", default="configs/training/low_vram.yaml")
    p.add_argument("--data", default="data/openarm.npz")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--model", choices=["bc", "state", "act", "official_act", "diffusion", "flow", "video"]
    )
    p.add_argument("--init")
    p.add_argument("--transfer", choices=["full", "partial", "frozen"], default="full")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/openarm.npz")
    p.add_argument("--output", default="runs/openarm_evaluation.json")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--device", choices=["cpu", "cuda"])
    for command in ["rollout", "hierarchy"]:
        p = sub.add_parser(command)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--output", default=f"runs/openarm_{command}.json")
        p.add_argument("--episodes", type=int, default=1)
        p.add_argument("--horizon", type=int, default=1)
        p.add_argument("--ensemble", action="store_true")
        p.add_argument("--steps", type=int, default=60)
        p.add_argument("--policy-hz", type=int, default=30)
        p.add_argument("--allow-unverified-scene", action="store_true")
    p = sub.add_parser("record")
    p.add_argument("--output", default="data/openarm_collected.npz")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--length", type=int, default=30)
    p = sub.add_parser("video")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/openarm.npz")
    p.add_argument("--output", default="runs/openarm_future.png")
    sub.add_parser("assets")
    p = sub.add_parser("sim-smoke")
    p.add_argument("--render", action="store_true")
    p = sub.add_parser("doctor")
    p.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    result = None
    c = args.command
    if c == "download":
        from va_lab.datasets.public import export_openarm

        result = export_openarm(args.output, args.episodes, args.cameras, args.size, args.root)
    elif c == "verify-lerobot":
        from va_lab.datasets.public import verify_lerobot

        result = verify_lerobot(args.root)
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

        result = evaluate_checkpoint(
            args.checkpoint, args.data, args.output, args.split, args.device
        )
    elif c in ["rollout", "hierarchy"]:
        from va_lab.simulation.benchmark import rollout

        result = rollout(
            args.checkpoint,
            args.output,
            args.episodes,
            args.horizon,
            args.ensemble,
            args.steps,
            args.allow_unverified_scene,
            args.policy_hz,
        )
    elif c == "record":
        from va_lab.simulation.benchmark import record

        result = record(args.output, args.episodes, args.seed, args.length)
    elif c == "video":
        from va_lab.evaluation.visualize import video

        video(args.checkpoint, args.data, args.output)
    elif c == "assets":
        from va_lab.simulation.assets import acquire_assets

        result = acquire_assets()
    elif c == "sim-smoke":
        from va_lab.simulation.benchmark import smoke

        result = smoke(args.render)
    elif c == "doctor":
        import platform

        import torch

        if args.require_gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA required: use .venv-gpu and permit NVIDIA device access")
        result = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        if torch.cuda.is_available():
            x = torch.randn(256, 256, device="cuda", requires_grad=True)
            y = (x @ x).square().mean()
            y.backward()
            torch.cuda.synchronize()
            result["gpu_forward_backward_finite"] = bool(torch.isfinite(x.grad).all())
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
