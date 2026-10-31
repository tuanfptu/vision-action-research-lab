"""Training, exact best-checkpoint validation, and held-out offline evaluation."""

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from va_lab.datasets.core import Windows, episode_split, read_arrays, statistics
from va_lab.models.policies import build, masked_mse
from va_lab.utils.runtime import seed_all, write_json


def to_device(batch: dict, device: str) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: str, steps: int) -> dict:
    model.eval()
    error, count, times = 0.0, 0, []
    smooth, pairs = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = model.sample(batch, steps)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        weight = int(batch["valid"].sum()) * 2
        error += float(masked_mse(pred, batch["action"], batch["valid"])) * weight
        count += weight
        if pred.shape[1] > 1:
            valid = batch["valid"][:, 1:] & batch["valid"][:, :-1]
            smooth += float(((pred[:, 1:] - pred[:, :-1]).square() * valid[..., None]).sum())
            pairs += int(valid.sum()) * 2
    return {
        "normalized_action_mse": error / max(count, 1),
        "normalized_chunk_delta_mse": smooth / max(pairs, 1),
        "batch_latency_ms_mean": sum(times) / max(len(times), 1),
        "evaluated_action_scalars": count,
        "sampling_steps": steps,
        "scope": "held-out episodes, offline; not closed-loop success",
    }


def train(
    cfg: dict, data: str, output: str, init: str | None = None, transfer: str = "full"
) -> dict:
    seed_all(cfg["seed"])
    device = cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use configs/training/cpu.yaml")
    a = read_arrays(data)
    train_ids, val_ids = episode_split(a["episode"], cfg["seed"])
    stats = statistics(a, train_ids)
    trainset = Windows(a, train_ids, cfg["chunk"], stats)
    valset = Windows(a, val_ids, cfg["chunk"], stats)
    loader = DataLoader(trainset, batch_size=cfg["batch_size"], shuffle=True)
    validation = DataLoader(valset, batch_size=cfg["batch_size"])
    model = build(cfg)
    if init:
        prior = torch.load(init, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(
            {
                k.removeprefix("encoder."): v
                for k, v in prior["model"].items()
                if k.startswith("encoder.")
            }
        )
        if transfer not in ["full", "partial", "frozen"]:
            raise ValueError("Unknown transfer strategy")
        if transfer != "full":
            for p in model.encoder.parameters():
                p.requires_grad_(False)
            if transfer == "partial":
                for p in model.encoder.fuse.parameters():
                    p.requires_grad_(True)
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg)
    import platform

    write_json(
        out / "runtime.json",
        {"python": platform.python_version(), "torch": str(torch.__version__), "device": device},
    )
    write_json(out / "split.json", {"train": train_ids.tolist(), "validation": val_ids.tolist()})
    best = float("inf")
    for epoch in range(cfg["epochs"]):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch["state"])
            n += len(batch["state"])
        # Reset the validation RNG so stochastic generators use comparable noise per epoch.
        torch.manual_seed(cfg["seed"] + 1000)
        metrics = evaluate_model(model, validation, device, cfg.get("sampling_steps", 10))
        record = {"epoch": epoch, "train_loss": total / n, **metrics}
        with (out / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if metrics["normalized_action_mse"] < best:
            best = metrics["normalized_action_mse"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "stats": stats,
                    "train_ids": train_ids.tolist(),
                    "val_ids": val_ids.tolist(),
                    "data_sha256": __import__("hashlib")
                    .sha256(Path(data).read_bytes())
                    .hexdigest(),
                    "epoch": epoch,
                    "provenance": json.loads(Path(data).with_suffix(".manifest.json").read_text())
                    if Path(data).with_suffix(".manifest.json").exists()
                    else {},
                    "transfer": transfer,
                    "init": init,
                },
                out / "best.pt",
            )
    return evaluate_checkpoint(str(out / "best.pt"), data, str(out / "evaluation.json"))


def load_checkpoint(path: str) -> tuple:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def evaluate_checkpoint(path: str, data: str, output: str) -> dict:
    import hashlib

    model, ckpt = load_checkpoint(path)
    if hashlib.sha256(Path(data).read_bytes()).hexdigest() != ckpt["data_sha256"]:
        raise ValueError("Dataset hash differs from checkpoint provenance")
    cfg = ckpt["config"]
    seed_all(cfg["seed"] + 1000)
    ds = Windows(read_arrays(data), ckpt["val_ids"], cfg["chunk"], ckpt["stats"])
    metrics = evaluate_model(
        model, DataLoader(ds, batch_size=cfg["batch_size"]), "cpu", cfg.get("sampling_steps", 10)
    )
    metrics.update(checkpoint=path, device="cpu", epoch=ckpt["epoch"])
    write_json(output, metrics)
    return metrics
