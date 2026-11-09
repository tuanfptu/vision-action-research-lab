"""GPU/CPU training with locked episode splits and an untouched explicit test path."""

import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from va_lab.datasets.core import (
    WindowBatches,
    Windows,
    episode_split,
    file_hash,
    read_arrays,
    statistics,
)
from va_lab.models.policies import build, masked_mse
from va_lab.utils.runtime import seed_all, write_json


def to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module, loader: DataLoader, device: str, steps: int, stats: dict
) -> dict:
    model.eval()
    error, count, times = 0.0, 0, []
    smooth, pairs = 0.0, 0
    raw_error = None
    video_error, video_count = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = model.sample(batch, steps)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        valid = batch["valid"][..., None]
        dim = pred.shape[-1]
        weight = int(batch["valid"].sum()) * dim
        error += float(masked_mse(pred, batch["action"], batch["valid"])) * weight
        count += weight
        per_joint = (
            (((pred - batch["action"]) * stats["action_std"].to(device)).square() * valid)
            .sum((0, 1))
            .cpu()
        )
        raw_error = per_joint if raw_error is None else raw_error + per_joint
        if pred.shape[1] > 1:
            v = batch["valid"][:, 1:] & batch["valid"][:, :-1]
            smooth += float(((pred[:, 1:] - pred[:, :-1]).square() * v[..., None]).sum())
            pairs += int(v.sum()) * dim
        if hasattr(model, "forward_video"):
            _, frames = model.forward_video(batch)
            b, k, v, c, h, w = batch["future"].shape
            target = F.interpolate(batch["future"].reshape(b * k * v, c, h, w), (16, 16)).view_as(
                frames
            )
            n = int(batch["future_valid"].sum()) * v * c * 16 * 16
            video_error += float(masked_mse(frames, target, batch["future_valid"])) * n
            video_count += n
    result = {
        "normalized_action_mse": error / max(count, 1),
        "joint_rmse_rad": (raw_error / max(count // dim, 1)).sqrt().tolist(),
        "normalized_chunk_delta_mse": smooth / max(pairs, 1),
        "batch_latency_ms_mean": float(np.mean(times)),
        "batch_latency_ms_p95": float(np.percentile(times, 95)),
        "evaluated_action_scalars": count,
        "sampling_steps": steps,
        "device": device,
        "scope": "offline joint prediction; not manipulation success",
    }
    if video_count:
        result["future_frame_mse"] = video_error / video_count
        result["future_frame_psnr_db"] = -10 * math.log10(max(video_error / video_count, 1e-12))
    return result


def prepare(cfg: dict, data: str) -> tuple:
    a = read_arrays(data)
    provenance = json.loads(Path(data).with_suffix(".manifest.json").read_text())
    if provenance.get("sha256") != file_hash(data):
        raise ValueError("Cache hash differs from manifest")
    available = provenance["cameras"]
    selected = cfg.get("cameras", available)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(c not in available for c in selected)
    ):
        raise ValueError(f"Camera selection {selected} not available in cache {available}")
    cfg = dict(
        cfg,
        state_dim=a["state"].shape[1],
        action_dim=a["action"].shape[1],
        cameras=selected,
        image_size=cfg.get("image_size", a["image"].shape[-1]),
    )
    indices = [available.index(c) for c in selected]
    if cfg["image_size"] != a["image"].shape[-1]:
        raise ValueError("Config image_size must match exported cache; re-export explicitly")
    return a, cfg, provenance, indices


def resolve_splits(a: dict, cfg: dict) -> tuple:
    tr, va, te = episode_split(a["episode"], cfg.get("split_seed", 7))
    if cfg.get("split_file"):
        split = json.loads(Path(cfg["split_file"]).read_text())
        tr, va, te = (np.asarray(split[k], dtype=int) for k in ["train", "validation", "test"])
    sets = [set(x.tolist()) for x in [tr, va, te]]
    if any(not x for x in sets) or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Empty or overlapping episode splits")
    if set.union(*sets) != set(np.unique(a["episode"]).tolist()):
        raise ValueError("Split must cover exactly the exported episodes")
    return tr, va, te


def train(
    cfg: dict, data: str, output: str, init: str | None = None, transfer: str = "full"
) -> dict:
    seed_all(cfg["seed"])
    device = cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU REQUIRED by config but CUDA is unavailable; no CPU fallback")
    a, cfg, provenance, cameras = prepare(cfg, data)
    tr, va, te = resolve_splits(a, cfg)
    stats = statistics(a, tr)
    fraction = cfg.get("train_fraction", 1.0)
    if not 0 < fraction <= 1:
        raise ValueError("train_fraction must be in (0,1]")
    used_tr = tr[: max(1, int(len(tr) * fraction))]
    # Fit statistics on the actual training budget, never validation/test frames.
    stats = statistics(a, used_tr)

    def windows(ids):
        return Windows(a, ids, cfg["chunk"], stats, cameras, cfg.get("history", 2))

    loader = DataLoader(
        windows(used_tr),
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=cfg["model"] == "official_act",
    )
    validation = DataLoader(windows(va), batch_size=cfg["batch_size"])
    if cfg.get("preload_device", False):
        loader = WindowBatches(
            windows(used_tr),
            cfg["batch_size"],
            device,
            shuffle=True,
            drop_last=cfg["model"] == "official_act",
        )
        validation = WindowBatches(windows(va), cfg["batch_size"], device, pool=loader.pool)
    if not len(loader):
        raise ValueError("Training budget smaller than batch_size")
    model = build(cfg)
    if init:
        if cfg["model"] not in ["bc", "state"]:
            raise ValueError("Controlled transfer currently supports BC encoders")
        prior = torch.load(init, map_location="cpu", weights_only=True)
        prior_cfg = prior["config"]
        for key in ["state_dim", "action_dim", "cameras", "dim"]:
            if prior_cfg[key] != cfg[key]:
                raise ValueError(f"Transfer architecture mismatch: {key}")
        if prior["data_sha256"] == file_hash(data):
            raise ValueError("Source/target data identity: not independent transfer")
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
    accumulation = cfg.get("gradient_accumulation", 1)
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg)
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": device,
        "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
        "cuda_runtime": torch.version.cuda,
        "preload_device": cfg.get("preload_device", False),
        "effective_batch_size": cfg["batch_size"] * cfg.get("gradient_accumulation", 1),
    }
    write_json(out / "runtime.json", runtime)
    write_json(
        out / "split.json",
        {
            "train": tr.tolist(),
            "training_budget": used_tr.tolist(),
            "validation": va.tolist(),
            "test": te.tolist(),
        },
    )
    best = float("inf")
    data_sha = file_hash(data)
    for epoch in range(cfg["epochs"]):
        epoch_start = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        for batch_idx, batch in enumerate(loader):
            batch = to_device(batch, device)
            loss = model.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            # The final partial accumulation group gets its own denominator.
            group = min(accumulation, len(loader) - (batch_idx // accumulation) * accumulation)
            (loss / group).backward()
            if (batch_idx + 1) % accumulation == 0 or batch_idx + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(batch["state"])
            n += len(batch["state"])
        rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if device.startswith("cuda") else None
        torch.manual_seed(cfg["seed"] + 1000)
        metrics = evaluate_model(model, validation, device, cfg.get("sampling_steps", 10), stats)
        torch.random.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        record = {
            "epoch": epoch,
            "train_loss": total / n,
            "epoch_wall_seconds": time.perf_counter() - epoch_start,
            "optimizer_steps": math.ceil(len(loader) / accumulation),
            **metrics,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20
            if device.startswith("cuda")
            else 0.0,
        }
        with (out / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        score = metrics["normalized_action_mse"] + metrics.get("future_frame_mse", 0.0)
        if score < best:
            best = score
            torch.save(
                {
                    "schema_version": 2,
                    "model": model.state_dict(),
                    "config": cfg,
                    "stats": stats,
                    "train_ids": tr.tolist(),
                    "val_ids": va.tolist(),
                    "test_ids": te.tolist(),
                    "camera_indices": cameras,
                    "data_sha256": data_sha,
                    "provenance": provenance,
                    "epoch": epoch,
                    "transfer": transfer,
                    "init_checkpoint": init,
                    "init_sha256": file_hash(init) if init else None,
                    "runtime": runtime,
                },
                out / "best.pt",
            )
    return evaluate_checkpoint(
        str(out / "best.pt"), data, str(out / "evaluation.json"), device=device
    )


def load_checkpoint(path: str, device: str = "cpu") -> tuple:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("schema_version") != 2:
        raise ValueError("Legacy checkpoint preserved but incompatible; retrain for OpenArm")
    model = build(dict(ckpt["config"], device=device))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


def evaluate_checkpoint(
    path: str, data: str, output: str, split: str = "validation", device: str | None = None
) -> dict:
    if split not in ["validation", "test"]:
        raise ValueError("Select validation or explicit final test")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    device = device or ckpt["config"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU required for evaluation; no CPU fallback")
    model, ckpt = load_checkpoint(path, device)
    if file_hash(data) != ckpt["data_sha256"]:
        raise ValueError("Dataset hash differs from checkpoint provenance")
    cfg = ckpt["config"]
    seed_all(cfg["seed"] + 1000)
    ids = ckpt["test_ids"] if split == "test" else ckpt["val_ids"]
    ds = Windows(
        read_arrays(data),
        ids,
        cfg["chunk"],
        ckpt["stats"],
        ckpt["camera_indices"],
        cfg.get("history", 2),
    )
    metrics = evaluate_model(
        model,
        WindowBatches(ds, cfg["batch_size"], device)
        if cfg.get("preload_device", False)
        else DataLoader(ds, batch_size=cfg["batch_size"]),
        device,
        cfg.get("sampling_steps", 10),
        ckpt["stats"],
    )
    metrics.update(checkpoint=path, epoch=ckpt["epoch"], split=split)
    write_json(output, metrics)
    return metrics
