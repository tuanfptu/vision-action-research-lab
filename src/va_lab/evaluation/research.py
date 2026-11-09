"""Episode-level research measurements; windows are never treated as IID trials."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from va_lab.datasets.core import WindowBatches, Windows, file_hash, read_arrays
from va_lab.training.engine import load_checkpoint
from va_lab.utils.runtime import seed_all


def freeze_json(path, value):
    """Write an auditable record once; refuse silent replacement of a test result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def crossed_interval(values, draws=10000, seed=2026):
    """Percentile bootstrap of a [training seed, episode] crossed design.

    Resample seed and episode axes independently; reuse those axes for paired
    differences. This is descriptive uncertainty, not a small-sample guarantee.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all() or draws < 1:
        raise ValueError("Finite nonempty [seed,episode] matrix and positive draws required")
    rng = np.random.default_rng(seed)
    si = rng.integers(x.shape[0], size=(draws, x.shape[0]))
    ei = rng.integers(x.shape[1], size=(draws, x.shape[1]))
    samples = x[si[:, :, None], ei[:, None, :]].mean((1, 2))
    return {
        "mean": float(x.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "training_seeds": x.shape[0],
        "episodes": x.shape[1],
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "unit": "rad^2",
        "method": "crossed seed/episode percentile bootstrap; descriptive small-sample interval",
    }


def matrix(records, model, seeds, episodes):
    lookup = {}
    for r in records:
        if r["model"] != model:
            continue
        key = (r["seed"], r["episode"])
        if key in lookup:
            raise ValueError("Duplicate seed/episode measurement")
        lookup[key] = r["action_mse_rad2"]
    if set(lookup) != {(s, e) for s in seeds for e in episodes}:
        raise ValueError("Missing or extra seed/episode measurement")
    return np.array([[lookup[s, e] for e in episodes] for s in seeds])


def summarize(records, models, seeds, episodes, reference="bc"):
    arrays = {m: matrix(records, m, seeds, episodes) for m in models}
    result = {}
    for name, values in arrays.items():
        result[name] = crossed_interval(values)
        result[name]["sqrt_macro_mse_rad"] = float(np.sqrt(values.mean()))
        if reference in arrays:
            result[name]["paired_difference_vs_" + reference] = crossed_interval(
                values - arrays[reference]
            )
    return result


def select_family(summary, eligible):
    if not eligible or any(m not in summary for m in eligible):
        raise ValueError("Missing eligible model")
    return min(eligible, key=lambda m: (summary[m]["mean"], m))


@torch.no_grad()
def measure_checkpoint(checkpoint, data, split="validation", baselines=True, steps=None):
    if split not in ["validation", "test"]:
        raise ValueError("Unknown split")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    device = saved["config"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA required by research protocol")
    if file_hash(data) != saved["data_sha256"]:
        raise ValueError("Dataset hash changed")
    model, ckpt = load_checkpoint(checkpoint, device)
    a, cfg, stats = read_arrays(data), ckpt["config"], ckpt["stats"]
    if baselines:
        meta = ckpt["provenance"]
        if not meta.get("state_names") or meta["state_names"] != meta.get("action_names"):
            raise ValueError("Persistence requires identical named state/action coordinates")
    ids = ckpt["val_ids"] if split == "validation" else ckpt["test_ids"]
    steps = cfg.get("sampling_steps", 10) if steps is None else steps
    records, pool = [], None
    std, mean = (stats["action_" + k].to(device) for k in ["std", "mean"])
    state_std, state_mean = (stats["state_" + k].to(device) for k in ["std", "mean"])
    checkpoint_sha = file_hash(checkpoint)
    for eid in sorted(ids):
        ds = Windows(a, [eid], cfg["chunk"], stats, ckpt["camera_indices"], cfg.get("history", 2))
        batches = WindowBatches(ds, cfg["batch_size"], device, pool=pool)
        pool = batches.pool
        # Warm-up is excluded from timings and does not advance the scored RNG stream.
        model.sample(next(iter(batches)), steps)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        seed_all(cfg["seed"] + 1000 + eid)
        names = [cfg["model"]] + (["persistence", "train_mean"] if baselines else [])
        errors = {m: torch.zeros(cfg["chunk"], cfg["action_dim"], device=device) for m in names}
        counts = torch.zeros(cfg["chunk"], device=device)
        video_error = torch.zeros(cfg["chunk"], device=device)
        persist_video_error = torch.zeros_like(video_error)
        video_count = torch.zeros_like(video_error)
        times = []
        for b in batches:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = model.sample(b, steps)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
            predictions = {cfg["model"]: prediction * std + mean}
            if baselines:
                q = b["state"] * state_std + state_mean
                predictions.update(
                    persistence=q[:, None].expand_as(prediction),
                    train_mean=mean[None, None].expand_as(prediction),
                )
            target = b["action"] * std + mean
            valid = b["valid"][..., None]
            for name, pred in predictions.items():
                if not torch.isfinite(pred).all():
                    raise FloatingPointError("Nonfinite research prediction")
                errors[name] += ((pred - target).square() * valid).sum(0)
            counts += b["valid"].sum(0)
            if hasattr(model, "forward_video"):
                _, frames = model.forward_video(b)
                nb, k, v, c, h, w = b["future"].shape
                truth = F.interpolate(b["future"].reshape(nb * k * v, c, h, w), (16, 16)).view_as(
                    frames
                )
                current = F.interpolate(b["image"].reshape(nb * v, c, h, w), (16, 16)).view(
                    nb, 1, v, c, 16, 16
                )
                fv = b["future_valid"]
                video_error += ((frames - truth).square().mean((2, 3, 4, 5)) * fv).sum(0)
                persist_video_error += ((current - truth).square().mean((2, 3, 4, 5)) * fv).sum(0)
                video_count += fv.sum(0)
        n = counts.sum().item()
        for name in names:
            sq = errors[name]
            record = {
                "model": name,
                "seed": cfg["seed"],
                "episode": eid,
                "split": split,
                "action_mse_rad2": float(sq.sum() / (n * cfg["action_dim"])),
                "joint_rmse_rad": (sq.sum(0) / n).sqrt().tolist(),
                "horizon_mse_rad2": (sq.mean(1) / counts.clamp_min(1)).tolist(),
                "valid_targets_per_horizon": counts.int().tolist(),
                "checkpoint_sha256": checkpoint_sha,
                "data_sha256": ckpt["data_sha256"],
                "sampling_steps": steps,
                "device": device,
                "scope": "offline action prediction; not manipulation success",
            }
            if name == cfg["model"]:
                record.update(
                    batch_latency_ms_mean=float(np.mean(times)),
                    batch_latency_ms_p95=float(np.percentile(times, 95)),
                    batch_size=cfg["batch_size"],
                    warmup_batches=1,
                )
                if video_count.sum() > 0:
                    record.update(
                        future_frame_mse=float(video_error.sum() / video_count.sum()),
                        persistence_frame_mse=float(persist_video_error.sum() / video_count.sum()),
                        horizon_frame_mse=(video_error / video_count.clamp_min(1)).tolist(),
                        horizon_persistence_frame_mse=(
                            persist_video_error / video_count.clamp_min(1)
                        ).tolist(),
                        video_scope="direct +1..K predictions at 16x16; not autoregressive drift",
                    )
            records.append(record)
    return records
