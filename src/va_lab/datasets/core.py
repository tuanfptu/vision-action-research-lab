"""Configurable multi-camera trajectories with episode-disjoint train/validation/test."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from va_lab.utils.runtime import write_json


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest(path: Path, metadata: dict) -> None:
    write_json(path.with_suffix(".manifest.json"), dict(metadata, sha256=file_hash(path)))


def generate(
    path: str,
    episodes: int = 8,
    length: int = 8,
    seed: int = 7,
    state_dim: int = 16,
    action_dim: int = 16,
    cameras: int = 2,
) -> None:
    """Small synthetic tensors for software tests only; not a manipulation benchmark."""
    rng = np.random.default_rng(seed)
    n = episodes * length
    if episodes < 3 or min(length, state_dim, action_dim, cameras) < 1:
        raise ValueError("Invalid test fixture dimensions")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image=rng.integers(0, 256, (n, cameras, 3, 32, 32), dtype=np.uint8),
        state=rng.normal(size=(n, state_dim)).astype(np.float32),
        action=rng.normal(size=(n, action_dim)).astype(np.float32),
        episode=np.repeat(np.arange(episodes), length),
        timestamp=np.tile(np.arange(length) / 30, episodes),
        frame=np.tile(np.arange(length), episodes),
    )
    manifest(
        path,
        {
            "source": "software-test fixture only",
            "seed": seed,
            "cameras": [f"test_{i}" for i in range(cameras)],
            "fps": 30,
        },
    )


def read_arrays(path: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        a = {
            key: data[key] for key in ["image", "state", "action", "episode", "timestamp", "frame"]
        }
    n = len(a["episode"])
    if n == 0 or any(len(v) != n for v in a.values()):
        raise ValueError("Empty or inconsistent dataset")
    if a["image"].ndim != 5 or a["image"].shape[2] != 3:
        raise ValueError("Images must be [N,V,3,H,W]; legacy caches must be re-exported")
    if any(a[k].ndim != 2 or a[k].shape[1] < 1 for k in ["state", "action"]):
        raise ValueError("State/action must be [N,D]")
    if not all(np.isfinite(v).all() for v in a.values()):
        raise ValueError("Nonfinite dataset value")
    upper = 255 if a["image"].dtype == np.uint8 else 1
    if a["image"].min() < 0 or a["image"].max() > upper:
        raise ValueError("Invalid image range")
    if a["episode"].dtype.kind not in "iu":
        raise ValueError("Episode IDs must be integers")
    for eid in np.unique(a["episode"]):
        idx = np.flatnonzero(a["episode"] == eid)
        if np.any(np.diff(idx) != 1) or not np.array_equal(a["frame"][idx], np.arange(len(idx))):
            raise ValueError("Episode frames must be contiguous and ordered from zero")
        if len(idx) > 1 and np.any(np.diff(a["timestamp"][idx]) <= 0):
            raise ValueError("Timestamps must increase inside each episode")
    return a


def episode_split(ids: np.ndarray, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(ids)
    if len(unique) < 3:
        raise ValueError("Need at least three episodes for train/validation/test")
    np.random.default_rng(seed).shuffle(unique)
    val_n = max(1, int(0.15 * len(unique)))
    test_n = max(1, int(0.15 * len(unique)))
    return unique[: -(val_n + test_n)], unique[-(val_n + test_n) : -test_n], unique[-test_n:]


class Windows(Dataset):
    def __init__(
        self,
        arrays: dict,
        episodes: np.ndarray | list,
        chunk: int,
        stats: dict,
        camera_indices: list[int] | None = None,
        history: int = 2,
    ):
        if chunk < 1 or history < 1 or len(episodes) == 0:
            raise ValueError("Positive chunk/history and nonempty episodes required")
        self.a, self.chunk, self.stats, self.history = arrays, chunk, stats, history
        self.camera_indices = (
            list(range(arrays["image"].shape[1])) if camera_indices is None else camera_indices
        )
        if not self.camera_indices:
            raise ValueError("At least one camera is required")
        if len(set(self.camera_indices)) != len(self.camera_indices):
            raise ValueError("Duplicate camera index")
        self.indices = np.flatnonzero(np.isin(arrays["episode"], episodes))
        self.starts, self.ends = {}, {}
        for eid in episodes:
            idx = np.flatnonzero(arrays["episode"] == eid)
            if not len(idx):
                raise ValueError(f"Unknown episode {eid}")
            for i in idx:
                self.starts[int(i)], self.ends[int(i)] = int(idx[0]), int(idx[-1])

    def __len__(self) -> int:
        return len(self.indices)

    def _images(self, idx: int | np.ndarray) -> torch.Tensor:
        x = torch.tensor(self.a["image"][idx]).float()
        x = x[:, self.camera_indices] if x.ndim == 5 else x[self.camera_indices]
        return x / 255 if self.a["image"].dtype == np.uint8 else x

    def __getitem__(self, index: int) -> dict:
        i = int(self.indices[index])
        end = self.ends[i]
        offsets = np.arange(i, i + self.chunk)
        idx = np.minimum(offsets, end)
        past = np.maximum(np.arange(i - self.history + 1, i + 1), self.starts[i])
        state = torch.tensor(self.a["state"][i]).float()
        action = torch.tensor(self.a["action"][idx]).float()
        return {
            "image": self._images(i),
            "past": self._images(past),
            "state": (state - self.stats["state_mean"]) / self.stats["state_std"],
            "action": (action - self.stats["action_mean"]) / self.stats["action_std"],
            "valid": torch.tensor(offsets <= end),
            "future": self._images(np.minimum(offsets + 1, end)),
            "future_valid": torch.tensor(offsets + 1 <= end),
        }


def statistics(arrays: dict, episodes: np.ndarray | list) -> dict:
    mask = np.isin(arrays["episode"], episodes)
    if not mask.any():
        raise ValueError("No training frames")
    out = {}
    for key in ["state", "action"]:
        t = torch.tensor(arrays[key][mask]).float()
        out[key + "_mean"] = t.mean(0)
        out[key + "_std"] = t.std(0, unbiased=False).clamp_min(1e-4)
    return out
