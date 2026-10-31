"""Episode-safe trajectory windows; all statistics use training episodes only."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from va_lab.utils.runtime import write_json


def generate(path: str, episodes: int = 12, length: int = 24, seed: int = 7) -> None:
    """Two-mode curved reaching trajectories, never labelled as PushT."""
    if episodes < 3 or length < 2:
        raise ValueError("Need >=3 episodes and >=2 frames")
    rng = np.random.default_rng(seed)
    images, states, actions, ids = [], [], [], []
    yy, xx = np.mgrid[:32, :32]
    for episode in range(episodes):
        start, goal = rng.uniform(0.15, 0.85, (2, 2))
        mode = rng.choice([-1, 1])
        t = np.linspace(0, 1, length + 1)[:, None]
        curve = start + (goal - start) * t + mode * 0.13 * np.sin(np.pi * t) * np.array([1, -1])
        curve = np.clip(curve, 0.02, 0.98)
        for k in range(length):
            frame = np.zeros((3, 32, 32), np.float32)
            frame[0] = (xx - curve[k, 0] * 31) ** 2 + (yy - curve[k, 1] * 31) ** 2 < 8
            frame[1] = (xx - goal[0] * 31) ** 2 + (yy - goal[1] * 31) ** 2 < 8
            images.append(frame)
            states.append(curve[k])
            actions.append(curve[k + 1])
            ids.append(episode)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image=np.array(images),
        state=np.array(states, np.float32),
        action=np.array(actions, np.float32),
        episode=np.array(ids),
    )
    manifest(
        path,
        {
            "source": "synthetic curved reaching",
            "seed": seed,
            "episodes": episodes,
            "action_schema": "absolute position in [0,1]^2",
        },
    )


def manifest(path: Path, metadata: dict) -> None:
    metadata = dict(metadata, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write_json(path.with_suffix(".manifest.json"), metadata)


def read_arrays(path: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in ["image", "state", "action", "episode"]}
    n = len(arrays["episode"])
    if n == 0 or any(len(v) != n for v in arrays.values()):
        raise ValueError("Missing or inconsistent frames")
    if arrays["image"].ndim != 4 or arrays["image"].shape[1] != 3:
        raise ValueError("Images must be NCHW RGB")
    if arrays["state"].shape != (n, 2) or arrays["action"].shape != (n, 2):
        raise ValueError("This benchmark requires 2D state and absolute 2D actions")
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Nonfinite dataset value")
    if arrays["image"].min() < 0 or arrays["image"].max() > 1:
        raise ValueError("Images must lie in [0,1]")
    for eid in np.unique(arrays["episode"]):
        idx = np.flatnonzero(arrays["episode"] == eid)
        if np.any(np.diff(idx) != 1):
            raise ValueError("Episode frames must be contiguous")
    return arrays


def episode_split(ids: np.ndarray, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(ids)
    if len(unique) < 3:
        raise ValueError("Need at least three episodes for train/validation")
    np.random.default_rng(seed).shuffle(unique)
    cut = max(1, min(len(unique) - 1, int(0.8 * len(unique))))
    return unique[:cut], unique[cut:]


class Windows(Dataset):
    def __init__(self, arrays: dict, episodes: np.ndarray, chunk: int, stats: dict):
        self.a, self.chunk, self.stats = arrays, chunk, stats
        self.indices = np.flatnonzero(np.isin(arrays["episode"], episodes))
        self.ends = {}
        self.starts = {}
        for e in episodes:
            idx = np.flatnonzero(arrays["episode"] == e)
            for i in idx:
                self.ends[int(i)] = int(idx[-1])
                self.starts[int(i)] = int(idx[0])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        i = int(self.indices[index])
        end = self.ends[i]
        idx = np.minimum(np.arange(i, i + self.chunk), end)
        valid = torch.tensor(np.arange(i, i + self.chunk) <= end)
        state = torch.tensor(self.a["state"][i]).float()
        action = torch.tensor(self.a["action"][idx]).float()
        image = torch.tensor(self.a["image"][i]).float()
        # Frame k+1, padded only after the true episode end.
        future = torch.tensor(self.a["image"][np.minimum(idx + 1, end)]).float()
        return {
            "image": image,
            "past": torch.tensor(
                self.a["image"][
                    np.maximum(
                        np.arange(i - 1, i + 1),
                        self.starts[i],
                    )
                ]
            ).float(),
            "state": (state - self.stats["state_mean"]) / self.stats["state_std"],
            "action": (action - self.stats["action_mean"]) / self.stats["action_std"],
            "valid": valid,
            "future": future,
            "future_valid": torch.tensor(np.arange(i + 1, i + 1 + self.chunk) <= end),
        }


def statistics(arrays: dict, episodes: np.ndarray) -> dict:
    mask = np.isin(arrays["episode"], episodes)
    out = {}
    for key in ["state", "action"]:
        t = torch.tensor(arrays[key][mask]).float()
        out[key + "_mean"] = t.mean(0)
        out[key + "_std"] = t.std(0, unbiased=False).clamp_min(1e-4)
    return out
