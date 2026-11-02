"""Official OpenArm LeRobot v2.1 acquisition and synchronized multi-view export."""

import json
from pathlib import Path

import numpy as np

from va_lab.datasets.core import file_hash, manifest, read_arrays
from va_lab.utils.runtime import write_json

REPO_ID = "enactic/openarm-2-cell-pick_up_cube_mujoco-lerobot"
REVISION = "c5f7d41a37f9d9456b74eff673870997545c2d25"
CAMERAS = ["ceiling", "head_left", "head_right", "wrist_left", "wrist_right"]
JOINT_NAMES = [
    f"{side}_{joint}.pos"
    for side in ["right", "left"]
    for joint in [*[f"joint{i}" for i in range(1, 8)], "gripper"]
]


def acquire(root: str = "data/openarm_raw", episodes: int = 30, revision: str = REVISION):
    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if episodes < 1 or episodes > 30:
        raise ValueError("OpenArm dataset has exactly 30 available episodes")
    root = Path(root).resolve()
    # LeRobot 0.3.3 parses a SHA as a version if files are missing. Prefetch exact SHA first.
    patterns = ["meta/*", "README.md"]
    for e in range(episodes):
        patterns += [
            f"data/chunk-000/episode_{e:06d}.parquet",
            f"videos/chunk-000/*/episode_{e:06d}.mp4",
        ]
    snapshot_download(
        REPO_ID, repo_type="dataset", revision=revision, local_dir=root, allow_patterns=patterns
    )
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1" or info["fps"] != 30:
        raise ValueError("Expected exact OpenArm v2.1 format at 30 FPS")
    for key in ["action", "observation.state"]:
        if info["features"][key]["shape"] != [16] or info["features"][key]["names"] != JOINT_NAMES:
            raise ValueError(f"Unverified joint schema: {key}")
    return LeRobotDataset(
        REPO_ID,
        root=root,
        revision=revision,
        episodes=list(range(episodes)),
        video_backend="pyav",
        tolerance_s=0.001,
    )


def verify_lerobot(root: str = "data/openarm_raw") -> dict:
    ds = acquire(root, 1)
    x = ds[0]
    return {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "episode": 0,
        "frames": len(ds),
        "sample_shapes": {k: list(v.shape) for k, v in x.items() if hasattr(v, "shape")},
        "action_names": JOINT_NAMES,
        "action": x["action"].tolist(),
        "timestamp": float(x["timestamp"]),
    }


def export_openarm(
    path: str = "data/openarm.npz",
    episodes: int = 30,
    cameras: list[str] | None = None,
    size: int = 32,
    root: str = "data/openarm_raw",
) -> dict:
    import av
    import torch
    from PIL import Image

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite cache {path}")
    cameras = cameras or ["ceiling", "wrist_right"]
    if not cameras or len(set(cameras)) != len(cameras) or any(c not in CAMERAS for c in cameras):
        raise ValueError(f"Select unique cameras from {CAMERAS}")
    if size < 16:
        raise ValueError("Image size must be >=16")
    ds = acquire(root, episodes)
    raw = Path(root).resolve()
    numeric = {k: [] for k in ["state", "action", "episode", "timestamp", "frame"]}
    labels = []
    last_frames = []
    for global_index, row in enumerate(ds.hf_dataset):
        if int(row["index"]) != global_index:
            raise ValueError("Global frame index mismatch")
        labels.append(int(row["success"]))
        last_frames.append(int(row["last_frame_index"]))
        for out, key in [
            ("state", "observation.state"),
            ("action", "action"),
            ("episode", "episode_index"),
            ("timestamp", "timestamp"),
            ("frame", "frame_index"),
        ]:
            value = row[key]
            numeric[out].append(value.numpy() if isinstance(value, torch.Tensor) else value)
    numeric = {k: np.asarray(v) for k, v in numeric.items()}
    images = np.empty((len(ds), len(cameras), 3, size, size), dtype=np.uint8)
    video_checks = []
    for e in range(episodes):
        indices = np.flatnonzero(numeric["episode"] == e)
        times = numeric["timestamp"][indices]
        if not np.all(np.asarray(last_frames)[indices] == len(indices) - 1):
            raise ValueError("last_frame_index disagrees with episode length")
        if len(set(np.asarray(labels)[indices].tolist())) != 1:
            raise ValueError("Inconsistent per-episode demonstration success flag")
        if not np.allclose(times, np.arange(len(indices)) / 30, atol=0.001):
            raise ValueError(f"Irregular timestamps in episode {e}")
        for v, camera in enumerate(cameras):
            video = raw / f"videos/chunk-000/observation.images.{camera}/episode_{e:06d}.mp4"
            count = 0
            max_error = 0.0
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                if abs(float(stream.average_rate) - 30) > 0.001:
                    raise ValueError("Video FPS differs from metadata")
                for count, frame in enumerate(container.decode(video=0), start=1):
                    if count > len(indices):
                        raise ValueError("Video has extra frames")
                    error = abs(float(frame.time) - float(times[count - 1]))
                    max_error = max(max_error, error)
                    if error > 0.001:
                        raise ValueError("Video/numeric synchronization error")
                    im = frame.to_image().resize((size, size), Image.Resampling.BILINEAR)
                    images[indices[count - 1], v] = np.asarray(im).transpose(2, 0, 1)
            if count != len(indices):
                raise ValueError("Video/numeric frame count mismatch")
            video_checks.append(
                {
                    "episode": e,
                    "camera": camera,
                    "frames": count,
                    "max_timestamp_error_s": max_error,
                }
            )
        print(f"Exported episode {e}: {len(indices)} frames, {len(cameras)} cameras", flush=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite cache {path}")
    np.savez_compressed(path, image=images, **numeric)
    read_arrays(str(path))
    metadata = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "revision_name": "v2.1",
        "format": "v2.1",
        "license": "Apache-2.0",
        "fps": 30,
        "episodes": episodes,
        "frames": len(ds),
        "cameras": cameras,
        "available_cameras": CAMERAS,
        "image_size": size,
        "state_names": JOINT_NAMES,
        "action_names": JOINT_NAMES,
        "action_schema": "absolute joint and finger_joint1 position, radians",
        "official_loader": "LeRobotDataset 0.3.3, pyav",
        "upstream_info": ds.meta.info,
        "video_checks": video_checks,
        "upstream_stats_sha256": file_hash(raw / "meta/episodes_stats.jsonl"),
        "normalization": "fit only training episodes; upstream aggregate stats are not used",
        "episode_success_flags": {
            str(e): int(np.asarray(labels)[numeric["episode"] == e][0]) for e in range(episodes)
        },
    }
    manifest(path, metadata)
    write_json(
        path.with_suffix(".integrity.json"),
        {"status": "TESTED", "frames": len(ds), "episodes": episodes, "video_checks": video_checks},
    )
    return {
        "path": str(path),
        "frames": len(ds),
        "episodes": episodes,
        "cameras": cameras,
        "revision": REVISION,
    }
