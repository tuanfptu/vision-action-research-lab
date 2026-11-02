"""Multi-camera observations, every joint trajectory and temporal video predictions."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from va_lab.datasets.core import Windows, read_arrays
from va_lab.training.engine import load_checkpoint


def inspect(data: str, output: str) -> dict:
    a = read_arrays(data)
    meta = json.loads(Path(data).with_suffix(".manifest.json").read_text())
    v = a["image"].shape[1]
    fig, axes = plt.subplots(2, max(v, 2), figsize=(4 * max(v, 2), 7), squeeze=False)
    for camera in range(v):
        image = a["image"][0, camera].transpose(1, 2, 0)
        axes[0, camera].imshow(image)
        axes[0, camera].set_title(meta["cameras"][camera])
    first = a["episode"] == a["episode"][0]
    axes[1, 0].plot(a["timestamp"][first], a["state"][first])
    axes[1, 0].set_title("All joint states (rad)")
    axes[1, 1].plot(a["timestamp"][first], a["action"][first])
    axes[1, 1].set_title("All absolute joint targets (rad)")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    action_dim = a["action"].shape[1]
    names = meta.get("action_names", [f"joint_{i}" for i in range(action_dim)])
    columns = 4
    rows = (action_dim + columns - 1) // columns
    joint_fig, joint_axes = plt.subplots(rows, columns, figsize=(16, 3 * rows), squeeze=False)
    for j, axis in enumerate(joint_axes.flat):
        if j >= action_dim:
            axis.axis("off")
            continue
        axis.plot(a["timestamp"][first], a["action"][first, j], label="target")
        if j < a["state"].shape[1]:
            axis.plot(a["timestamp"][first], a["state"][first, j], label="state", alpha=0.8)
        axis.set_title(names[j])
        axis.set_xlabel("seconds")
        axis.set_ylabel("radians")
        axis.legend()
    joint_fig.tight_layout()
    joint_fig.savefig(Path(output).with_name(Path(output).stem + "_joints.png"))
    plt.close(joint_fig)
    return {k: {"shape": list(value.shape), "dtype": str(value.dtype)} for k, value in a.items()}


def video(checkpoint: str, data: str, output: str) -> None:
    model, ckpt = load_checkpoint(checkpoint, "cpu")
    cfg = ckpt["config"]
    if cfg["model"] != "video":
        raise ValueError("Video checkpoint required")
    ds = Windows(
        read_arrays(data),
        ckpt["val_ids"],
        cfg["chunk"],
        ckpt["stats"],
        ckpt["camera_indices"],
        cfg.get("history", 2),
    )
    batch = {k: v[None] for k, v in ds[0].items()}
    with torch.no_grad():
        _, pred = model.forward_video(batch)
    k, v = pred.shape[1:3]
    fig, axes = plt.subplots(v * 2, k, figsize=(2 * k, 3 * v), squeeze=False)
    for camera in range(v):
        for t in range(k):
            axes[2 * camera, t].imshow(batch["future"][0, t, camera].permute(1, 2, 0))
            axes[2 * camera + 1, t].imshow(pred[0, t, camera].permute(1, 2, 0))
            axes[2 * camera, t].set_title(f"{cfg['cameras'][camera]} truth +{t + 1}")
            axes[2 * camera + 1, t].set_title("Prediction")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
