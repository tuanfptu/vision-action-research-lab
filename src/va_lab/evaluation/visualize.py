"""Headless observation, action, and predicted future-frame visualization."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from va_lab.datasets.core import Windows, read_arrays
from va_lab.training.engine import load_checkpoint


def inspect(data: str, output: str) -> dict:
    a = read_arrays(data)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(a["image"][0].transpose(1, 2, 0))
    axes[0].set_title("Observation frame 0")
    axes[1].plot(a["action"][:24, 0], label="x")
    axes[1].plot(a["action"][:24, 1], label="y")
    axes[1].legend()
    axes[1].set_title("First 24 absolute targets")
    axes[2].plot(a["state"][:24, 0], a["state"][:24, 1], "o-")
    axes[2].set_title("Observed xy trajectory")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in a.items()}


def video(checkpoint: str, data: str, output: str) -> None:
    model, ckpt = load_checkpoint(checkpoint)
    if ckpt["config"]["model"] != "video":
        raise ValueError("Video visualization requires video checkpoint")
    ds = Windows(read_arrays(data), ckpt["val_ids"], ckpt["config"]["chunk"], ckpt["stats"])
    batch = {k: v[None] for k, v in ds[0].items()}
    with torch.no_grad():
        _, pred = model.forward_video(batch)
    k = pred.shape[1]
    fig, axes = plt.subplots(2, k, figsize=(2 * k, 4), squeeze=False)
    for t in range(k):
        axes[0, t].imshow(batch["future"][0, t].permute(1, 2, 0))
        axes[1, t].imshow(pred[0, t].permute(1, 2, 0))
        axes[0, t].set_title(f"Truth +{t + 1}")
        axes[1, t].set_title("Prediction")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
