"""Pinned public PushT image/parquet export, plus official LeRobot verification."""

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from va_lab.datasets.core import manifest, read_arrays


def export_pusht(path: str, episodes: int = 6, revision: str | None = None) -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    repo = "lerobot/pusht_image"
    revision = revision or HfApi().dataset_info(repo, revision="v2.1").sha
    frames = {k: [] for k in ["image", "state", "action", "episode"]}
    for episode in range(episodes):
        file = hf_hub_download(
            repo,
            f"data/chunk-000/episode_{episode:06d}.parquet",
            repo_type="dataset",
            revision=revision,
        )
        ds = load_dataset("parquet", data_files=file, split="train")
        for row in ds:
            im = row["observation.image"]
            if isinstance(im, dict):
                im = Image.open(BytesIO(im["bytes"])) if im.get("bytes") else Image.open(im["path"])
            im = np.asarray(im.convert("RGB").resize((32, 32)), dtype=np.float32) / 255
            frames["image"].append(im.transpose(2, 0, 1))
            frames["state"].append(row["observation.state"])
            frames["action"].append(row["action"])
            frames["episode"].append(episode)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in frames.items()})
    read_arrays(str(path))
    manifest(
        path,
        {
            "repo_id": repo,
            "revision": revision,
            "episodes": episodes,
            "fps": 10,
            "resize": [32, 32],
            "license": "MIT",
            "action_schema": "absolute PushT agent target xy, simulator pixel coordinates",
        },
    )


def verify_lerobot() -> dict:
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    torch.set_num_threads(2)
    torch.manual_seed(7)
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    }
    cfg = ACTConfig(
        input_features=features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        chunk_size=4,
        n_action_steps=2,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
        pretrained_backbone_weights=None,
    )
    stats = {
        key: {"mean": torch.zeros(feature.shape), "std": torch.ones(feature.shape)}
        for key, feature in {**features, **cfg.output_features}.items()
    }
    policy = ACTPolicy(cfg, dataset_stats=stats)
    batch = {
        "observation.state": torch.zeros(2, 2),
        "observation.images.top": torch.zeros(2, 3, 96, 96),
        "action": torch.zeros(2, 4, 2),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }
    loss, _ = policy(batch)
    loss.backward()
    # Metadata retrieval checks the public version independently of the image exporter.
    meta = LeRobotDatasetMetadata("lerobot/pusht_image")
    dataset = LeRobotDataset("lerobot/pusht_image", episodes=[0], video_backend="pyav")
    sample = dataset[0]
    return {
        "act_loss": float(loss.detach()),
        "features": str(meta.features),
        "official_dataset_length": len(dataset),
        "official_sample_shapes": {
            k: list(v.shape) for k, v in sample.items() if isinstance(v, torch.Tensor)
        },
    }
