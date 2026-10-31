"""Adapter around installed official LeRobot ACT; no upstream code is copied."""

import torch
from torch import nn


class OfficialACT(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        # Outer pipeline already normalizes state/action. Identity avoids double normalization.
        config = ACTConfig(
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
            normalization_mapping={
                key: NormalizationMode.IDENTITY for key in ["VISUAL", "STATE", "ACTION"]
            },
            chunk_size=cfg["chunk"],
            n_action_steps=1,
            dim_model=cfg.get("dim", 64),
            n_heads=4,
            dim_feedforward=cfg.get("dim", 64) * 2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            n_vae_encoder_layers=1,
            latent_dim=cfg.get("latent", 8),
            kl_weight=cfg.get("beta", 0.01),
            dropout=0.0,
            pretrained_backbone_weights=None,
        )
        self.policy = ACTPolicy(config)

    @staticmethod
    def adapt(batch: dict) -> dict:
        out = {"observation.state": batch["state"], "observation.images.top": batch["image"]}
        if "action" in batch:
            out.update(action=batch["action"], action_is_pad=~batch["valid"])
        return out

    def loss(self, batch: dict) -> torch.Tensor:
        return self.policy(self.adapt(batch))[0]

    def sample(self, batch: dict, steps: int = 10) -> torch.Tensor:
        return self.policy.predict_action_chunk(self.adapt(batch))
