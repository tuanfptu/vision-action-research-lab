"""Small original educational models; not official ACT/APT/LingBot reproductions."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.to(pred.dtype)
    while weight.ndim < pred.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(pred)
    return ((pred - target).square() * weight).sum() / weight.sum().clamp_min(1)


class Encoder(nn.Module):
    def __init__(self, dim: int = 64, vision: bool = True):
        super().__init__()
        self.vision = vision
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(512, dim),
            nn.ReLU(),
        )
        self.fuse = nn.Linear(dim + 2 if vision else 2, dim)

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.cnn(image), state], -1) if self.vision else state
        return F.relu(self.fuse(x))


class BC(nn.Module):
    def __init__(self, chunk: int, dim: int = 64, vision: bool = True):
        super().__init__()
        self.chunk = chunk
        self.encoder = Encoder(dim, vision)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, chunk * 2))

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(self.encoder(batch["image"], batch["state"])).view(-1, self.chunk, 2)

    def loss(self, batch: dict) -> torch.Tensor:
        return masked_mse(self(batch), batch["action"], batch["valid"])

    def sample(self, batch: dict, steps: int = 10) -> torch.Tensor:
        return self(batch)


class ACT(nn.Module):
    def __init__(self, chunk: int, dim: int = 64, latent: int = 8, beta: float = 0.01):
        super().__init__()
        self.chunk, self.latent, self.beta = chunk, latent, beta
        self.encoder = Encoder(dim)
        self.action_in = nn.Linear(2, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, chunk + 1, dim) * 0.02)
        self.posterior = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, dropout=0), 1
        )
        self.moments = nn.Linear(dim, latent * 2)
        self.z_in = nn.Linear(latent, dim)
        self.queries = nn.Parameter(torch.randn(1, chunk, dim) * 0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, 4, dim * 2, batch_first=True, dropout=0), 1
        )
        self.output = nn.Linear(dim, 2)

    def forward(self, batch: dict, posterior: bool = False) -> tuple:
        context = self.encoder(batch["image"], batch["state"])
        b = len(context)
        mu = context.new_zeros(b, self.latent)
        logvar = torch.zeros_like(mu)
        if posterior:
            tokens = torch.cat(
                [self.cls.expand(b, -1, -1) + context[:, None], self.action_in(batch["action"])], 1
            )
            pad = torch.cat(
                [torch.zeros(b, 1, dtype=torch.bool, device=context.device), ~batch["valid"]], 1
            )
            h = self.posterior(tokens + self.pos, src_key_padding_mask=pad)[:, 0]
            mu, logvar = self.moments(h).chunk(2, -1)
            logvar = logvar.clamp(-10, 10)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu  # deterministic prior mean, never posterior actions during evaluation
        memory = torch.stack([context, self.z_in(z)], 1)
        pred = self.output(self.decoder(self.queries.expand(b, -1, -1), memory))
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
        return pred, kl

    def loss(self, batch: dict) -> torch.Tensor:
        pred, kl = self(batch, posterior=True)
        return masked_mse(pred, batch["action"], batch["valid"]) + self.beta * kl

    def sample(self, batch: dict, steps: int = 10) -> torch.Tensor:
        return self(batch)[0]


class Generative(nn.Module):
    def __init__(self, chunk: int, dim: int = 64, flow: bool = False):
        super().__init__()
        self.chunk, self.flow = chunk, flow
        self.encoder = Encoder(dim)
        self.net = nn.Sequential(
            nn.Linear(chunk * 2 + dim + 3, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, chunk * 2),
        )
        self.register_buffer("betas", torch.linspace(0.0001, 0.02, 1000))
        self.register_buffer("alpha_bar", (1 - self.betas).cumprod(0))

    def predict(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        te = torch.stack([t, torch.sin(t * math.pi), torch.cos(t * math.pi)], -1)
        return self.net(torch.cat([x.flatten(1), context, te], -1)).view_as(x)

    def loss(self, batch: dict) -> torch.Tensor:
        action = batch["action"]
        noise = torch.randn_like(action)
        context = self.encoder(batch["image"], batch["state"])
        if self.flow:
            t = torch.rand(len(action), device=action.device)
            x = (1 - t[:, None, None]) * noise + t[:, None, None] * action
            target = action - noise
        else:
            idx = torch.randint(1000, (len(action),), device=action.device)
            t = idx / 999
            ab = self.alpha_bar[idx, None, None]
            x, target = ab.sqrt() * action + (1 - ab).sqrt() * noise, noise
        return masked_mse(self.predict(x, t, context), target, batch["valid"])

    def sample(self, batch: dict, steps: int = 10) -> torch.Tensor:
        if steps < 1 or steps > 1000:
            raise ValueError("Sampling steps must be in [1,1000]")
        context = self.encoder(batch["image"], batch["state"])
        x = context.new_empty(len(context), self.chunk, 2).normal_()
        if self.flow:
            for k in range(steps):
                t = context.new_full((len(x),), k / steps)
                x = x + self.predict(x, t, context) / steps
        else:
            indices = torch.linspace(999, 0, steps).long().tolist()
            for k, idx in enumerate(indices):
                eps = self.predict(x, context.new_full((len(x),), idx / 999), context)
                ab = self.alpha_bar[idx]
                clean = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
                previous = self.alpha_bar[indices[k + 1]] if k + 1 < steps else x.new_tensor(1.0)
                x = previous.sqrt() * clean + (1 - previous).sqrt() * eps  # deterministic DDIM
        return x


class VideoAction(BC):
    def __init__(self, chunk: int, dim: int = 64):
        super().__init__(chunk, dim)
        self.frames = nn.Linear(dim, chunk * 3 * 16 * 16)
        self.temporal = nn.GRU(dim, dim, batch_first=True)

    def encode_history(self, batch: dict) -> torch.Tensor:
        past = batch.get("past", batch["image"][:, None])
        b, t, c, h, w = past.shape
        states = batch["state"][:, None].expand(-1, t, -1).reshape(b * t, 2)
        embedding = self.encoder(past.reshape(b * t, c, h, w), states).view(b, t, -1)
        _, hidden = self.temporal(embedding)
        return hidden[0]

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(self.encode_history(batch)).view(-1, self.chunk, 2)

    def forward_video(self, batch: dict) -> tuple:
        h = self.encode_history(batch)
        return (
            self.head(h).view(-1, self.chunk, 2),
            self.frames(h).sigmoid().view(-1, self.chunk, 3, 16, 16),
        )

    def loss(self, batch: dict) -> torch.Tensor:
        action, frames = self.forward_video(batch)
        b, k, c, h, w = batch["future"].shape
        target = F.interpolate(batch["future"].reshape(b * k, c, h, w), (16, 16)).view_as(frames)
        return masked_mse(action, batch["action"], batch["valid"]) + masked_mse(
            frames, target, batch["future_valid"]
        )


def build(cfg: dict) -> nn.Module:
    name, chunk, dim = cfg["model"], cfg["chunk"], cfg.get("dim", 64)
    if name == "official_act":
        from va_lab.models.official import OfficialACT

        return OfficialACT(cfg)
    if name in ["bc", "state"]:
        return BC(chunk, dim, name != "state")
    if name == "act":
        return ACT(chunk, dim, cfg.get("latent", 8), cfg.get("beta", 0.01))
    if name in ["diffusion", "flow"]:
        return Generative(chunk, dim, name == "flow")
    if name == "video":
        return VideoAction(chunk, dim)
    raise ValueError(f"Unknown model: {name}")
