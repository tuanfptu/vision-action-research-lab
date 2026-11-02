"""Real 16-dimensional multi-camera forward/backward; CUDA is required."""

import torch

from va_lab.models.policies import ACT

if not torch.cuda.is_available():
    raise RuntimeError("Activate .venv-gpu and allow NVIDIA device access")
model = ACT(chunk=4, dim=32, state_dim=16, action_dim=16, cameras=2).cuda()
batch = {
    "image": torch.rand(2, 2, 3, 32, 32, device="cuda"),
    "state": torch.rand(2, 16, device="cuda"),
    "action": torch.rand(2, 4, 16, device="cuda"),
    "valid": torch.ones(2, 4, dtype=torch.bool, device="cuda"),
}
loss = model.loss(batch)
loss.backward()
print(
    {
        "device": str(next(model.parameters()).device),
        "loss": float(loss.detach()),
        "output_shape": list(model.sample(batch).shape),
    }
)
