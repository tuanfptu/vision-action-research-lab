"""A small real forward/backward pass without dataset/model downloads."""

import torch

from va_lab.models.policies import ACT

model = ACT(chunk=4, dim=32)
batch = {
    "image": torch.rand(2, 3, 32, 32),
    "state": torch.rand(2, 2),
    "action": torch.rand(2, 4, 2),
    "valid": torch.ones(2, 4, dtype=torch.bool),
}
loss = model.loss(batch)
loss.backward()
model.eval()
print({"loss": float(loss.detach()), "predicted_shape": list(model.sample(batch).shape)})
