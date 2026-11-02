import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from va_lab.controllers.hierarchy import BoundedController, TemporalEnsemble
from va_lab.datasets.core import Windows, episode_split, generate, read_arrays, statistics
from va_lab.models.policies import build, masked_mse
from va_lab.training.engine import evaluate_checkpoint, load_checkpoint, train
from va_lab.utils.runtime import seed_all


def config(model="bc", device="cpu", state_dim=16, action_dim=16, cameras=None):
    return dict(
        model=model,
        chunk=4,
        dim=32,
        seed=7,
        batch_size=4,
        device=device,
        epochs=1,
        lr=0.001,
        sampling_steps=3,
        state_dim=state_dim,
        action_dim=action_dim,
        cameras=cameras or ["test_0", "test_1"],
        image_size=32,
        gradient_accumulation=3,
    )


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "test.npz"
    generate(str(path), episodes=8, length=5)
    return path


@pytest.mark.parametrize("name", ["bc", "state", "act", "diffusion", "flow", "video"])
@pytest.mark.parametrize("state_dim,action_dim,views", [(16, 16, 2), (9, 13, 1), (16, 16, 5)])
def test_configurable_forward_backward(name, state_dim, action_dim, views):
    seed_all(7)
    cfg = config(
        name, state_dim=state_dim, action_dim=action_dim, cameras=[f"v{i}" for i in range(views)]
    )
    batch = {
        "image": torch.rand(2, views, 3, 32, 32),
        "past": torch.rand(2, 2, views, 3, 32, 32),
        "state": torch.rand(2, state_dim),
        "action": torch.rand(2, 4, action_dim),
        "valid": torch.ones(2, 4, dtype=torch.bool),
        "future": torch.rand(2, 4, views, 3, 32, 32),
        "future_valid": torch.ones(2, 4, dtype=torch.bool),
    }
    model = build(cfg)
    loss = model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    model.eval()
    with torch.no_grad():
        pred = model.sample(batch, 3)
        assert pred.shape == (2, 4, action_dim) and torch.isfinite(pred).all()
        if name == "act":
            batch["action"] += 100
            assert torch.allclose(pred, model.sample(batch), atol=1e-6)


def test_split_padding_history_and_stats(dataset):
    a = read_arrays(str(dataset))
    tr, va, te = episode_split(a["episode"])
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    stats = statistics(a, tr)
    ds = Windows(a, tr, 4, stats)
    before = stats["action_mean"].clone()
    a["action"][~np.isin(a["episode"], tr)] += 1000
    assert torch.equal(before, statistics(a, tr)["action_mean"])
    for j, raw in enumerate(ds.indices):
        x = ds[j]
        assert torch.all(x["valid"][1:].int() <= x["valid"][:-1].int())
        if raw == ds.starts[int(raw)]:
            assert torch.equal(x["past"][0], x["image"])
        if raw == ds.ends[int(raw)]:
            assert int(x["valid"].sum()) == 1 and int(x["future_valid"].sum()) == 0
    assert masked_mse(torch.ones(1, 2, 16), torch.zeros(1, 2, 16), torch.tensor([[1, 0]])) == 1


def test_train_checkpoint_and_explicit_test(dataset, tmp_path):
    out = tmp_path / "run"
    m = train(config(), str(dataset), str(out))
    model, ckpt = load_checkpoint(str(out / "best.pt"))
    assert not model.training and m["split"] == "validation"
    assert len(m["joint_rmse_rad"]) == 16
    assert set(ckpt["test_ids"]).isdisjoint(ckpt["val_ids"])
    again = evaluate_checkpoint(str(out / "best.pt"), str(dataset), str(tmp_path / "eval.json"))
    assert again["normalized_action_mse"] == m["normalized_action_mse"]
    test = evaluate_checkpoint(
        str(out / "best.pt"), str(dataset), str(tmp_path / "test.json"), split="test"
    )
    assert test["split"] == "test"
    generate(str(dataset), episodes=9)
    with pytest.raises(ValueError, match="hash"):
        evaluate_checkpoint(str(out / "best.pt"), str(dataset), str(tmp_path / "bad.json"))


def test_data_integrity_rejects_bad_order(dataset, tmp_path):
    a = read_arrays(str(dataset))
    a["frame"][1] = 0
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **a)
    with pytest.raises(ValueError, match="ordered"):
        read_arrays(str(bad))


def test_controller_and_ensemble():
    c = BoundedController(-np.ones(16), np.ones(16))
    q = np.zeros(16)
    assert np.allclose(c.step(q, np.ones(16) * 10, 0, 0.01), 0.02)
    assert np.allclose(c.step(q, np.ones(16), 1, 0.01), q)
    e = TemporalEnsemble(0)
    e.update(0, np.zeros((2, 16)))
    assert np.allclose(e.update(1, np.ones((2, 16)) * 2), 1)


@pytest.mark.skipif(
    not Path("assets/upstream/openarm_mujoco/v2/cell/demo.xml").exists(),
    reason="Run va-lab assets for official model integration",
)
def test_official_mujoco_joint_mapping():
    pytest.importorskip("mujoco")
    from va_lab.simulation.benchmark import OpenArmEnv, smoke

    result = smoke()
    assert result["finite"] and result["action_shape"] == [16] and result["nu"] == 17
    env = OpenArmEnv(render_images=False)
    try:
        assert len(set(env.mapping["ctrl"])) == 16
        obs, info = env.step(np.ones(16) * 1e6)
        assert info["command_clipped_values"] == 16 and np.isfinite(obs["state"]).all()
        assert np.all(env.data.ctrl[env.mapping["ctrl"]] <= env.mapping["upper"])
        assert np.all(env.data.ctrl[env.mapping["ctrl"]] >= env.mapping["lower"])
        with pytest.raises(ValueError, match="shape"):
            env.step(np.zeros(15))
    finally:
        env.close()


@pytest.mark.skipif(
    os.environ.get("VA_LAB_INTEGRATION") != "1", reason="Explicit real dataset integration"
)
def test_real_official_openarm():
    from va_lab.datasets.public import acquire

    ds = acquire(episodes=1)
    sample = ds[0]
    assert sample["action"].shape == (16,) and sample["observation.state"].shape == (16,)
    assert len([k for k in sample if k.startswith("observation.images.")]) == 5
    a = read_arrays("data/openarm.npz")
    assert len(a["state"]) == 5405 and len(np.unique(a["episode"])) == 30
    meta = json.loads(Path("data/openarm.manifest.json").read_text())
    tr, _, _ = episode_split(a["episode"])
    b = next(iter(DataLoader(Windows(a, tr, 4, statistics(a, tr)), batch_size=2)))
    assert b["image"].shape == (2, len(meta["cameras"]), 3, 32, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("name", ["bc", "act", "diffusion", "flow", "video"])
def test_gpu_forward_backward(name):
    seed_all(7)
    a = {
        "image": torch.rand(2, 2, 3, 32, 32, device="cuda"),
        "state": torch.rand(2, 16, device="cuda"),
        "action": torch.rand(2, 4, 16, device="cuda"),
        "valid": torch.ones(2, 4, dtype=torch.bool, device="cuda"),
        "future": torch.rand(2, 4, 2, 3, 32, 32, device="cuda"),
        "future_valid": torch.ones(2, 4, dtype=torch.bool, device="cuda"),
    }
    model = build(config(name, "cuda")).cuda()
    loss = model.loss(a)
    loss.backward()
    assert torch.isfinite(loss) and next(model.parameters()).is_cuda
