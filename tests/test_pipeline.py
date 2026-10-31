import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from va_lab.controllers.hierarchy import BoundedController, TemporalEnsemble
from va_lab.datasets.core import Windows, episode_split, generate, read_arrays, statistics
from va_lab.models.policies import build, masked_mse
from va_lab.training.engine import evaluate_checkpoint, load_checkpoint, train
from va_lab.utils.runtime import seed_all


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "toy.npz"
    generate(str(path), episodes=5, length=6)
    return path


def config(model="bc"):
    return dict(
        model=model,
        chunk=4,
        dim=32,
        seed=7,
        batch_size=8,
        device="cpu",
        epochs=1,
        lr=0.001,
        sampling_steps=3,
    )


@pytest.mark.parametrize("name", ["bc", "state", "act", "diffusion", "flow", "video"])
def test_forward_backward_and_prior(name, dataset):
    seed_all(7)
    a = read_arrays(str(dataset))
    tr, va = episode_split(a["episode"])
    batch = next(iter(DataLoader(Windows(a, tr, 4, statistics(a, tr)), batch_size=3)))
    model = build(config(name))
    loss = model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    model.eval()
    with torch.no_grad():
        sample = model.sample(batch, 3)
    assert sample.shape == (3, 4, 2)
    assert torch.isfinite(sample).all()
    if name == "act":
        before = sample.clone()
        batch["action"] = torch.randn_like(batch["action"]) * 100
        with torch.no_grad():
            assert torch.allclose(before, model.sample(batch), atol=1e-6)
    assert set(tr).isdisjoint(va)


def test_padding_and_stats(dataset):
    a = read_arrays(str(dataset))
    tr, _ = episode_split(a["episode"])
    stats = statistics(a, tr)
    ds = Windows(a, tr, 4, stats)
    for x in ds:
        assert x["valid"][0]
        assert torch.all(x["valid"][1:].int() <= x["valid"][:-1].int())
    mean = stats["action_mean"].clone()
    val = ~np.isin(a["episode"], tr)
    a["action"][val] += 1000
    assert torch.equal(mean, statistics(a, tr)["action_mean"])
    assert masked_mse(torch.ones(1, 2, 2), torch.zeros(1, 2, 2), torch.tensor([[1, 0]])) == 1


def test_train_checkpoint_and_hash(dataset, tmp_path):
    out = tmp_path / "run"
    metrics = train(config(), str(dataset), str(out))
    assert metrics["evaluated_action_scalars"] > 0
    model, _ = load_checkpoint(str(out / "best.pt"))
    assert not model.training
    again = evaluate_checkpoint(str(out / "best.pt"), str(dataset), str(tmp_path / "eval.json"))
    assert again["normalized_action_mse"] == metrics["normalized_action_mse"]
    generate(str(dataset), episodes=6)
    with pytest.raises(ValueError, match="hash"):
        evaluate_checkpoint(str(out / "best.pt"), str(dataset), str(tmp_path / "bad.json"))


def test_watchdog_interpolation_and_ensemble():
    c = BoundedController(np.zeros(2), np.ones(2))
    position = np.array([0.5, 0.5])
    assert np.allclose(c.step(position, np.ones(2) * 10, 0, 0.01), [0.52, 0.52])
    assert np.allclose(c.step(position, np.ones(2), 1, 0.01), position)
    assert np.allclose(c.step(position, np.array([np.nan, 0]), 0, 0.01), position)
    e = TemporalEnsemble(0)
    assert np.allclose(e.update(0, np.array([[0.0, 0.0], [2.0, 2.0]])), 0)
    assert np.allclose(e.update(1, np.array([[4.0, 4.0], [8.0, 8.0]])), 3)


def test_mujoco_smoke():
    pytest.importorskip("mujoco")
    from va_lab.simulation.benchmark import smoke

    assert smoke()["finite"]


def test_history_never_crosses_episode(dataset):
    a = read_arrays(str(dataset))
    tr, _ = episode_split(a["episode"])
    ds = Windows(a, tr, 4, statistics(a, tr))
    for j, raw in enumerate(ds.indices):
        if raw == 0 or a["episode"][raw] != a["episode"][raw - 1]:
            assert torch.equal(ds[j]["past"][0], ds[j]["image"])


def test_simulator_schema_rejects_synthetic(dataset, tmp_path):
    from va_lab.simulation.benchmark import pusht_rollout

    pytest.importorskip("gym_pusht")
    train(config(), str(dataset), str(tmp_path / "run"))
    with pytest.raises(ValueError, match="PushT"):
        pusht_rollout(str(tmp_path / "run/best.pt"), str(tmp_path / "rollout.json"))
