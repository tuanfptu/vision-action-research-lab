import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from va_lab.datasets.core import WindowBatches, Windows, generate, read_arrays, statistics
from va_lab.evaluation.research import crossed_interval, freeze_json, matrix, summarize


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_vector_batches_exact_episode_padding_and_shared_pool(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    path = tmp_path / "fixture.npz"
    generate(str(path), episodes=4, length=3, cameras=3)
    a = read_arrays(path)
    stats = statistics(a, [0, 1])
    ds = Windows(a, [0, 2], 5, stats, [2, 0], history=5)
    fast = WindowBatches(ds, 4, device)
    for expected, actual in zip(DataLoader(ds, batch_size=4), fast, strict=True):
        for key in expected:
            assert torch.allclose(expected[key], actual[key].cpu(), atol=1e-7)
    other = WindowBatches(Windows(a, [3], 5, stats, [2, 0], history=5), 2, device, pool=fast.pool)
    assert other.pool["image"].data_ptr() == fast.pool["image"].data_ptr()
    assert len(WindowBatches(ds, 4, device, drop_last=True)) == 1
    torch.manual_seed(42)
    first = torch.cat([b["state"] for b in WindowBatches(ds, 2, device, shuffle=True)])
    torch.manual_seed(42)
    second = torch.cat([b["state"] for b in WindowBatches(ds, 2, device, shuffle=True)])
    assert torch.equal(first, second)


def test_paired_bootstrap_and_incomplete_design_rejected():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    diff = crossed_interval((a + 2) - a, draws=100)
    assert diff["mean"] == 2 and diff["ci95"] == [2, 2]
    assert crossed_interval(a, draws=100) == crossed_interval(a, draws=100)
    with pytest.raises(ValueError):
        crossed_interval([[float("nan")]])
    with pytest.raises(ValueError, match="Missing"):
        matrix([], "bc", [7], [0])
    rows = [{"model": "bc", "seed": 7, "episode": 0, "action_mse_rad2": 1}]
    with pytest.raises(ValueError, match="Duplicate"):
        matrix(rows + rows, "bc", [7], [0])
    assert summarize(rows, ["bc"], [7], [0])["bc"]["paired_difference_vs_bc"]["mean"] == 0


def test_study_selection_lock_final_test_and_resume(tmp_path):
    spec = importlib.util.spec_from_file_location("research_runner", "scripts/research.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    data = tmp_path / "fixture.npz"
    generate(str(data), episodes=8, length=3)
    path = data.with_suffix(".manifest.json")
    meta = json.loads(path.read_text())
    meta.update(state_names=[f"j{i}" for i in range(16)], action_names=[f"j{i}" for i in range(16)])
    path.write_text(json.dumps(meta))
    cfg = {
        "models": ["bc", "state"],
        "seeds": [7, 17],
        "training": {
            "model": "bc",
            "seed": 7,
            "device": "cpu",
            "preload_device": True,
            "epochs": 1,
            "chunk": 2,
            "dim": 16,
            "batch_size": 4,
            "lr": 0.001,
            "cameras": ["test_0", "test_1"],
            "image_size": 32,
            "sampling_steps": 2,
        },
    }
    config = tmp_path / "study.yaml"
    config.write_text(yaml.safe_dump(cfg))
    args = Namespace(
        config=str(config),
        data=str(data),
        output=str(tmp_path / "study"),
        resume=False,
        finalize=False,
    )
    runner.run(args)
    root = Path(args.output)
    assert (root / "selection.json").exists() and not (root / "final_test.json").exists()
    args.resume, args.finalize = True, True
    runner.run(args)
    before = (root / "final_test.json").read_bytes()
    runner.run(args)
    assert before == (root / "final_test.json").read_bytes()
    with pytest.raises(FileExistsError):
        freeze_json(root / "selection.json", {})
    cfg["training"]["epochs"] = 2
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="mismatch"):
        runner.run(args)
