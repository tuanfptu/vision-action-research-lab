"""PushT closed-loop evaluation, independent MuJoCo reacher and recording."""

import time

import numpy as np
import torch

from va_lab.controllers.hierarchy import BoundedController, TemporalEnsemble
from va_lab.training.engine import load_checkpoint
from va_lab.utils.runtime import seed_all, write_json


def pusht_rollout(
    checkpoint: str,
    output: str,
    episodes: int = 3,
    horizon: int = 1,
    ensemble: bool = False,
    max_steps: int = 300,
) -> dict:
    import gym_pusht  # noqa: F401
    import gymnasium as gym
    from PIL import Image

    seed_all(10000)
    model, ckpt = load_checkpoint(checkpoint)
    if ckpt.get("provenance", {}).get("repo_id") != "lerobot/pusht_image":
        raise ValueError("PushT rollout requires a checkpoint trained on public PushT coordinates")
    stats = ckpt["stats"]
    if horizon < 1 or horizon > ckpt["config"]["chunk"]:
        raise ValueError("Execution horizon must lie in [1, chunk]")
    if ensemble and horizon != 1:
        raise ValueError("Temporal ensembling requires horizon=1")
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    scores, successes, latencies, deltas = [], [], [], []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=10000 + episode)
            cache, cursor, prev = None, 0, None
            temporal = TemporalEnsemble()
            score, success = 0.0, False
            for tick in range(max_steps):
                if cursor == 0 or ensemble:
                    image = (
                        np.asarray(Image.fromarray(obs["pixels"]).resize((32, 32)), np.float32)
                        / 255
                    )
                    batch = {
                        "image": torch.tensor(image.transpose(2, 0, 1))[None],
                        "state": (
                            torch.tensor(obs["agent_pos"]).float()[None] - stats["state_mean"]
                        )
                        / stats["state_std"],
                    }
                    start = time.perf_counter()
                    with torch.no_grad():
                        cache = (
                            model.sample(batch, ckpt["config"].get("sampling_steps", 10))[0]
                            * stats["action_std"]
                            + stats["action_mean"]
                        ).numpy()
                    latencies.append((time.perf_counter() - start) * 1000)
                action = temporal.update(tick, cache) if ensemble else cache[cursor]
                action = np.clip(action, env.action_space.low, env.action_space.high)
                if prev is not None:
                    deltas.append(float(np.square(action - prev).mean()))
                prev = action.copy()
                obs, reward, terminated, truncated, _ = env.step(action)
                score = max(score, float(reward))
                success = success or bool(terminated)
                cursor = (cursor + 1) % horizon
                if terminated or truncated:
                    break
            scores.append(score)
            successes.append(success)
    finally:
        env.close()
    result = {
        "success_rate": float(np.mean(successes)),
        "max_reward_mean": float(np.mean(scores)),
        "policy_latency_ms": float(np.mean(latencies)),
        "action_delta_mse": float(np.mean(deltas)) if deltas else 0.0,
        "episodes": episodes,
        "seed_start": 10000,
        "execution_horizon": horizon,
        "temporal_ensemble": ensemble,
        "max_steps": max_steps,
    }
    write_json(output, result)
    return result


XML = """<mujoco><option timestep="0.01"/><worldbody>
<geom type="plane" size="1 1 .1"/><body pos="0 0 .06">
<joint name="x" type="slide" axis="1 0 0" limited="true" range="0 1"/>
<joint name="y" type="slide" axis="0 1 0" limited="true" range="0 1"/>
<geom type="sphere" size=".04" mass="1" rgba="1 0 0 1"/>
</body></worldbody></mujoco>"""


def hierarchy(checkpoint: str, output: str, ticks: int = 200, policy_hz: int = 10) -> dict:
    """System 2 chooses a goal; learned System 1 proposes targets; System 0 bounds tracking.

    Original two-slide MJCF, not Panda or a physical robot. Software stepping is simulated time.
    """
    import mujoco

    seed_all(10000)
    model, ckpt = load_checkpoint(checkpoint)
    stats = ckpt["stats"]
    mj = mujoco.MjModel.from_xml_string(XML)
    sim = mujoco.MjData(mj)
    sim.qpos[:] = [0.2, 0.2]
    controller = BoundedController(np.zeros(2), np.ones(2))
    if policy_hz < 1 or 100 % policy_hz:
        raise ValueError("policy_hz must divide controller rate 100")
    stride = 100 // policy_hz
    goal, target, last = np.array([0.8, 0.7]), sim.qpos.copy(), 0
    yy, xx = np.mgrid[:32, :32]
    latency, trajectory = [], []
    for tick in range(ticks):
        if tick % stride == 0:
            frame = np.zeros((3, 32, 32), np.float32)
            frame[0] = (xx - sim.qpos[0] * 31) ** 2 + (yy - sim.qpos[1] * 31) ** 2 < 8
            frame[1] = (xx - goal[0] * 31) ** 2 + (yy - goal[1] * 31) ** 2 < 8
            batch = {
                "image": torch.tensor(frame)[None],
                "state": (torch.tensor(sim.qpos).float()[None] - stats["state_mean"])
                / stats["state_std"],
            }
            start = time.perf_counter()
            with torch.no_grad():
                target = (
                    model.sample(batch)[0, 0] * stats["action_std"] + stats["action_mean"]
                ).numpy()
            latency.append((time.perf_counter() - start) * 1000)
            last = tick
        next_position = controller.step(sim.qpos.copy(), target, (tick - last) * 0.01, 0.01)
        # Critically damped PD tracking rather than teleporting qpos.
        sim.qfrc_applied[:] = 100 * (next_position - sim.qpos) - 20 * sim.qvel
        mujoco.mj_step(mj, sim)
        trajectory.append(sim.qpos.copy().tolist())
    result = {
        "policy_hz": policy_hz,
        "controller_hz": 100,
        "ticks": ticks,
        "final_goal_distance": float(np.linalg.norm(sim.qpos - goal)),
        "policy_latency_ms": float(np.mean(latency)),
        "trajectory": trajectory,
        "scope": "educational synthetic reacher; no Helix reproduction or stability guarantee",
    }
    write_json(output, result)
    return result


def smoke() -> dict:
    import mujoco

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    return {
        "mujoco_version": mujoco.__version__,
        "time": float(data.time),
        "finite": bool(np.isfinite(data.qpos).all()),
    }


def record(output: str, episodes: int = 8, seed: int = 17) -> dict:
    """Record actual MuJoCo PD demonstrations; analytic top-view observations of sim state.

    No external robot asset, renderer, dataset or checkpoint is reused. These are reacher
    demonstrations, never PushT demonstrations. Physics advances ten 100-Hz steps per sample.
    """
    from pathlib import Path

    import mujoco

    from va_lab.datasets.core import manifest, read_arrays

    if episodes < 3:
        raise ValueError("Record at least three episodes")
    rng = np.random.default_rng(seed)
    mj = mujoco.MjModel.from_xml_string(XML)
    yy, xx = np.mgrid[:32, :32]
    frames = {key: [] for key in ["image", "state", "action", "episode"]}
    for episode in range(episodes):
        sim = mujoco.MjData(mj)
        sim.qpos[:] = rng.uniform(0.15, 0.4, 2)  # reset only, not policy execution
        goal = rng.uniform(0.6, 0.85, 2)
        for _ in range(30):
            image = np.zeros((3, 32, 32), np.float32)
            image[0] = (xx - sim.qpos[0] * 31) ** 2 + (yy - sim.qpos[1] * 31) ** 2 < 8
            image[1] = (xx - goal[0] * 31) ** 2 + (yy - goal[1] * 31) ** 2 < 8
            target = np.clip(sim.qpos + 0.15 * (goal - sim.qpos), 0, 1)
            frames["image"].append(image)
            frames["state"].append(sim.qpos.copy().astype(np.float32))
            frames["action"].append(target.astype(np.float32))
            frames["episode"].append(episode)
            for _ in range(10):
                sim.qfrc_applied[:] = 100 * (target - sim.qpos) - 20 * sim.qvel
                mujoco.mj_step(mj, sim)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in frames.items()})
    read_arrays(str(path))
    metadata = {
        "source": "original MuJoCo two-slide reacher PD demonstrations",
        "episodes": episodes,
        "seed": seed,
        "fps": 10,
        "observation": "analytic top view of actual simulation state and goal",
        "action_schema": "absolute position in [0,1]^2",
    }
    manifest(path, metadata)
    return metadata
