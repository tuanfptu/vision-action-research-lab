"""Independent official OpenArm v2 Cell simulation; exact dataset scene NOT VERIFIED."""

import time
from pathlib import Path

import numpy as np
import torch

from va_lab.controllers.hierarchy import BoundedController, TaskGoal, TemporalEnsemble
from va_lab.datasets.core import manifest
from va_lab.datasets.public import CAMERAS, JOINT_NAMES
from va_lab.simulation.assets import acquire_assets
from va_lab.training.engine import load_checkpoint
from va_lab.utils.runtime import seed_all, write_json


def resolve_mapping(model) -> dict:
    import mujoco

    qpos, ctrl, joints, names = [], [], [], []
    for feature in JOINT_NAMES:
        side, joint = feature.removesuffix(".pos").split("_", 1)
        suffix = "finger_joint1" if joint == "gripper" else joint
        name = f"openarm_{side}_{suffix}"
        actuator = f"{side}_finger1_ctrl" if joint == "gripper" else f"{side}_{joint}_ctrl"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        if jid < 0 or aid < 0 or int(model.actuator_trnid[aid, 0]) != jid:
            raise ValueError(f"Unverified joint/actuator mapping: {feature}")
        # Position servos must have nonzero position gain and negative position bias.
        if model.actuator_gainprm[aid, 0] <= 0 or model.actuator_biasprm[aid, 1] >= 0:
            raise ValueError(f"Actuator is not a position servo: {actuator}")
        qpos.append(int(model.jnt_qposadr[jid]))
        ctrl.append(aid)
        joints.append(jid)
        names.append(name)
    lower = np.maximum(model.jnt_range[joints, 0], model.actuator_ctrlrange[ctrl, 0])
    upper = np.minimum(model.jnt_range[joints, 1], model.actuator_ctrlrange[ctrl, 1])
    if np.any(lower >= upper) or len(set(ctrl)) != 16:
        raise ValueError("Invalid controller limits or duplicate actuator")
    return {
        "qpos": np.array(qpos),
        "ctrl": np.array(ctrl),
        "joint_ids": np.array(joints),
        "joint_names": names,
        "lower": lower,
        "upper": upper,
    }


class OpenArmEnv:
    """30-Hz target interface over 1-kHz physics. No implicit scene-equivalence claim."""

    def __init__(
        self,
        asset_root: str = "assets/upstream",
        cameras: list[str] | None = None,
        image_size: int = 32,
        render_images: bool = True,
    ):
        import mujoco

        acquire_assets(asset_root)
        self.goal = TaskGoal()
        self.mj = mujoco
        self.scene = Path(asset_root).resolve() / "openarm_mujoco/v2/cell/demo.xml"
        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.data = mujoco.MjData(self.model)
        self.mapping = resolve_mapping(self.model)
        self.cameras = cameras or ["ceiling", "wrist_right"]
        if any(c not in CAMERAS for c in self.cameras):
            raise ValueError("Unknown OpenArm camera")
        self.size = image_size
        self.renderer = None
        self.render_images = render_images
        self.dt = float(self.model.opt.timestep)
        self.controller = BoundedController(
            self.mapping["lower"], self.mapping["upper"], timeout=0.25, max_speed=1.0
        )
        self.target = np.zeros(16)
        self.last_command_time = 0.0
        self.success_ticks = 0
        self.clipped_values = 0
        self.invalid_commands = 0
        self.cube = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "orange_cube")
        self.reset()

    def reset(self, seed: int = 7, initial_state: np.ndarray | None = None) -> dict:
        self.mj.mj_resetData(self.model, self.data)
        home = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_KEY, "home")
        if home >= 0:
            self.mj.mj_resetDataKeyframe(self.model, self.data, home)
        if initial_state is not None:
            q = np.asarray(initial_state, float)
            if q.shape != (16,) or not np.isfinite(q).all():
                raise ValueError("Initial state must be finite ordered 16-vector")
            q = np.clip(q, self.mapping["lower"], self.mapping["upper"])
            self.data.qpos[self.mapping["qpos"]] = q
            for side in ["right", "left"]:
                j1 = self.model.joint(f"openarm_{side}_finger_joint1").qposadr[0]
                j2 = self.model.joint(f"openarm_{side}_finger_joint2").qposadr[0]
                self.data.qpos[j2] = self.data.qpos[j1]
        # Independent reset protocol; original dataset object/reset distribution is unknown.
        rng = np.random.default_rng(seed)
        cube_joint = int(self.model.body_jntadr[self.cube])
        cube_qpos = int(self.model.jnt_qposadr[cube_joint])
        self.data.qpos[cube_qpos : cube_qpos + 2] += rng.uniform(-0.02, 0.02, 2)
        # Hold every actuator, including lifter; policy never controls or slices the lifter.
        for aid in range(self.model.nu):
            jid = int(self.model.actuator_trnid[aid, 0])
            self.data.ctrl[aid] = self.data.qpos[self.model.jnt_qposadr[jid]]
        self.mj.mj_forward(self.model, self.data)
        self.target = self.data.qpos[self.mapping["qpos"]].copy()
        self.command = self.target.copy()
        self.last_command_time = float(self.data.time)
        self.success_ticks = 0
        self.clipped_values = 0
        self.invalid_commands = 0
        self.next_observation_time = 0.0
        self.baseline_cube_z = 1.025  # table top 1.005 + cube half-size .020
        return self.get_observation()

    def get_observation(self) -> dict:
        out = {
            "state": self.data.qpos[self.mapping["qpos"]].copy(),
            "timestamp": float(self.data.time),
        }
        if self.render_images:
            out["image"] = np.stack([self.render(c) for c in self.cameras])
        return out

    def render(self, camera: str = "ceiling") -> np.ndarray:
        if camera not in self.cameras:
            raise ValueError("Camera not selected")
        if self.renderer is None:
            self.renderer = self.mj.Renderer(self.model, height=self.size, width=self.size)
        self.renderer.update_scene(self.data, camera=f"camera_{camera}")
        return self.renderer.render().transpose(2, 0, 1).copy()

    def step(self, action: np.ndarray | None) -> tuple[dict, dict]:
        if action is not None:
            a = np.asarray(action, float)
            if a.shape != (16,):
                raise ValueError("Expected every ordered joint target, shape (16,)")
            if not np.isfinite(a).all():
                self.invalid_commands += 1
                self.target = self.data.qpos[self.mapping["qpos"]].copy()
            else:
                clipped = np.clip(a, self.mapping["lower"], self.mapping["upper"])
                self.clipped_values += int(np.count_nonzero(clipped != a))
                self.target = clipped
            self.last_command_time = float(self.data.time)
        # Fractional 1000/30 ratio: alternate 33/34 physics steps, rather than drifting at 30.3 Hz.
        self.next_observation_time += 1 / 30
        while self.data.time + 1e-10 < self.next_observation_time:
            q = self.data.qpos[self.mapping["qpos"]].copy()
            age = float(self.data.time) - self.last_command_time
            if age > self.controller.timeout:
                self.command = q.copy()
            target = self.controller.step(self.command, self.target, age, self.dt)
            self.command = target.copy()
            self.data.ctrl[self.mapping["ctrl"]] = target
            self.mj.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all():
                raise FloatingPointError("Nonfinite simulation state")
        lifted = (
            float(self.data.xpos[self.cube, 2]) > self.baseline_cube_z + self.goal.lift_height_m
        )
        self.success_ticks = self.success_ticks + 1 if lifted else 0
        state = self.data.qpos[self.mapping["qpos"]]
        return self.get_observation(), {
            "joint_tracking_rmse_rad": float(np.sqrt(np.mean((state - self.target) ** 2))),
            "command_clipped_values": self.clipped_values,
            "invalid_commands": self.invalid_commands,
            "success": self.check_success(),
            "cube_z_m": float(self.data.xpos[self.cube, 2]),
        }

    def check_success(self) -> bool:
        """Project-specific cube lift >=8 cm for 10 consecutive observation ticks (~.33 s)."""
        return self.success_ticks >= self.goal.hold_ticks

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def smoke(render: bool = False) -> dict:
    env = OpenArmEnv(render_images=render)
    try:
        q = env.get_observation()["state"]
        obs, info = env.step(q)
        return {
            "status": "TESTED",
            "mujoco_version": env.mj.__version__,
            "nq": env.model.nq,
            "nu": env.model.nu,
            "camera_count": env.model.ncam,
            "action_shape": list(q.shape),
            "finite": bool(np.isfinite(obs["state"]).all()),
            "time": obs["timestamp"],
            "joint_names": env.mapping["joint_names"],
            "limits": [env.mapping["lower"].tolist(), env.mapping["upper"].tolist()],
            "rendered_shape": list(obs["image"].shape) if render else None,
            "exact_demonstration_scene": "NOT VERIFIED",
            **info,
        }
    finally:
        env.close()


def rollout(
    checkpoint: str,
    output: str,
    episodes: int = 1,
    horizon: int = 1,
    ensemble: bool = False,
    max_steps: int = 60,
    allow_unverified_scene: bool = False,
    policy_hz: int = 30,
) -> dict:
    if not allow_unverified_scene:
        raise ValueError(
            "Exact demonstration scene NOT VERIFIED; pass explicit --allow-unverified-scene for an independent OpenArm experiment"
        )
    seed_all(7)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    device = cfg["device"]
    model, ckpt = load_checkpoint(checkpoint, device)
    if ckpt["provenance"].get("action_names") != JOINT_NAMES:
        raise ValueError("Checkpoint has no verified OpenArm joint schema")
    if horizon < 1 or horizon > cfg["chunk"] or (ensemble and horizon != 1):
        raise ValueError("Invalid execution horizon/ensemble combination")
    if policy_hz < 1 or 30 % policy_hz:
        raise ValueError("Policy rate must divide 30-Hz observation rate")
    if ensemble and policy_hz != 30:
        raise ValueError("Ensembling requires a new prediction each observation tick")
    stats = ckpt["stats"]
    env = OpenArmEnv(cameras=cfg["cameras"], image_size=cfg["image_size"])
    timings, tracking, successes = [], [], []
    try:
        for episode in range(episodes):
            obs = env.reset(10000 + episode)
            history = []
            cursor = 0
            cache = None
            success = False
            temporal = TemporalEnsemble()
            for tick in range(max_steps):
                history.append(obs["image"])
                history = history[-cfg.get("history", 2) :]
                if tick % (30 // policy_hz) == 0 and (cursor == 0 or ensemble or cache is None):
                    image = torch.tensor(obs["image"], device=device).float()[None] / 255
                    past = [history[0]] * (cfg.get("history", 2) - len(history)) + history
                    batch = {
                        "image": image,
                        "past": torch.tensor(np.stack(past), device=device).float()[None] / 255,
                        "state": (
                            torch.tensor(obs["state"], device=device).float()[None]
                            - stats["state_mean"].to(device)
                        )
                        / stats["state_std"].to(device),
                    }
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.no_grad():
                        cache = (
                            (
                                model.sample(batch, cfg.get("sampling_steps", 10))[0]
                                * stats["action_std"].to(device)
                                + stats["action_mean"].to(device)
                            )
                            .cpu()
                            .numpy()
                        )
                    timings.append((time.perf_counter() - start) * 1000)
                    cursor = 0
                action = (
                    temporal.update(tick, cache) if ensemble else cache[min(cursor, len(cache) - 1)]
                )
                obs, info = env.step(action)
                cursor = (cursor + 1) % horizon
                tracking.append(info["joint_tracking_rmse_rad"])
                success |= info["success"]
            successes.append(success)
        result = {
            "scope": "independent official OpenArm Cell scene; not dataset-equivalent benchmark",
            "goal": "lift cube >=8cm for 10 consecutive ticks",
            "success_rate": float(np.mean(successes)),
            "episodes": episodes,
            "max_steps": max_steps,
            "policy_hz": policy_hz,
            "controller_hz": 1000,
            "joint_tracking_rmse_rad": float(np.mean(tracking)),
            "policy_latency_ms": float(np.mean(timings)),
            "policy_deadline_misses": sum(t > 1000 / policy_hz for t in timings),
            "command_clipped_values": env.clipped_values,
            "device": device,
            "scene_compatibility": "NOT VERIFIED",
            "reset_protocol": "official home, cube XY jitter +/-2cm, episode seeds 10000+index",
            "execution_horizon": horizon,
            "ensemble": ensemble,
        }
        write_json(output, result)
        return result
    finally:
        env.close()


def record(output: str, episodes: int = 6, seed: int = 17, length: int = 30) -> dict:
    """Real OpenArm physics joint-excitation collection; NOT pick-up expert demonstrations."""
    env = OpenArmEnv()
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ["image", "state", "action", "episode", "timestamp", "frame"]}
    try:
        for e in range(episodes):
            obs = env.reset(seed + e)
            home = obs["state"].copy()
            phase = rng.uniform(0, 2 * np.pi, 16)
            for k in range(length):
                target = np.clip(
                    home + 0.03 * np.sin(k / 15 + phase), env.mapping["lower"], env.mapping["upper"]
                )
                for key, value in [
                    ("image", obs["image"]),
                    ("state", obs["state"]),
                    ("action", target),
                    ("episode", e),
                    ("timestamp", k / 30),
                    ("frame", k),
                ]:
                    rows[key].append(value)
                obs, _ = env.step(target)
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise FileExistsError("Refusing to overwrite collected dataset")
        np.savez_compressed(p, **{k: np.asarray(v) for k, v in rows.items()})
        metadata = {
            "source": "independent OpenArm joint-excitation collection; no manipulation expertise",
            "episodes": episodes,
            "frames": episodes * length,
            "fps": 30,
            "seed": seed,
            "cameras": env.cameras,
            "action_names": JOINT_NAMES,
            "state_names": JOINT_NAMES,
            "action_schema": "absolute joint and finger_joint1 position, radians",
            "scene_compatibility": "independent scene; exact demonstration environment NOT VERIFIED",
        }
        manifest(p, metadata)
        return metadata
    finally:
        env.close()
