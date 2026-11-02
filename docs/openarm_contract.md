# OpenArm dataset → policy → controller contract

## Verified public data

Repository: enactic/openarm-2-cell-pick_up_cube_mujoco-lerobot, Apache-2.0, v2.1,
SHA c5f7d41a37f9d9456b74eff673870997545c2d25. Metadata declares OpenArm 2 bimanual,
30 episodes / 5,405 frames / 30 FPS, task `Pick up the cube.` The original card allocates
0:24 train and 24:30 val. This lab deliberately creates a new deterministic 22/4/4 episode
split to preserve an untouched final test set; never present its metrics as the card's split.

## Joint order, units and grippers

| Dataset indices | Exact feature names | Meaning and simulator mapping |
|---|---|---|
| 0–6 | right_joint1.pos … right_joint7.pos | Absolute radians → openarm_right_joint1 … joint7 |
| 7 | right_gripper.pos | Signed finger_joint1 angle, radians → openarm_right_finger_joint1 |
| 8–14 | left_joint1.pos … left_joint7.pos | Absolute radians → openarm_left_joint1 … joint7 |
| 15 | left_gripper.pos | Signed finger_joint1 angle, radians → openarm_left_finger_joint1 |

The official public JointResolver explicitly defines right[0:8] followed by left[8:16].
Right finger angles are mostly negative; left mostly positive. No sign flip, normalization to
[0,1], arbitrary slicing or index-based mapping is inferred. A negative angle is not a torque.
The second finger is equality constrained, not a seventeenth policy action. The Cell lifter
is an extra simulator actuator held at reset position; it is absent from this dataset vector.

Mapping resolves each joint name to `jnt_qposadr` and each actuator name to actuator ID,
checks the actuator transmission targets the same joint and verifies position-servo gain/bias.
The valid command interval is the intersection of joint range and actuator ctrlrange. Dataset
values can exceed this public model's limits slightly; clipping is counted and reported.
This discrepancy is evidence that exact simulation equivalence has NOT been established.

## Camera-to-policy mapping

| Dataset suffix | Raw H×W×C | Public MJCF camera |
|---|---|---|
| ceiling | 600×960×3 | camera_ceiling |
| head_left | 720×1280×3 | camera_head_left |
| head_right | 720×1280×3 | camera_head_right |
| wrist_left | 600×960×3 | camera_wrist_left |
| wrist_right | 600×960×3 | camera_wrist_right |

Default cached views: ceiling, wrist_right, in that order. Export all five with `--cameras
ceiling head_left head_right wrist_left wrist_right`. Original videos remain untouched; caches
resize RGB to configured square resolution (default 32), so aspect ratio changes are explicit.
Every selected view retains its name and index in manifest/config/checkpoint. Shared-CNN fusion
concatenates per-view embeddings; it is order-sensitive. Official ACT uses separate feature keys.
A two-view checkpoint cannot silently accept a reordered or five-view cache. The current independent
simulator renders square views directly; this changes projection relative to resizing the original
rectangular dataset frames. This is an additional documented visual domain gap, not a claimed match.

## Synchronization, windows and statistics

Validate numeric frame_index 0..T−1, increasing timestamp k/30, exact per-video frame counts
and video PTS matching numeric timestamp within 1 ms. Export uses the official LeRobotDataset
for feature/numeric loading; sequential PyAV decoding avoids repeatedly seeking full-resolution
videos during every training sample. Official LeRobot item decoding is independently tested.
Pinned 0.3.3 has a SHA-as-version fallback: exact files are prefetched before constructing it.

Chunks include actions t..t+K−1. History clips to episode start; future images use t+1..t+K.
Tail repetition is only padding; valid and future_valid masks exclude nonexistent targets.
All 16 joint statistics fit **only actual training-budget episodes**. No upstream global mean/std
is used for training, because it may include validation/test. Upstream stats are retained with
a checksum for inspection. Standard deviations clamp at 1e−4 (a near-constant gripper can
otherwise make normalized errors unstable); report raw per-joint RMSE alongside normalized MSE.

Normalize state/action with (x−mean)/std; images uint8 → float/255. Inference must reverse
a_norm*action_std+action_mean before controller clipping. Official ACT uses IDENTITY internal
normalization because this outer pipeline already normalizes state/action. Double normalization
would be an error. Current temporal video model repeats the current state alongside historical
image embeddings; it does not claim to consume historical joint velocities.

## Control and evidence boundary

Physics timestep 1 ms; observations/actions 30 Hz use alternating 33/34 steps to avoid drift.
Low-level interpolation updates each physics step with bounded setpoint velocity, followed
by official MJCF position servos. NaN commands hold state; stale commands after 250 ms trigger
the watchdog. This is a software safety mechanism, not a hardware safety certification.

The independent reset uses the official home keyframe and cube XY jitter ±2 cm with a recorded
episode seed. This is a project-defined distribution, not the original demonstration reset.

Independent task criterion: cube center > table top + half cube height + 8 cm for 10 consecutive
observation ticks (~0.33 s). It is project-defined, not an extracted official evaluation rule.
No dataset scene SHA, exact reset pose distribution or original success rule was supplied in
public metadata. Match joint schema does not prove camera/dynamics/task equivalence. Always
label offline regression and independent closed-loop behavior separately.

Khi đổi training-data budget, normalization statistics cũng đổi. Vì vậy normalized MSE giữa
các budget không cùng thang đo; dùng raw per-joint RMSE (radians) để so sánh các budget.
