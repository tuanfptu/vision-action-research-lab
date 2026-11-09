# Research scope and limitations — updated 2026-11-08

- IMPLEMENTED: official v2.1 acquisition, exact SHA, ordered 16-joint schema, five available
  camera views, synchronized two-view export, train/validation/test episode split, train-only
  statistics, generic state/action dimensions and multi-camera policies.
- IMPLEMENTED: GPU-required train/evaluate, gradient accumulation, exact checkpoint reload,
  per-joint raw RMSE, video PSNR, actual memory/runtime logging, controlled ablation scripts.
- IMPLEMENTED: educational and official ACT, BC/state, diffusion/flow, multi-view video GRU.
- IMPLEMENTED: named OpenArm joint/controller adapter, reset/step/observation/render/success,
  interpolated setpoints, clipping, NaN rejection and simulated watchdog, independent Cell assets.
- NOT VERIFIED: exact demonstration MJCF commit, cameras, gains, lifter pose, object reset
  distribution and original success rule. Offline training compatibility does not certify
  closed-loop reconstruction. Independent rollouts require explicit acknowledgment.
- IMPLEMENTED / TESTED: GPU-resident vectorized windows checked against the reference loader,
  seven policy families × three seeds, 18 one-factor ACT controls, episode-level raw metrics,
  paired crossed-bootstrap intervals, immutable selection/test records, sampler sensitivity,
  video persistence baseline, three-seed same-task transfer, nine controller fault cases and
  thirty independent-scene 10-second rollout trials. See research_report.md and its raw artifacts.
- PARTIAL: full factorial Lab 03 grid, interaction effects, adequate statistical power and
  cross-task/generalization study. Four validation/test episodes make intervals descriptive.
  Half-data uses fewer optimizer updates at equal epochs; camera changes also change parameters.
- PARTIAL: Lab 05 uses disjoint episodes from one task and only one target validation episode. No evidence of cross-task transfer,
  negative-transfer robustness or sample-efficiency benefit. The added simulation collector
  produces joint-excitation data, not expert pick-up trajectories. Next: public teleoperation,
  task diversity, success labeling, task-balanced sources and reserved final test tasks.
  Source/target state normalization changes the input to transferred fusion weights; the
  current transfer arms do not affine-compensate it. Do not interpret their gap causally.
- PARTIAL: Lab 06 is a small GRU + frame/action heads, not LingBot latent dual-stream diffusion,
  autoregressive planning or a robot foundation world model.
- PARTIAL: Lab 07 has an explicit single-task TaskGoal dispatch interface, not a trained high-level planner. It uses explicit position servos, not proprietary learned Helix System 0.
  Watchdog uses simulated command age; invalid clock/state/shape and nonfinite targets are tested. Latency misses are measured, but full real-time IPC
  stalls, network failure and hardware safety are not certified.
- Thirty episodes from one task are a small sample. Near-constant left gripper coordinates
  make normalized MSE sensitive; inspect all 16 raw joint RMSEs. Training loss scales between
  BC/CVAE/diffusion/flow are different; compare validation prediction and behavior separately.
- No old local data/checkpoint is deleted. Schema-incompatible artifacts remain historical.
  Unrelated repositories, drivers, global packages, ROS and system Python are untouched.

Executed checks and measured results are listed only in validation.md. A feature being
IMPLEMENTED does not imply it has passed an integration test or achieved manipulation success.

The original final-test episodes were evaluated after a locked validation selection on
2026-11-08; they are now a used test set. Do not describe later tuning on these episodes as
untouched evaluation. Independent-scene successes were 0/10 in each of three cases; offline
ACT error improvements did not establish successful manipulation. The video head was worse
than persistence and the diffusion sampler retained very large error; neither failure is hidden.
The simulator's clipping counter resets each episode; followup reports explicitly label the
reported count as belonging to the last episode, rather than a total across the trial batch.
