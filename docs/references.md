# Official public references and licenses

- [OpenArm dataset v2.1](https://huggingface.co/datasets/enactic/openarm-2-cell-pick_up_cube_mujoco-lerobot/tree/v2.1),
  Apache-2.0; immutable SHA c5f7d41a37f9d9456b74eff673870997545c2d25.
- [OpenArm MuJoCo](https://github.com/enactic/openarm_mujoco), Apache-2.0;
  pinned a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c; v2/cell/demo.xml and JointResolver inspected.
- [OpenArm Dataset tooling](https://github.com/enactic/openarm_dataset), Apache-2.0;
  pinned 58af735f600bd4f907fb12ac6ca30c76c8524329; v2.1 conversion/name conventions inspected.
- [LeRobot 0.3.3](https://github.com/huggingface/lerobot/tree/v0.3.3), Apache-2.0;
  official LeRobotDataset and ACTPolicy are imported from the installed distribution.
- [Official PyTorch previous-version commands](https://pytorch.org/get-started/previous-versions/)
  for torch 2.7.1 / torchvision 0.22.1 CUDA 11.8 wheels.
- [ACT paper](https://arxiv.org/abs/2304.13705), [authors code](https://github.com/tonyzhaozh/act).
- [Diffusion Policy](https://arxiv.org/abs/2303.04137), [DDPM](https://arxiv.org/abs/2006.11239),
  [DDIM](https://arxiv.org/abs/2010.02502), [Flow Matching](https://arxiv.org/abs/2210.02747).
- [APT authors](https://github.com/xukechun/APT), [paper](https://arxiv.org/abs/2606.12366).
- [Official LingBot-VA documentation](https://huggingface.co/docs/lerobot/main/en/lingbot_va).
- [Figure Helix 02](https://www.figure.ai/news/helix-02), company-reported proprietary system.

Original laboratory code stays MIT. Official upstream assets are not relicensed or committed
into this repository. Download scripts preserve the upstream checkout and its Apache-2.0
license notices. Dataset metadata/provenance are recorded; raw images and weights are ignored.
Educational APT/LingBot/Helix components are not claimed to reproduce official systems.

Research-method references rechecked 2026-11-08: [statistical uncertainty in few-run experiments](https://arxiv.org/abs/2108.13264).
The laboratory implements its own crossed seed/episode bootstrap, not the complete rliable benchmark.
APT, LingBot-VA and Helix references describe upstream systems; their architecture and scale are
not attributed to the educational prototypes. See [methodology](research_methodology.md).
