# Official sources and attribution

Sources inspected on 2026-10-31. Project implementations are original reduced educational
models; no upstream source was copied. Dependency installation does not transfer upstream
licensing to this repository. Preserve upstream notices if adapting their code later.

- [LeRobot 0.3.3 release metadata](https://pypi.org/project/lerobot/0.3.3/): Python >=3.10;
  torch >=2.2.1,<2.8, torchvision >=0.21,<0.23; Apache-2.0 framework.
- [Versioned ACT config](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/policies/act/configuration_act.py)
  and [model source](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/policies/act/modeling_act.py).
- [Current LeRobot docs](https://huggingface.co/docs/lerobot/index): newer releases require
  different Python/PyTorch versions. Do not combine current CLI instructions with the pinned API.
- [Public PushT image dataset](https://huggingface.co/datasets/lerobot/pusht_image): MIT as
  declared by its dataset card; v2.1 tag (default main has migrated to v3), 206 episodes, 25,650 frames at 10 Hz.
  Exporter retrieves an immutable revision and only requested episodes.
- [Official PushT environment](https://github.com/huggingface/gym-pusht): target agent XY,
  not force, torque, joint angle, velocity, or robot end-effector displacement.
- [ACT paper: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
  and [authors' implementation](https://github.com/tonyzhaozh/act).
- [Diffusion Policy paper](https://arxiv.org/abs/2303.04137) and
  [official code](https://github.com/real-stanford/diffusion_policy).
- [DDPM](https://arxiv.org/abs/2006.11239), [DDIM](https://arxiv.org/abs/2010.02502),
  [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
- [APT authors' repository](https://github.com/xukechun/APT) and
  [paper](https://arxiv.org/abs/2606.12366): action expert pretraining for VLA instruction
  generalization. This lab isolates supervised VA encoder transfer; it does not establish
  instruction generalization or reproduce the full action expert pretraining recipe.
- [LingBot-VA official integration](https://huggingface.co/docs/lerobot/main/en/lingbot_va):
  video–action world modeling; the small lab joint predictor is conceptually related only.
- [Figure Helix 02 announcement](https://www.figure.ai/news/helix-02): company-reported
  full-body system; its System 0 is learned, unlike this lab's explicit PD controller.
- [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html): Apache-2.0 engine.
  The included simple MJCF is original. [Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  is a future optional source; inspect each model's own license before distribution.
