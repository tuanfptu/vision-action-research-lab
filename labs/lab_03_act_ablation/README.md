# LAB 03 — OpenArm ACT Ablation

## 1. Mục tiêu

Kiểm soát chunk, latent, camera sets, data budget và control horizon.

## 2. Prerequisites

Python/PyTorch, supervised learning và các lab trước. Dùng `.venv-gpu`, clear ROS PYTHONPATH
trong shell dự án. Dataset không cần ROS. Không đọc hoặc tái sử dụng previous robotics projects.

## 3. Nền tảng toán học

Paired comparison giữ episode split/seed và validation/test cố định. Tỷ lệ train_fraction chỉ lấy subset training; fit lại stats trên subset đó. Báo confidence interval khi đủ seeds.

## 4. Architecture và computational flow

Cartesian grid → fresh run dirs → best validation checkpoint → optional independent-scene rollout với horizon/ensemble.

## 5. Tensor shapes

N frame, V camera được chọn, B batch, K action chunk, H history, D feature width.
Cache image [N,V,3,R,R]; batch image [B,V,3,R,R], state [B,16]. CNN xử lý [B×V,3,R,R]
→ [B,V,D]; concatenate [B,V×D+16] → context [B,D]. Action target/prediction [B,K,16],
mask [B,K]. ACT posterior [B,K+1,D], μ/logvar/z [B,L], queries [B,K,D]. Generative input
[B,K×16+D+3]. Video history [B,H,V,3,R,R], future [B,K,V,3,16,16]. Mọi dimension state/action
được suy từ validated cache và lưu config; không hardcode 2D hoặc tùy ý reorder joints.

## 6. Dataset representation và semantics

30 episodes / 5,405 frames / 30Hz, v2.1 immutable SHA
c5f7d41a37f9d9456b74eff673870997545c2d25. Thứ tự: right joints1–7, right gripper,
left joints1–7, left gripper. Absolute position radians; gripper là signed finger_joint1
angle, không phải [0,1] opening ratio. Default views ceiling+wrist_right; all five views
được hỗ trợ. [Contract chi tiết](../../docs/openarm_contract.md) là nguồn mapping duy nhất.

## 7. Walkthrough implementation

Đọc datasets/public.py acquire → export_openarm, datasets/core.py read_arrays → episode_split
→ statistics → Windows. Đọc models/policies.py hoặc models/official.py, sau đó training/engine.py.
Training dùng zero_grad, loss.backward, accumulation, gradient clipping, optimizer.step;
checkpoints ghi stats, split, camera order, schema và hash. Evaluation eval()+no_grad(), không
posterior labels. Simulation resolve mapping bằng tên thay vì suy thứ tự qpos từ vector dataset.

## 8. Lệnh training / inference

```bash
source .venv-gpu/bin/activate
unset PYTHONPATH
export HF_HOME="$PWD/data/hf"
export MUJOCO_GL=egl
python scripts/ablate.py --output runs/lab03 --grid configs/models/act_smoke_ablation.yaml
```

Download cache lần đầu theo README, bỏ qua nếu cache đã tồn tại. Output run directory phải mới.

## 9. Expected behavior và evidence

Shapes đúng và loss/gradient hữu hạn là kiểm tra software path; không phải bằng chứng học task.
Chỉ báo số được ghi ở [validation](../../docs/validation.md). Report device, peak VRAM, episode
split, raw per-joint RMSE và camera names. Test episodes chỉ truy cập bằng evaluate --split test
sau model selection. Migration ban đầu chưa mở test; study 08/11/2026 đã đánh giá test sau
selection lock. Không coi split này là untouched cho việc tuning tiếp theo.

## 10. Bài tập

So state-only/vision với cùng split; đổi một yếu tố mỗi lần. So một/two/five cameras, K=1/4/8,
history=1/2/4, data budget=50/100% trong training pool. Plot từng joint và gripper, không chỉ
scalar aggregate. Đo sampling steps/latency và ít nhất ba seed khi đủ budget. Với simulation,
đối chiếu named limits, actual commands, tracking error, deadline misses và cube lift criterion.

## 11. Failure analysis và giới hạn

Camera count, K và data budget thay cả compute; cần báo thời gian/VRAM. Full multi-seed study vẫn PARTIAL.

Timestamps sai gây image/action misalignment; split theo frame gây leakage; double-normalization
làm target sai đơn vị. Constant joint std cần clamp và raw-RMSE kiểm tra. MSE thấp chưa bảo đảm
contact dynamics tốt. Không truyền normalized action trực tiếp vào actuator, không gọi
controller tracking thành công là successful learned manipulation.

## 12. Câu hỏi tự kiểm tra

Vì sao right gripper có giá trị âm mà không phải torque? Vì sao dataset action order khác
MJCF qpos order? Tại sao val/test không được fit normalization? Camera order đổi ảnh hưởng
fusion thế nào? K khác execution horizon ra sao? Điều gì đủ để chứng minh dataset/simulator
tương thích, và điều gì mới chỉ xác nhận offline regression?

## 13. Tài liệu chính thức

[Dataset OpenArm](https://huggingface.co/datasets/enactic/openarm-2-cell-pick_up_cube_mujoco-lerobot),
[OpenArm MuJoCo](https://github.com/enactic/openarm_mujoco),
[OpenArm Dataset](https://github.com/enactic/openarm_dataset),
[LeRobot v0.3.3](https://github.com/huggingface/lerobot/tree/v0.3.3).
Paper/implementation ACT, Diffusion/Flow, APT, LingBot và Helix được ghi ở
[references](../../docs/references.md), cùng license và ghi chú educational scope.

Khi đổi training-data budget, normalization statistics cũng đổi. Vì vậy normalized MSE giữa
các budget không cùng thang đo; dùng raw per-joint RMSE (radians) để so sánh các budget.

## 14. Thiết kế đối chứng và cách đọc uncertainty

Study mới có ba seed 7/17/27 và sáu thay đổi độc lập: K=1, K=8, latent=4, β=0, chỉ ceiling,
50% train episodes. Baseline ACT dùng lại, không train thêm một baseline khác cho từng ô.
Mọi split và learning budget nằm trong protocol. Đây là one-factor study; interaction và
full factorial grid chưa được thực nghiệm này giải quyết.

K khác nhau làm số target trong chunk khác nhau, nên bảng đối chiếu dùng **first-action MSE**
(+0) thay vì so mean của các horizon không tương ứng. Data budget khác thì std normalization
khác, nên phải so raw radian MSE. Camera count thay đổi cả input information và fusion weights;
đừng mô tả nó như chỉ một phép tắt sensor trên cùng model.

Paired difference giữ cùng episode và seed. Bootstrap resample hai trục seed/episode, không
resample overlapping windows như IID. Với bốn validation episodes, interval vẫn rất dễ lệch;
không biến một CI không qua 0 thành tuyên bố tổng quát sau nhiều exploratory comparisons.
Bài tập: viết lại giả thuyết có thể bị bác bỏ, endpoint chính và điều kiện dừng trước khi chạy.
Lệnh study và báo cáo: [reproduction](../../docs/reproduction.md), [results](../../docs/research_report.md).
