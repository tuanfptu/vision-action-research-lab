# LAB 04 — OpenArm Diffusion / Flow

## 1. Mục tiêu

Sinh phân phối joint action chunks, đo inference steps và multimodality.

## 2. Prerequisites

Python/PyTorch, supervised learning và các lab trước. Dùng `.venv-gpu`, clear ROS PYTHONPATH
trong shell dự án. Dataset không cần ROS. Không đọc hoặc tái sử dụng previous robotics projects.

## 3. Nền tảng toán học

Diffusion x_t=√ᾱ_t a+√(1−ᾱ_t)ε, học εθ; 1000 noise steps. DDIM có thể sample ít bước. Flow x_t=(1−t)ε+ta, v*=a−ε, Euler x←x+Δt·vθ. Cả hai condition trên ảnh và state.

## 4. Architecture và computational flow

Flatten [B,K,16] concat feature+time → noise/velocity MLP → chunk. Không cắt chỉ một arm hoặc gripper.

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
va-lab train --model diffusion --output runs/lab04_diffusion
va-lab train --model flow --output runs/lab04_flow
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

Không so loss DDPM, flow, CVAE trực tiếp. Point MSE chưa đo mode coverage; đầu ra stochastic cần nhiều samples và seeds.

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

## 14. Kiểm tra sampler trước khi tin training loss

Trong diffusion, noise target có variance xấp xỉ 1 dù action đã normalize. Khi noise loss giảm,
clean-action error vẫn có thể lớn vì bước suy x0 chia cho sqrt(alpha_bar). Timestep cuối có
alpha_bar nhỏ; một noise estimate lệch nhỏ có thể tạo joint target ngoài khoảng demonstration.
Không clip output sau khi xem test để che failure; nếu cần clipping, đăng ký protocol mới.

Trong flow, thứ tự đường đi là noise tại t=0, action tại t=1, nên velocity target là action−noise.
Đảo dấu target mà vẫn Euler tiến sẽ đi sai hướng. `sampling_steps` tác động numerical integration,
không đổi trained weights; nhiều bước không sửa được field đã học sai.

Sampler sensitivity dùng 1/5/10/20/50 bước trên cùng checkpoint cho mỗi seed và validation
episode. Mỗi window có một stochastic sample, không chọn sample gần label nhất. Latency là batch
latency CUDA-synchronized sau warm-up, không gọi đó là per-action control latency. Bài tập:
phân biệt noise prediction error, sampled action error và manipulation success; đề xuất phép đo
mode coverage cho task có hai đường đi hợp lệ. Xem [phân tích](../../docs/research_methodology.md#5-diffusion-và-flow-objective-khác-cùng-action-contract).
