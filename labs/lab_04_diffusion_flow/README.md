# LAB 04 — Diffusion và Flow Matching

## 1. Mục tiêu học tập

Hiểu conditional multimodal generation và sự khác nhau giữa noise prediction với velocity prediction.

## 2. Kiến thức và môi trường cần có

Lab 01; Gaussian noise, stochastic processes và ODE cơ bản.

## 3. Nền tảng toán học

Diffusion: x_t=√ᾱ_t·a+√(1−ᾱ_t)·ε; L=E||εθ(x_t,t,c)−ε||² có padding mask. ᾱ là cumulative product của 1−β_t với 1000 bước training (sampling có thể ít hơn). DDIM suy x0 từ ε rồi chuyển sang ᾱ trước. Flow: x_t=(1−t)x_noise+t·a, velocity target v=a−x_noise; L=E||vθ−v||². Euler: x(t+Δ)=x(t)+Δvθ. Hai objective không có loss scale so sánh trực tiếp.

## 4. Kiến trúc và computational flow

Condition encoder giống BC. Flatten action chunk, concat condition và time features [t,sin πt,cos πt] → MLP → noise/velocity [B,K,2]. Inference khởi tạo noise và lặp sampling steps. Không dùng ground-truth actions ở inference.

## 5. Tensor shapes

Noise/clean/x_t [B,K,2]; t [B], embedding [B,3]; condition [B,D]; MLP input [B,2K+D+3]; output [B,2K] reshape [B,K,2].

## 6. Biểu diễn dữ liệu

Mỗi file NPZ chứa `image [N,3,32,32]` float trong [0,1], `state [N,2]`,
`action [N,2]`, `episode [N]`. Synthetic dùng tọa độ [0,1]; PushT dùng pixel coordinate
cho cả state và absolute target. Không truyền normalized action trực tiếp vào simulator.
Window có `valid [B,K]`; cuối episode lặp lại frame/action chỉ để padding và dùng mask.
Normalization chỉ fit trên train episodes. Chia frame ngẫu nhiên sẽ rò rỉ temporal context.
Manifest ghi SHA256, source và Hub revision; checkpoint kiểm tra hash khi đánh giá.

## 7. Walkthrough mã nguồn

Đọc `src/va_lab/datasets/core.py` theo thứ tự generate → read_arrays → episode_split →
statistics → Windows. Sau đó đọc model tương ứng trong `models/policies.py`,
`training/engine.py`, và `simulation/benchmark.py`. Train dùng AdamW, zero_grad,
loss.backward, clip_grad_norm, optimizer.step; JSONL ghi số thực, best.pt lưu model,
config, split và normalization. Evaluation load đúng checkpoint, `eval()` và `no_grad()`.
Không xem tensor random forward pass là bằng chứng policy có kỹ năng.

## 8. Lệnh training và inference


```bash
source .venv/bin/activate
unset PYTHONPATH
va-lab generate --output data/synthetic.npz
va-lab inspect --data data/synthetic.npz
```

```bash
va-lab train --data data/synthetic.npz --output runs/lab04_diffusion --model diffusion
va-lab train --data data/synthetic.npz --output runs/lab04_flow --model flow
va-lab train --data data/pusht.npz --output runs/pusht_flow --model flow
va-lab rollout --checkpoint runs/pusht_flow/best.pt
```

## 9. Hành vi kỳ vọng và cách đọc kết quả

Forward/backward và sampler tạo output hữu hạn. Đây là MLP nhỏ, không phải temporal U-Net của Diffusion Policy. Với ít epochs mode coverage chưa được chứng minh.

Không có giá trị performance kỳ vọng giả định. Đọc số đo đã chạy trong
[validation](../../docs/validation.md); không so training loss giữa objective khác nhau.

## 10. Bài tập thực nghiệm

Giữ observation cố định, sample ≥100 lần và plot endpoints để kiểm tra mode coverage. Thay sampling_steps 1/5/10/20/100; đo quality/latency. So BC, ACT và generators trên cùng dataset/split.

## 11. Failure analysis

MSE point prediction làm trung bình mode; generative model vẫn có thể mode collapse. Euler ít bước tạo discretization error. Noise schedule không đủ phá signal làm prior mismatch. Padded tokens cần xem xét nếu nghiên cứu horizon dài.

## 12. Câu hỏi kiểm tra hiểu sâu

Tại sao noise prediction học distribution thay vì một action duy nhất? DDIM và DDPM khác nhau ở randomness như thế nào? Velocity field phụ thuộc time ra sao?

## 13. Tài liệu chính thức

DDPM, DDIM, Flow Matching và Diffusion Policy papers trong docs/references.md. Các URL, version và ghi chú license ở [references](../../docs/references.md).
Đọc [limitations](../../docs/limitations.md) trước khi diễn giải một smoke experiment thành
kết quả nghiên cứu. Các claim của công trình gốc không phải kết quả của repository này.
