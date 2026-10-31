# LAB 01 — Vision–Action Fundamentals

## 1. Mục tiêu học tập

Xây dựng policy π(a|I,s); phân biệt state-only và vision-conditioned; hiểu closed-loop feedback và covariate shift.

## 2. Kiến thức và môi trường cần có

Lab 00; supervised regression, gradient descent và CNN.

## 3. Nền tảng toán học

BC tối thiểu hóa L=Σ m·||πθ(I,s)−a||² / Σm, tính cả action dimensions. Encoder f(I), fusion h=ReLU(W[f(I),s]), head dự đoán K target. State-only bỏ f(I). Sai lệch distribution làm policy vào state ít thấy; lỗi tích lũy có thể tăng nhanh theo horizon. Offline MSE nhỏ không suy ra rollout success cao.

## 4. Kiến trúc và computational flow

CNN hai Conv stride 2 → AdaptivePool 4×4 → Linear → fusion state → MLP action head. Action được normalize khi train và denormalize trước env.step. Feedback lấy observation mới mỗi execution horizon.

## 5. Tensor shapes

Image [B,3,32,32] → [B,16,16,16] → [B,32,8,8] → [B,32,4,4] → [B,512] → [B,D]. Fusion [B,D+2] → [B,D]. Head [B,2K] → [B,K,2].

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
va-lab train --data data/synthetic.npz --output runs/lab01_bc --model bc
va-lab train --data data/synthetic.npz --output runs/lab01_state --model state
va-lab evaluate --checkpoint runs/lab01_bc/best.pt --data data/synthetic.npz
va-lab download --episodes 6
va-lab train --data data/pusht.npz --output runs/pusht_bc --model bc
va-lab rollout --checkpoint runs/pusht_bc/best.pt
```

## 9. Hành vi kỳ vọng và cách đọc kết quả

Loss hữu hạn và checkpoint load được. Synthetic goal thể hiện bằng kênh xanh trong ảnh; state-only không được goal nên bài toán có ambiguity. Không giả định vision luôn thắng với ít dữ liệu.

Không có giá trị performance kỳ vọng giả định. Đọc số đo đã chạy trong
[validation](../../docs/validation.md); không so training loss giữa objective khác nhau.

## 10. Bài tập thực nghiệm

Giữ split/seed/budget giống nhau; so MSE và rollout success riêng. Tăng số episode và epochs; kiểm tra ảnh bị che, observation perturbation và execution horizon.

## 11. Failure analysis

Action-space mismatch tạo hành vi sai dù loss giảm. MSE có thể lấy trung bình hai mode. Normalization dùng validation gây leakage. Ảnh 32×32 có thể mất chi tiết góc khối T.

## 12. Câu hỏi kiểm tra hiểu sâu

Tại sao state-only có thể đủ ở một task nhưng thiếu goal ở task khác? Tại sao open-loop error và closed-loop success không tương đương? Khi nào DAgger giải quyết distribution shift?

## 13. Tài liệu chính thức

PushT dataset and gym-pusht; Diffusion Policy benchmark paper. Các URL, version và ghi chú license ở [references](../../docs/references.md).
Đọc [limitations](../../docs/limitations.md) trước khi diễn giải một smoke experiment thành
kết quả nghiên cứu. Các claim của công trình gốc không phải kết quả của repository này.
