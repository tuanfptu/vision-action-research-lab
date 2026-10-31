# LAB 07 — Hierarchical Visuomotor Control

## 1. Mục tiêu học tập

Hiểu goal interface, learned VA targets, low-level feedback và independent control rates.

## 2. Kiến thức và môi trường cần có

Lab 01; PD control, joint limits và discrete-time execution.

## 3. Nền tảng toán học

System 2 chọn goal g. System 1 đề xuất target q* từ image/state mỗi 1/fpolicy giây. System 0 theo target ở 100 Hz. Bounded interpolation q_next=clip(q+clip(q*−q,±vmaxΔt),limits). PD force=Kp(q_next−q)−Kd·qdot. Watchdog quá timeout giữ vị trí hiện tại. Đây là engineering analogy; Helix System 0 chính thức là learned controller.

## 4. Kiến trúc và computational flow

Original MuJoCo model có hai slide joints, không cần asset bên ngoài. Synthetic observation vẽ agent đỏ và goal xanh; VA policy đã train trên synthetic đề xuất absolute targets. PD chạy physics mj_step; không teleport qpos. Lịch được tính theo simulated ticks, chưa mô hình hóa realtime IPC deadline.

## 5. Tensor shapes

Goal [2], observation [1,3,32,32], state [1,2], action [1,K,2]; chọn [0,0] → target [2]. Controller xử lý q,qdot,target [2] mỗi 0.01 s. Policy 10 Hz cập nhật mỗi 10 controller ticks.

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
va-lab train --data data/synthetic.npz --output runs/lab07_bc --model bc
va-lab hierarchy --checkpoint runs/lab07_bc/best.pt --policy-hz 10 --output runs/hierarchy10.json
va-lab hierarchy --checkpoint runs/lab07_bc/best.pt --policy-hz 20 --output runs/hierarchy20.json
```

## 9. Hành vi kỳ vọng và cách đọc kết quả

JSON ghi trajectory, goal distance, latency thực của forward và simulated rates. PARTIAL: không phải Panda, humanoid hoặc proprietary Helix; không có stability theorem hay hardware safety certification.

Không có giá trị performance kỳ vọng giả định. Đọc số đo đã chạy trong
[validation](../../docs/validation.md); không so training loss giữa objective khác nhau.

## 10. Bài tập thực nghiệm

Thay policy_hz 5/10/20/50; plot tracking error. Unit-test NaN target, stale target và joint bounds. Chèn delay/jitter ở simulator wrapper, kiểm tra watchdog và report deadline misses. Bổ sung Panda chỉ khi có license và action adapter rõ.

## 11. Failure analysis

Policy tốt nhưng controller không match action schema vẫn thất bại. Interpolation bound trên setpoint không phải bound tuyệt đối của physics q do inertia. Gain cao hoặc delay gây oscillation. Watchdog simulated age không bắt wall-clock stalls.

## 12. Câu hỏi kiểm tra hiểu sâu

Tại sao policy và controller chạy hai tần số? Khi nào target interpolation gây lag? Vì sao limit setpoint không đủ để chứng minh joint không vượt giới hạn?

## 13. Tài liệu chính thức

Figure Helix 02 announcement; MuJoCo official documentation. Chỉ học nguyên lý hierarchy. Các URL, version và ghi chú license ở [references](../../docs/references.md).
Đọc [limitations](../../docs/limitations.md) trước khi diễn giải một smoke experiment thành
kết quả nghiên cứu. Các claim của công trình gốc không phải kết quả của repository này.
