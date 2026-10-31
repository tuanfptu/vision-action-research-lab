# LAB 03 — ACT Ablation Study

## 1. Mục tiêu học tập

Thiết kế controlled comparison; đọc success, latency và smoothness mà không nhầm tương quan với nguyên nhân.

## 2. Kiến thức và môi trường cần có

Lab 02; experiment design, mean/variance và confidence intervals.

## 3. Nền tảng toán học

Một paired experiment giữ data split, optimizer, training budget và seed giống nhau; chỉ thay yếu tố cần đo. Success estimate p̂=Σsuccess/n có uncertainty lớn khi n nhỏ. Action smoothness ở đây mean squared first difference, chưa phải jerk hoặc chứng minh stability. Batch latency và policy-call latency là hai đại lượng khác nhau.

## 4. Kiến trúc và computational flow

scripts/ablate.py dùng Cartesian grid chunk×latent×beta, train mỗi job và có thể rollout horizon×ensemble. scripts/report.py chỉ tổng hợp evaluation JSON thực. Có file config nhưng không coi mọi config trong repository là đã chạy.

## 5. Tensor shapes

K thay đổi action tensor [B,K,2], positional embedding [1,K+1,D], query [1,K,D]. L thay đổi μ/z [B,L]. h chỉ thay lịch execution, không đổi output shape. Ensemble buffer giữ các chunk còn dự đoán cho timestep hiện tại.

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
python scripts/ablate.py --data data/pusht.npz --output runs/lab03 --rollouts
python scripts/report.py runs/lab03 --output runs/lab03/report.md
```

## 9. Hành vi kỳ vọng và cách đọc kết quả

Mỗi job có config, split, metrics, checkpoint, evaluation và rollout JSON riêng. PARTIAL: chưa tự động hóa image encoder/history/data-budget/unseen-state sweeps; chưa có multi-seed study đầy đủ.

Không có giá trị performance kỳ vọng giả định. Đọc số đo đã chạy trong
[validation](../../docs/validation.md); không so training loss giữa objective khác nhau.

## 10. Bài tập thực nghiệm

Bắt đầu grid nhỏ; sau đó ≥3 seeds, final untouched episodes và confidence interval. Đo batch-1 p50/p95 sau warmup. So data fractions 10/25/50/100% theo episode và giữ validation cố định.

## 11. Failure analysis

Sử dụng validation để chọn architecture rồi báo cùng tập đó là test dẫn đến optimistic bias. Thay K cũng thay compute và action targets; cần báo cả wall-clock budget. Test rollout quá ít tạo ranking ngẫu nhiên.

## 12. Câu hỏi kiểm tra hiểu sâu

Tại sao phải dùng cùng episode split? Vì sao K lớn cần kiểm soát compute? Một success-rate 0/3 có chứng minh model vô dụng không?

## 13. Tài liệu chính thức

ACT original paper and versioned implementation; benchmark protocol in project README. Các URL, version và ghi chú license ở [references](../../docs/references.md).
Đọc [limitations](../../docs/limitations.md) trước khi diễn giải một smoke experiment thành
kết quả nghiên cứu. Các claim của công trình gốc không phải kết quả của repository này.
