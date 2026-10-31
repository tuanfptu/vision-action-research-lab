# LAB 05 — APT-inspired VA Prior Transfer

## 1. Mục tiêu học tập

Tách tác dụng pretraining khỏi architecture; kiểm tra frozen/partial/full fine-tuning và negative transfer.

## 2. Kiến thức và môi trường cần có

Lab 01 và 04; transfer learning, frozen parameters và optimizer.

## 3. Nền tảng toán học

Source pretrain tối thiểu hóa Lsource(θenc,θhead). Target giữ architecture và split cố định, so random với θenc←θsource. Frozen: ∇θenc=0; partial chỉ fusion học; full mọi parameter học. Sample efficiency cần curve quality theo số target episodes. Negative transfer là pretrained thấp hơn random dưới protocol thống nhất, cần uncertainty để kết luận.

## 4. Kiến trúc và computational flow

scripts/transfer.py train BC source, copy encoder weights sang target model mới, giữ target head initialization cùng seed. Frozen encoder gồm CNN và fusion; partial mở fusion; full mở toàn encoder. Đây là supervised VA encoder prior, không phải toàn bộ Action Expert của APT.

## 5. Tensor shapes

Source/target cùng RGB [B,3,32,32], state [B,2], action [B,K,2], encoder [B,D]. Không copy normalization source sang target: target stats fit trên target train. Encoder vẫn nhìn state normalized theo target nên cần phân tích domain shift.

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
va-lab generate --episodes 12 --output data/source.npz
va-lab generate --episodes 8 --seed 17 --output data/target.npz
python scripts/transfer.py --source data/source.npz --target data/target.npz --output runs/lab05
```

## 9. Hành vi kỳ vọng và cách đọc kết quả

Four-arm run tạo checkpoint và metric thật; hai synthetic sets trong lệnh mẫu chỉ kiểm tra workflow, chia sẻ task distribution nhưng khác seed. Không dùng chúng làm bằng chứng cross-task transfer; xem limitations.

Không có giá trị performance kỳ vọng giả định. Đọc số đo đã chạy trong
[validation](../../docs/validation.md); không so training loss giữa objective khác nhau.

## 10. Bài tập thực nghiệm

Tạo source/target khác goal distribution và seed. Kiểm tra episode overlap bằng manifest và hash; match compute budget. Thử target fractions, ≥3 seeds, compare confidence intervals và gradient norms.

## 11. Failure analysis

Source/target overlap làm optimistic transfer. Frozen feature thiếu thông tin target không thể cứu bằng head. Full fine-tuning có thể catastrophic forgetting. Nếu action schema đổi thì phải thiết kế adapter, không copy head tùy ý.

## 12. Câu hỏi kiểm tra hiểu sâu

Pretrained encoder và Action Expert pretraining khác nhau gì? Vì sao frozen không luôn sample-efficient hơn full? Khi nào phải giữ normalization giống source?

## 13. Tài liệu chính thức

APT authors repository and arXiv 2606.12366; không suy diễn kiến trúc độc quyền từ tên gọi. Các URL, version và ghi chú license ở [references](../../docs/references.md).
Đọc [limitations](../../docs/limitations.md) trước khi diễn giải một smoke experiment thành
kết quả nghiên cứu. Các claim của công trình gốc không phải kết quả của repository này.
