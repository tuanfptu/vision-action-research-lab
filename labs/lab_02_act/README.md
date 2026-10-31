# LAB 02 — ACT: Action Chunking with Transformers

## 1. Mục tiêu học tập

Hiểu CVAE posterior, KL regularization, reparameterization, chunk và temporal ensemble; trace official implementation.

## 2. Kiến thức và môi trường cần có

Lab 01; attention, Transformer và Gaussian distributions.

## 3. Nền tảng toán học

qφ(z|a1:K,s,I)=N(μ,diag(exp(logvar))); z=μ+exp(0.5 logvar)⊙ε. KL(q||N(0,I))=−0.5 Σ(1+logvar−μ²−exp(logvar)). L=Lmasked-MSE+βKL trong bản giáo dục; official ACT sử dụng reconstruction recipe riêng, thường L1. Inference dùng prior mean z=0, tuyệt đối không đưa target action vào posterior. Chunk K dự đoán tương lai, execution horizon h≤K quyết định khi nào lấy ảnh mới.

## 4. Kiến trúc và computational flow

CNN+state → context token. Posterior Transformer đọc CLS+action tokens cùng positional embedding và padding mask → μ/logvar. Decoder đọc K learned queries với memory [context,z]. Đây là educational reduction; official ACT có spatial image tokens và encoder/decoder lớn hơn. Temporal ensemble tổng hợp các dự đoán cùng timestep từ các chunk chồng lấn.

## 5. Tensor shapes

Posterior tokens [B,K+1,D], key padding [B,K+1]; μ,logvar,z [B,L]. Memory [B,2,D]. Query [B,K,D] → decoder [B,K,D] → Linear [B,K,2]. KL scalar; reconstruction scalar. Decoder inference không cần action tensor.

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
va-lab train --data data/synthetic.npz --output runs/lab02_act --model act
va-lab verify-lerobot
va-lab train --data data/pusht.npz --output runs/pusht_act --model act
va-lab rollout --checkpoint runs/pusht_act/best.pt --horizon 1 --ensemble
```

## 9. Hành vi kỳ vọng và cách đọc kết quả

Test kiểm tra prior prediction không đổi khi sửa ground-truth actions trong batch. Official ACT API smoke không tải pretrained backbone. Short run không chứng minh success-rate paper.

Không có giá trị performance kỳ vọng giả định. Đọc số đo đã chạy trong
[validation](../../docs/validation.md); không so training loss giữa objective khác nhau.

## 10. Bài tập thực nghiệm

In shape tại posterior và decoder; tự suy KL một chiều. So β=0 với β>0; plot latent variance. Giữ K cố định và thay h. So ensemble và không ensemble ở h=1.

## 11. Failure analysis

Posterior collapse khi β quá lớn; train/inference gap nếu decoder quá phụ thuộc posterior. K lớn có thể gây stale actions. Padding không mask làm target cuối episode chi phối loss. Ensembling có thể làm mượt nhưng tăng độ trễ.

## 12. Câu hỏi kiểm tra hiểu sâu

Tại sao posterior dùng action khi train không phải leakage nếu inference không dùng? K và execution horizon khác nhau thế nào? Temporal ensemble ghép theo thời gian thực hay vị trí trong chunk?

## 13. Tài liệu chính thức

ACT paper 2304.13705; official ACT and LeRobot v0.3.3 source. Các URL, version và ghi chú license ở [references](../../docs/references.md).
Đọc [limitations](../../docs/limitations.md) trước khi diễn giải một smoke experiment thành
kết quả nghiên cứu. Các claim của công trình gốc không phải kết quả của repository này.
