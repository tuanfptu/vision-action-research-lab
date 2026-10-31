# LAB 00 — Environment và Research Infrastructure

## 1. Mục tiêu học tập

Hiểu môi trường độc lập, dependency compatibility và khác biệt giữa smoke test với kết quả research.

## 2. Kiến thức và môi trường cần có

Ubuntu 22.04, Python 3.10, terminal, kiến thức tensor PyTorch. Không cần ROS trong Python env.

## 3. Nền tảng toán học

Với cùng seed s, pipeline cần cùng split và khởi tạo. Determinism CPU không bảo đảm bitwise equivalence giữa GPU, kernel và phiên bản thư viện. Hash H(D) kiểm tra identity của dữ liệu; nó không chứng minh chất lượng hay license.

## 4. Kiến trúc và computational flow

CPU: Python → venv → torch/torchvision → package → pytest → MuJoCo physics. Config YAML chọn batch, chunk, model; run directory ghi JSONL và checkpoint.

## 5. Tensor shapes

B=8, C=3, H=W=32, state=2, K=4, action=2. Input batch: image [8,3,32,32], state [8,2], target [8,4,2], valid [8,4].

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
bash scripts/setup.sh
make test lint
va-lab doctor
va-lab verify-lerobot
```

## 9. Hành vi kỳ vọng và cách đọc kết quả

Import, finite loss/gradient và save/load thành công chỉ cho biết software path hoạt động. GPU chưa được xác nhận nếu cuda_available=false.

Không có giá trị performance kỳ vọng giả định. Đọc số đo đã chạy trong
[validation](../../docs/validation.md); không so training loss giữa objective khác nhau.

## 10. Bài tập thực nghiệm

Đổi seed và kiểm tra split thay đổi; lặp seed giữ split. Cố tình sửa dataset sau train rồi chạy evaluate để quan sát hash rejection. Ghi pip freeze và phân biệt package version với CUDA runtime.

## 11. Failure analysis

PYTHONPATH từ ROS có thể làm pytest nạp plugin ngoài venv; bỏ biến trong tiến trình dự án, không gỡ ROS. pip check không phát hiện mọi binary ABI mismatch; vẫn phải forward/backward.

## 12. Câu hỏi kiểm tra hiểu sâu

Tại sao venv vẫn có thể bị PYTHONPATH làm nhiễm? Vì sao seed không đủ để tái lập experiment? CUDA wheel khác NVIDIA driver như thế nào?

## 13. Tài liệu chính thức

LeRobot versioned metadata; PyTorch release compatibility; MuJoCo overview. Các URL, version và ghi chú license ở [references](../../docs/references.md).
Đọc [limitations](../../docs/limitations.md) trước khi diễn giải một smoke experiment thành
kết quả nghiên cứu. Các claim của công trình gốc không phải kết quả của repository này.
