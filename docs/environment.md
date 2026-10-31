# Môi trường độc lập và phần cứng

Python 3.10.12 hệ thống chỉ dùng để tạo `.venv`; mọi thư viện cài trong môi trường mới.
Không thay ROS 2, driver hoặc package toàn cục. Shell đã source ROS có thể đặt PYTHONPATH;
chạy `unset PYTHONPATH` trong shell dự án, hoặc `env -u PYTHONPATH .venv/bin/python ...`.
Đây là thay đổi môi trường của tiến trình, không sửa cài đặt ROS.

Cặp chính thức: torch 2.7.1 / torchvision 0.22.1. Chọn LeRobot 0.3.3 vì metadata PyPI
cho phép Python >=3.10 và torch <2.8. CPU wheel cài trước để pip không kéo CUDA runtime
không cần thiết. MuJoCo 3.3.2 chạy physics không cần renderer cho smoke test.
TorchCodec 0.5 yêu cầu torch mới hơn cặp đã chọn; dùng 0.4.0 với torch 2.7 và PyAV
cho video nếu cần. Dataset PushT ảnh không cần video decoder.

```bash
bash scripts/setup.sh
source .venv/bin/activate
unset PYTHONPATH
pip check
va-lab doctor
va-lab verify-lerobot
```

GPU 6 GB: bắt đầu batch 8, ảnh 32×32, dim 64, chunk 4, FP32. Không dùng BF16 như giả định
trên GPU Turing này. Nếu driver hỗ trợ CUDA 11.8, có thể thay wheel trong `.venv`:

```bash
pip install --force-reinstall torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
va-lab train --config configs/training/low_vram.yaml --data data/pusht.npz --output runs/gpu_bc
```

Không tự chạy lệnh GPU khi driver chưa kiểm chứng. Cloud preset là điểm bắt đầu để tăng
budget, không phải một kết quả benchmark. Đo peak VRAM bằng torch.cuda.max_memory_allocated
khi chạy trên GPU thực. Không tự tải checkpoint nền tảng hoặc model hàng tỷ tham số.

`requirements-cpu.lock.txt` ghi snapshot môi trường kiểm thử; setup từ source tiếp tục dùng
pins trong pyproject và torch index. Muốn khôi phục snapshot, cài CPU torch trước rồi cài
lock, sau đó `pip install -e . --no-deps`. Lock phụ thuộc nền tảng Linux x86_64/Python 3.10.
