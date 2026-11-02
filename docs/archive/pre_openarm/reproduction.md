# Tái lập các lệnh đã chạy

Chạy từ repository root trong `.venv`. Tất cả commands dùng `unset PYTHONPATH` để tránh
ROS Python path. Không cần previous repository. CPU environment snapshot ở
requirements-cpu.lock.txt; normal setup ở scripts/setup.sh.

```bash
source .venv/bin/activate
unset PYTHONPATH
export HF_HOME="$PWD/data/hf"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
va-lab doctor
va-lab verify-lerobot
va-lab generate --output data/synthetic.npz --episodes 12 --seed 7
va-lab download --output data/pusht.npz --episodes 6 --revision 96d10ccbf94cb6ab28fb73431272fc8651141a5d
for model in bc state act flow official_act; do
  va-lab train --data data/pusht.npz --model "$model" --output "runs/reproduce_$model"
done
va-lab train --data data/pusht.npz --model diffusion --output runs/reproduce_diffusion
python scripts/ablate.py --data data/synthetic.npz --output runs/reproduce_ablation
va-lab generate --episodes 8 --seed 17 --output data/target.npz
python scripts/transfer.py --source data/synthetic.npz --target data/target.npz --output runs/reproduce_transfer
va-lab train --data data/synthetic.npz --model video --output runs/reproduce_video
va-lab video --checkpoint runs/reproduce_video/best.pt --data data/synthetic.npz
va-lab record --episodes 5 --seed 17 --output data/reacher.npz
va-lab train --data data/reacher.npz --model bc --output runs/reproduce_reacher
va-lab hierarchy --checkpoint runs/reproduce_reacher/best.pt
make test lint
```

Các job đo PushT trong validation dùng 2 episode seed 10000/10001, h=1. CLI rollout mặc
định 300 steps; để tái lập smoke 100 steps, gọi API rõ budget:

```python
from va_lab.simulation.benchmark import pusht_rollout
pusht_rollout("runs/reproduce_bc/best.pt", "runs/reproduce_bc/rollout.json",
              episodes=2, max_steps=100)
```

Mỗi output directory phải mới; train từ chối overwrite. So configs/splits/hashes trước khi
so số. Latency không tái lập bitwise. Training budget smoke rất nhỏ; tăng epochs trong bản
sao CPU config và tăng số episode trước khi nghiên cứu performance. Tạo untouched test
split trước khi chọn kiến trúc nếu mục tiêu là báo cáo kết quả nghiên cứu.
