# Tái lập migration OpenArm trên GPU

```bash
cd ~/vision-action-research-lab
bash scripts/setup_gpu.sh
source .venv-gpu/bin/activate
unset PYTHONPATH
export HF_HOME="$PWD/data/hf"
export MUJOCO_GL=egl
va-lab doctor --require-gpu
va-lab verify-lerobot
va-lab download --episodes 30 --cameras ceiling wrist_right --size 32
va-lab inspect
va-lab train --model bc --output runs/my_openarm_bc
va-lab train --model state --output runs/my_openarm_state
va-lab train --model act --output runs/my_openarm_act
va-lab train --model official_act --output runs/my_openarm_official_act
va-lab train --model diffusion --output runs/my_openarm_diffusion
va-lab train --model flow --output runs/my_openarm_flow
va-lab train --model video --output runs/my_openarm_video
va-lab video --checkpoint runs/my_openarm_video/best.pt
python scripts/ablate.py --output runs/my_openarm_ablation --grid configs/models/act_smoke_ablation.yaml
python scripts/transfer.py --output runs/my_openarm_transfer
va-lab assets
va-lab sim-smoke --render
va-lab rollout --checkpoint runs/my_openarm_bc/best.pt --allow-unverified-scene
va-lab record --output data/my_openarm_collected.npz
VA_LAB_INTEGRATION=1 make test
make lint
```

Skip download if data/openarm.npz already exists. All exports and experiment directories
must be new; no silent overwrite. Standard training uses validation only; an initially reserved
final episode set is accessed by explicit `evaluate --split test` after model selection. The
recorded 2026-11-08 study has already used this test set.
GPU logs identify the actual device, CUDA runtime and peak allocated VRAM. CPU tests can run
separately with `make test PY=.venv/bin/python`; this is not a substitute for CUDA training.

All five original video views are acquired for each requested episode. To export five-view
or higher-resolution training caches, choose a new output file, all desired camera names and
--size. Match cameras/image_size in the training YAML. Named camera selection may reduce a
larger cache to a subset without changing episode splits or action coordinates.


## Bộ nghiên cứu GPU có khóa final test

Bộ ngày 08/11/2026 chạy FP32 trên GTX 1660 Ti, không tải foundation weights. Mỗi model
12 epochs, seed 7/17/27. Batch64; official ACT batch32, accumulation2. `preload_device`
đưa cache tensor lên GPU và dùng batch gather thay vì Python per-frame loading.

```bash
source .venv-gpu/bin/activate
unset PYTHONPATH
export HF_HOME="$PWD/data/hf"
export MUJOCO_GL=egl
# Cache 30 episode/hai camera đã được xác minh; bỏ download khi file đã có.
python scripts/research.py --output runs/my_research
python scripts/research_extensions.py --study runs/my_research
# Chỉ sau selection trên validation, mở test một lần:
python scripts/research.py --output runs/my_research --resume --finalize
python examples/openarm_followups.py --study runs/my_research --output runs/my_followups
```

`protocol.json` khóa config, source hash, data hash và split. `selection.json` khóa family,
validation-record digest và checkpoint hashes. Ghi JSON bằng exclusive create, không ghi đè.
Resume chỉ nhận run hoàn tất có config/checkpoint phù hợp; run training bị ngắt giữa chừng
được giữ nguyên và cần output mới. Cùng software/device hỗ trợ reproducibility; không cam kết
bitwise equivalence giữa driver, PyTorch hoặc GPU khác nhau.

Các artifact đã đo nằm ở `docs/results/research_20261108/`; checkpoints/raw data vẫn local,
không commit. Main study và followup không dùng test để chọn seed hoặc sửa hyperparameters.
Sau finalization ngày 08/11, test này đã dùng; nghiên cứu xác nhận mới cần test collection mới.

Để tổng hợp báo cáo bằng artifact mới, dùng `examples/report_research.py --help`. GPU telemetry
CSV có cột unix_time,gpu_util_percent,memory_used_mib,temperature_c,power_w, lấy bằng nvidia-smi
mỗi hai giây trong main study/extensions. Không tự tạo số liệu utilization nếu chưa monitor.
Report generator yêu cầu thư mục artifact output mới và chỉ sao chép số đo nhỏ, không weights.

## Đối chứng camera/độ phân giải lớn hơn trên GPU

```bash
va-lab download --output data/openarm_fiveviews64.npz --episodes 30 \
  --cameras ceiling head_left head_right wrist_left wrist_right --size 64
python examples/camera_resolution.py --study runs/my_research --output runs/my_camera_controls
```

Script kiểm tra equality toàn bộ numeric trajectories giữa hai cache; train mới ba seed cho
2view32, 2view64 và 5view64, mỗi case 3 epochs và batch64/accumulation1. Cùng model family đã
được chọn từ primary validation, cùng episode split, không evaluate test hoặc thay selection.
Đây là exploratory study riêng; không dùng main 12-epoch scores làm đối chứng 3-epoch.

Telemetry wrapper dùng được cho một command (output paths mới):

```bash
python examples/gpu_monitor.py --csv runs/my_gpu.csv --log runs/my_train.log -- \
  python scripts/research.py --output runs/my_research
```

Tổng hợp số đo sau khi các bước training/followup hoàn tất:

```bash
python examples/report_research.py --study runs/my_research --followups runs/my_followups \
  --telemetry runs/my_gpu.csv --camera-study runs/my_camera_controls \
  --output runs/my_report_artifacts --report runs/my_report.md --figure-dir runs/my_figures
```

Với report đặt ngoài docs, sửa relative links hoặc giữ layout tương ứng khi chia sẻ. Protocol
chính ghi source hash của src/ và scripts/; helper trong examples/ có hash riêng trong artifact
của thí nghiệm phụ/report generator. CSV primary telemetry phải bao phủ đúng command đã chạy;
wrapper một command không tự bao phủ các followup chạy sau đó.
