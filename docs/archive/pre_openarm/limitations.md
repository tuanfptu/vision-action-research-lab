# Trạng thái, giới hạn và công việc còn lại

Mã nguồn thực thi có cho cả tám lab. "Có implementation" không đồng nghĩa đã tái lập
kết quả paper. Xem validation.md để biết lệnh và số liệu đã thực sự chạy.

- Lab 00: CPU được kiểm tra; CUDA trên máy hiện tại chưa hoạt động. Không sửa driver.
- Lab 01: BC/state-only có data, train, checkpoint, offline và PushT closed-loop. Dataset
  nhỏ và hai epoch chỉ kiểm tra pipeline; chưa có success-rate đáng tin cậy.
- Lab 02: ACT giáo dục dùng CNN pooling, posterior Transformer và decoder, không có toàn
  bộ visual token encoder của paper. Official ACT có adapter train/checkpoint/evaluate riêng với reduced config. Chưa tái lập
  full-budget ACT hoặc ALOHA.
- Lab 03 PARTIAL: chunk/latent/KL và rollout horizon/ensemble có script tự động. Chưa có
  sweep encoder/history, dataset size, unseen-state perturbations và báo cáo thống kê nhiều
  seed. Latency số nhỏ chịu ảnh hưởng warmup/CPU scheduling; cần p50/p95, batch 1.
- Lab 04: genuine diffusion/flow objective và samplers; chưa xác nhận mode coverage,
  calibration hoặc ưu thế so với ACT. Không so training loss giữa các objective.
- Lab 05: four-arm encoder transfer chạy được. Chưa phải APT action expert gốc; source và
  target synthetic khác seed vẫn chưa chứng minh cross-robot/task transfer. Cần nguồn pretraining độc
  lập, target subsampling và nhiều seed để nghiên cứu sample efficiency/negative transfer.
- Lab 06 PARTIAL: two-frame GRU joint frame/action predictor; chưa có dual-stream latent
  diffusion, autoregressive rollout, refresh experiment hay drift curves. Không tái lập LingBot.
- Lab 07 PARTIAL: original MuJoCo slide reacher và learned target tracking; không phải Panda,
  humanoid, sim-to-real hoặc Helix. Controller PD bị giới hạn không chứng minh stability toàn
  cục; watchdog dùng simulated age, chưa mô phỏng wall-clock overruns/IPC/network jitter.
- Validation split dùng để chọn checkpoint; chưa có untouched test set. Không đưa ra claims
  về generalization trước khi thêm final episode split và repeated-seed confidence intervals.
- Dataset tải theo yêu cầu; `--episodes 6` không đại diện toàn bộ 206 episodes.
- Video model chưa có perceptual/PSNR metric riêng ngoài joint loss và hình so sánh.

Ưu tiên tiếp: benchmark PushT đủ budget, episode test split, fixed seeds và paired comparisons;
sau đó spatial visual tokens cho ACT, temporal history, controlled transfer, video rollout drift;
cuối cùng mới thêm Panda từ Menagerie cùng license, IK và action-schema adapter riêng.
