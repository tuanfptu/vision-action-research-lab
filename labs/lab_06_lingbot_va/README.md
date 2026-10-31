# LAB 06 — Educational Video–Action Prediction

## 1. Mục tiêu học tập

Phân biệt Vision–Action mapping với joint future video/action prediction; hiểu drift và observation refresh.

## 2. Kiến thức và môi trường cần có

Lab 01 và 04; CNN decoder, multi-task loss và temporal modeling.

## 3. Nền tảng toán học

Prototype học (â1:K,Î1:K)=fθ(I_{t−1:t},s_t), L=Laction+Lpixel, cùng valid mask cho các frame thực. Pixel MSE tối ưu conditional mean nên có thể làm mờ khi future đa mode. World model dài hạn cần p(I_future,a_future|history) và rollout prediction vào input; sai số có thể khuếch đại mỗi bước. Prototype này chưa có autoregressive evaluator.

## 4. Kiến trúc và computational flow

Shared CNN → two-frame GRU → action MLP và frame head sigmoid 16×16. Baseline two-frame context được giữ nhỏ để đo backward, visualization và learning trước khi thêm temporal latent/diffusion. Official LingBot-VA dual-stream/world-model khác đáng kể; không gọi baseline này là reproduction.

## 5. Tensor shapes

Input past [B,2,3,32,32], state [B,2] → CNN [B,2,D] → GRU hidden [B,D]. Action [B,K,2]; frame head [B,K×3×16×16] → [B,K,3,16,16]. Target ảnh resize bilinear từ [B,K,3,32,32]. future_valid khác action valid tại episode end.

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
va-lab train --data data/synthetic.npz --output runs/lab06_video --model video
va-lab video --checkpoint runs/lab06_video/best.pt --data data/synthetic.npz --output runs/lab06_video/future.png
```

## 9. Hành vi kỳ vọng và cách đọc kết quả

Hình hàng trên ground truth, hàng dưới prediction. PARTIAL: đã có two-frame GRU encoder; chưa có latent diffusion/flow dual-stream, autoregressive/refresh experiments và drift metrics. Có genuine trainable joint model, không dùng output random thay model.

Không có giá trị performance kỳ vọng giả định. Đọc số đo đã chạy trong
[validation](../../docs/validation.md); không so training loss giữa objective khác nhau.

## 10. Bài tập thực nghiệm

Tách action và video loss, đo PSNR/SSIM và action error từng future horizon. So history 2/4 frames, GRU hoặc temporal attention; so teacher-forced và autoregressive prediction. Refresh bằng observation thật mỗi r bước và vẽ drift.

## 11. Failure analysis

Pixel loss có thể bị background chi phối; ảnh mờ không đồng nghĩa world understanding. Dự đoán future và chọn action có causal ambiguity. Padding frame không được tính vào loss; action đúng không đảm bảo video đúng.

## 12. Câu hỏi kiểm tra hiểu sâu

Tại sao thêm video loss có thể giúp hoặc hại policy? Vì sao teacher forcing che giấu rollout drift? Một model dự đoán frame có đủ để gọi là world model dùng planning chưa?

## 13. Tài liệu chính thức

Official LingBot-VA LeRobot documentation; diffusion/flow references. Các URL, version và ghi chú license ở [references](../../docs/references.md).
Đọc [limitations](../../docs/limitations.md) trước khi diễn giải một smoke experiment thành
kết quả nghiên cứu. Các claim của công trình gốc không phải kết quả của repository này.
