# Phương pháp nghiên cứu Vision–Action trên OpenArm

Tài liệu này là hướng dẫn đọc và phản biện nghiên cứu, đi cùng code chạy thật. Báo cáo
[số đo](research_report.md) trả lời điều đã quan sát; tài liệu này định nghĩa phép đo,
phương trình và điều kiện để một kết luận có giá trị. Các mô hình nhỏ là công cụ học thuật;
chúng không đại diện cho kết quả của các foundation model gốc.

## 1. Câu hỏi và đơn vị thí nghiệm

| Câu hỏi | Đối chứng | Đại lượng quan sát | Điều chưa suy ra được |
|---|---|---|---|
| Ảnh có giúp dự đoán action? | BC vs state-only cùng split, seed, budget | Raw action MSE theo episode | Nhân quả của vision; cần thêm image intervention |
| ACT có lợi trong budget này? | Educational ACT, official ACT, BC | Mean MSE và paired difference | Tái lập thứ hạng paper trên task khác |
| Chunk/CVAE/camera/budget ảnh hưởng thế nào? | Thay một yếu tố ACT, dùng lại baseline | First-action MSE, ba seed | Interaction giữa các yếu tố; full factorial chưa chạy |
| Noise/flow sampler có ổn? | Cùng checkpoint, 1/5/10/20/50 bước | Sai số và latency | Mode coverage hoặc xác suất thành công |
| Video head học được chuyển động? | Giữ nguyên ảnh hiện tại | Future pixel MSE theo horizon | Dynamics nhân quả, planning hoặc autoregressive stability |
| Encoder pretraining có lợi? | Random/frozen/partial/full | Target-only validation | Cross-task transfer khi chỉ có một task |
| Controller phản ứng với lỗi? | Fresh/stale/NaN/clock/shape cases | Hold, rejection, slew bound | An toàn phần cứng và deadline của IPC thực |

Một run là một lần khởi tạo và huấn luyện với seed xác định. Một episode là đơn vị lấy mẫu
đánh giá. Các frame và chunk chồng lấn trong cùng episode có tương quan mạnh; không coi
hàng chục nghìn action scalar là hàng chục nghìn trial độc lập. Với chỉ bốn validation và
bốn test episodes, ngay cả ba seed cũng chưa tạo ra nghiên cứu có statistical power cao.

## 2. Contract dữ liệu và biểu diễn

Tại thời điểm t, observation là ảnh I_t có shape [V,3,32,32] và joint position q_t∈R^16.
Action a_t∈R^16 là absolute position target, đơn vị radian. Chuỗi label dài K là
A_t=[a_t,…,a_(t+K−1)], không phải joint displacement, torque hay Cartesian pose.

Cameras có thứ tự; concatenate feature không bất biến với hoán đổi view. Camera intrinsics
và tỉ lệ ảnh sau resize không được coi là giống renderer độc lập. Joint order lấy từ metadata,
không suy từ qpos index. Xem [contract](openarm_contract.md) để tra toàn bộ tên và mapping.

Split theo episode với split_seed=7: 22 train, 4 validation, 4 test. Train-only statistics:

    x_norm = (x − μ_train) / max(σ_train, 1e−4)
    a_rad  = a_norm · σ_action_train + μ_action_train

Với half-data, fit lại stats trên đúng phần training budget. Vì vậy normalized MSE giữa hai
budget không cùng thước đo; dùng raw radian MSE cho so sánh. Một left-gripper coordinate gần
hằng có thể khuếch đại normalized error; report cả 16 joint RMSE để không che điều này.

Window tail lặp giá trị cuối để giữ shape nhưng valid mask loại phần padding. History đầu
episode lặp frame đầu. Future-frame mask dịch thêm một bước: prediction ảnh bắt đầu t+1,
trong khi action chunk bắt đầu t. Vectorized WindowBatches giữ chính xác các quy tắc này.

## 3. BC, state-only và persistence

Encoder CNN dùng chung weights cho từng view: [B·V,3,R,R] → [B,V,D]. Fusion concatenate
V·D feature với q_norm rồi chiếu về [B,D]. MLP xuất [B,K,16]. BC tối thiểu hóa:

    L_BC = Σ_(b,k,j) m_bk (â_bkj − a_bkj)^2 / (16 Σ_(b,k) m_bk)

State-only giữ cùng head/fusion width nhưng bỏ CNN feature khỏi dự đoán. Hai kiến trúc có
số parameter khác nhau; đây là đối chứng modality thực dụng, không phải matched-parameter
ablation. Parameter counts và trạng thái pretrained backbone phải được công bố khi mở rộng.

Persistence dự đoán q_t cho mọi offset của chunk; train-mean dự đoán μ_action_train. Cả hai
không đọc action label tương lai. Persistence hợp lệ chỉ khi tên, thứ tự và đơn vị state/action
giống nhau; research evaluator kiểm tra điều kiện này. Với action position chậm biến đổi,
persistence có thể rất mạnh dù không có khả năng quyết định khi nào tiến đến/nắm/nâng vật.

## 4. ACT và CVAE: đọc đúng gradient và inference

Educational ACT dùng posterior qφ(z|A_t,I_t,q_t), diagonal Gaussian với μ và log σ².
Reparameterization z=μ+exp(logvar/2)·ε cho phép reconstruction gradient đi qua posterior.
Posterior Transformer đọc context/CLS và action tokens, dùng padding mask cho tail.

    KL(q||N(0,I)) = −0.5 Σ_l (1 + logvar_l − μ_l² − exp(logvar_l))
    L_ACT = L_masked_action + β E_batch[KL]

Decoder queries [B,K,D] attend vào context và latent memory. Khi đánh giá, z=0 và posterior
không đọc label; đây là deterministic prior-mean inference, không phải posterior reconstruction.
Nếu β=0 cải thiện validation trong study nhỏ, chưa đủ kết luận latent là vô ích: posterior có
thể mang thông tin không có ở prior, hoặc learning budget không đủ cho regularization.

Official adapter gọi ACTPolicy của LeRobot 0.3.3, ResNet18 không pretrained, mỗi camera là
một input key. Width/depth đã giảm. Internal normalization là IDENTITY vì outer pipeline đã
normalize. Objective official và educational không hoàn toàn giống nhau; so raw prediction,
không xếp hạng bằng training loss. [ACT paper](https://arxiv.org/abs/2304.13705) là nguồn gốc
của action chunking/CVAE; lab không tái lập setup phần cứng hoặc dataset của paper.

Prediction chunk K khác execution horizon h≤K. Offline +0 action là điểm chung để so K=1/4/8.
Temporal ensemble gộp các dự đoán từ chunk khác nhau cho cùng một thời điểm; không lấy trung
bình sai offset. Các control ablation trong simulator độc lập phải ghi riêng khỏi offline MSE.

## 5. Diffusion và flow: objective khác, cùng action contract

DDPM noise prediction dùng 1,000 linear β steps; ᾱ_t là cumulative product của 1−β_t:

    x_t = sqrt(ᾱ_t) A + sqrt(1−ᾱ_t) ε
    L_noise = E ||εθ(x_t,t,context) − ε||²_masked

DDIM deterministic update suy clean action từ noise estimate rồi đi đến timestep kế tiếp.
Sai số noise ở timestep có ᾱ rất nhỏ bị chia cho sqrt(ᾱ), dễ tạo action rất lớn khi chưa học
đủ. Lab giữ số đo thất bại; không clip x0 sau khi thấy validation để làm đẹp bảng. Sửa schedule,
loss weighting hoặc backbone cần một protocol mới và validation selection mới.

Flow prototype dùng đường thẳng noise→action, không phải optimal-transport coupling:

    x_t = (1−t) ε + t A;  v* = A−ε
    L_flow = E ||vθ(x_t,t,context) − v*||²_masked
    x_(t+Δt) = x_t + Δt vθ(x_t,t,context)

Euler nhiều bước không bảo đảm tốt hơn nếu learned field sai. Sampler sweep dùng cùng
checkpoint và không chọn best-of-N bằng ground-truth. Point MSE đánh giá độ gần label duy
nhất, có thể phạt một mode hợp lệ khác; cần task success/mode coverage trên benchmark có nhiều
mode để kết luận về lợi ích generative policy. Nền tảng: [Diffusion Policy](https://arxiv.org/abs/2303.04137),
[DDIM](https://arxiv.org/abs/2010.02502), [Flow Matching](https://arxiv.org/abs/2210.02747).

## 6. VA prior và transfer: kiểm soát được gì?

Script transfer chỉ chia global training episodes thành source và target rời nhau. Global
validation/test được giữ ngoài. Mỗi pool có internal split riêng; target validation mới là
điểm đánh giá của bốn transfer arm. Source checkpoint không đọc target episodes.

Random không copy weights. Frozen copy toàn encoder và khóa gradient; partial mở fusion;
full mở toàn encoder. Mọi arm khởi tạo action head mới, cùng target data và update budget.
Đây là nghiên cứu initialization cùng task. Source compute là chi phí bổ sung, không được
ẩn khi báo sample efficiency. Source/target normalization khác nhau làm phần state của
encoder chịu coordinate shift; hiện chưa có affine compensation, cần ghi như confound.

[APT](https://arxiv.org/abs/2606.12366) học VA prior trong action expert trước khi ghép language.
Lab không có frozen VLM, gated language fusion hoặc đánh giá unseen instructions. Encoder
transfer ở đây chỉ minh họa kiểm soát initialization, không kiểm chứng kết luận của APT.
Cross-task extension cần task_id, split theo task trước episode, source/target schema chung,
đối chứng equal-compute và learning curves ở nhiều target budgets.

## 7. Video–Action và giới hạn world-model claim

GRU đọc history [B,H,V,3,R,R], mỗi frame dùng cùng current q_t (không có past-state sequence).
Hidden state sinh action chunk và future pixels [B,K,V,3,16,16]. Loss cộng action MSE và
pixel MSE, mỗi loại dùng valid mask riêng. Pixel target resize nearest như implementation.

Baseline giữ frame I_t cho mọi future offset. Report từng horizon và aggregate MSE; PSNR
là −10 log10(MSE) với pixel range [0,1]. High PSNR có thể do background tĩnh và downsampling;
cần foreground/contact metrics trước khi nói hiểu dynamics. Nếu prediction thua persistence,
không nên dùng pixel head như bằng chứng của world model thành công.

[Official LingBot-VA](https://huggingface.co/docs/lerobot/main/en/lingbot_va) dùng mô hình latent
video/action quy mô lớn và autoregressive inference. Prototype ở đây là direct prediction;
không có latent diffusion, KV cache, action-conditioned dynamics hoặc rollout imagination.
Đo autoregressive drift cần định nghĩa state feedback và image refresh rõ ràng; lặp video
head với ground-truth state ở mỗi bước sẽ trộn teacher forcing vào kết quả.

## 8. Controller và phân tầng

Pipeline thực thi: TaskGoal → policy chunk → denormalize → named joint targets → bounded
command interface → MuJoCo position servos. Policy 10/15/30 Hz; physics 1 kHz; image observation
30 Hz với số physics steps xen kẽ 33/34. Đây là các rate của lab, không phải Helix.

Controller bound với position q và target u:

    next = clip(q + clip(u−q, −v_max Δt, v_max Δt), lower, upper)

Stale (>250 ms) hoặc nonfinite target dẫn đến hold. State nonfinite không cung cấp hold
đáng tin nên reject; invalid clock/shape cũng reject. Test interface không chứng minh robot
không va chạm hoặc giữ ổn định khi mất nguồn. Temporal ensemble, slew bound và actuator joint
limit là các lớp khác nhau, không thay thế collision checking hoặc safety controller thực.

[Figure Helix 02](https://www.figure.ai/news/helix-02) mô tả S2/S1/S0 ở các timescale khác nhau.
Lab chỉ lấy ý tưởng phân tầng: TaskGoal một task, policy học offline và servo explicit. Không
có trained reasoner, learned whole-body System 0, humanoid balance hoặc sim-to-real validation.

## 9. Thống kê, model selection và final test

Định nghĩa error episode e, training seed s:

    M_se = Σ_(t,k,j) valid_tk (â_stkj − a_tkj)^2 / (16 Σ_(t,k) valid_tk)
    primary = mean_(s,e) M_se

Mỗi episode có trọng số bằng nhau. Chunk overlapping là một phần định nghĩa prediction
risk, không tạo thêm independent evidence. sqrt(primary) có đơn vị radian, khác mean của
các episode RMSE. Joint RMSE dùng tất cả valid targets trong episode.

Bootstrap 10,000 lần resample seed axis và episode axis độc lập; paired difference dùng
cùng seed/episode indices cho hai mô hình. Percentile 2.5/97.5% là interval mô tả. N=4 rất
nhỏ, có thể under-cover; không gọi nó là bảo đảm population-level 95% và không dùng CI overlap
như hypothesis test. Nhiều exploratory ablation chưa multiple-comparison correction.
Nguồn động lực cho việc báo uncertainty: [Agarwal et al.](https://arxiv.org/abs/2108.13264);
implementation này là crossed bootstrap riêng cho study, không phải toàn bộ rliable protocol.

Protocol ghi config, data hash, source hash, split trước training. Trainer chọn checkpoint
bằng validation objective có sẵn. Family chọn bằng validation raw primary metric, average
cả ba seed; tie break alphabetic. Selection file đóng băng checkpoint SHA trước khi đọc test.
Final test chỉ gồm selected family, BC reference và hai baselines; không chọn seed tốt nhất.
Khi đã mở test, split đó là used test. Mọi architecture/hyperparameter tuning tiếp theo phải
quay về validation, và claim final confirmation mới cần một independent test collection.

Half-data giữ cùng số epochs nhưng có ít optimizer steps hơn vì ít frame; chưa cô lập data
quantity khỏi update count. Khi đổi camera/resolution cần train fresh matched-budget controls,
không so một run 3 epochs với baseline 12 epochs. Official ACT BatchNorm cũng nhạy batch size.

## 10. Điều kiện nâng cấp kết luận

1. Tái dựng đúng source scene hoặc thu dataset mới cùng exact MJCF/controller/camera/reset
   manifest, sau đó đánh giá dài đủ để có thể hoàn thành pick-up.
2. Tăng số independent episodes, tasks, object poses, textures và seeds; preregister power
   target dựa trên variance pilot, không dựa trên frame count.
3. Thêm action-conditioned world model, intervention/occlusion tests và contact-sensitive
   metrics; giữ baseline persistence trong mọi bảng.
4. Cross-task transfer dùng source/target tasks khác nhau, equal-compute controls và target
   budget curves; audit state normalization và encoder transfer trước khi diễn giải.
5. Báo negative results, confidence intervals, runtime và compute cost cùng positive cases.
