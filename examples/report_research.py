"""Publish small measured artifacts and scientific plots from a completed study."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from va_lab.datasets.core import file_hash
from va_lab.utils.runtime import write_json


def load(path):
    return json.loads(Path(path).read_text())


def interval(value):
    lo, hi = value["ci95"]
    return f"{value['mean']:.7f} [{lo:.7f}, {hi:.7f}]"


def run(args):
    root, follow, out = Path(args.study), Path(args.followups), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    for name in [
        "protocol",
        "started",
        "selection",
        "validation_records",
        "ablations",
        "sampler_sensitivity",
        "controller_faults",
        "final_test",
        "completed",
    ]:
        shutil.copyfile(root / f"{name}.json", out / f"{name}.json")
    for name in ["protocol", "transfer_summary", "independent_control_summary"]:
        shutil.copyfile(follow / f"{name}.json", out / f"followup_{name}.json")
    training = []
    bases = [root, follow / "transfer"]
    if args.camera_study:
        camera_root = Path(args.camera_study)
        shutil.copyfile(camera_root / "protocol.json", out / "camera_resolution_protocol.json")
        shutil.copyfile(camera_root / "summary.json", out / "camera_resolution_summary.json")
        bases.append(camera_root)
    for base in bases:
        for p in sorted(base.rglob("metrics.jsonl")):
            training.append(
                {
                    "run": str(p.parent),
                    "config": load(p.parent / "config.json"),
                    "runtime": load(p.parent / "runtime.json"),
                    "split": load(p.parent / "split.json"),
                    "checkpoint_sha256": file_hash(p.parent / "best.pt"),
                    "epochs": [json.loads(line) for line in p.read_text().splitlines()],
                    "validation": load(p.parent / "evaluation.json"),
                }
            )
    write_json(out / "training_runs.json", {"runs": training})
    with Path(args.telemetry).open() as handle:
        telemetry = list(csv.DictReader(handle))
    stats = {}
    for key in ["gpu_util_percent", "memory_used_mib", "temperature_c", "power_w"]:
        x = np.array([float(r[key]) for r in telemetry])
        stats[key] = {
            "mean": float(x.mean()),
            "p95": float(np.percentile(x, 95)),
            "max": float(x.max()),
        }
    stats.update(
        samples=len(telemetry),
        scope="main study and validation extensions only; nvidia-smi total GPU includes desktop; 2-second samples",
    )
    write_json(out / "gpu_telemetry_summary.json", stats)
    (out / "gpu_telemetry.csv").write_text(Path(args.telemetry).read_text())
    selection, test = load(root / "selection.json"), load(root / "final_test.json")
    validation, final = selection["validation_summary"], test["summary"]
    ablation = load(root / "ablations.json")["summary"]
    video = [r for r in load(root / "validation_records.json") if r["model"] == "video"]
    sampler = load(root / "sampler_sensitivity.json")["records"]
    transfer = load(follow / "transfer_summary.json")
    control = load(follow / "independent_control_summary.json")["results"]
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, summary, title in zip(
        axes,
        [validation, final],
        ["Validation: 3 seeds x 4 episodes", "Final test: locked family + references"],
        strict=True,
    ):
        for i, (name, value) in enumerate(summary.items()):
            mean, (lo, hi) = value["mean"], value["ci95"]
            axis.errorbar(
                mean, i, xerr=[[max(0, mean - lo)], [max(0, hi - mean)]], fmt="o", capsize=3
            )
        axis.set_yticks(range(len(summary)), list(summary))
        axis.invert_yaxis()
        axis.set_xscale("log")
        axis.set_xlabel("Mean episode action MSE (rad^2), log scale")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("OpenArm offline pilot: descriptive crossed-bootstrap 95% intervals")
    fig.tight_layout()
    fig.savefig(figure_dir / "research_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axis = axes[0, 0]
    for name in ["diffusion", "flow"]:
        xs, ys = [], []
        for steps in [1, 5, 10, 20, 50]:
            rows = [r for r in sampler if r["model"] == name and r["sampling_steps"] == steps]
            xs.append(steps)
            ys.append(np.mean([r["action_mse_rad2"] for r in rows]))
        axis.plot(xs, ys, "o-", label=name)
    axis.set(
        xlabel="Sampler steps",
        ylabel="Validation MSE (rad^2)",
        yscale="log",
        title="Same checkpoint; one sample/window",
    )
    axis.legend()
    axis = axes[0, 1]
    for key, label in [
        ("horizon_frame_mse", "video model"),
        ("horizon_persistence_frame_mse", "repeat current image"),
    ]:
        values = np.array([r[key] for r in video])
        axis.plot(np.arange(1, values.shape[1] + 1), values.mean(0), "o-", label=label)
    axis.set(
        xlabel="Future frame offset at 30 Hz",
        ylabel="Pixel MSE at 16x16",
        title="Validation video vs persistence",
    )
    axis.legend()
    axis = axes[1, 0]
    for i, (name, value) in enumerate(ablation.items()):
        d = value["paired_difference_vs_base"]
        mean, (lo, hi) = d["mean"], d["ci95"]
        axis.errorbar(mean, i, xerr=[[max(0, mean - lo)], [max(0, hi - mean)]], fmt="o", capsize=3)
    axis.axvline(0, color="gray", linewidth=1)
    axis.set_xscale("symlog", linthresh=0.001)
    axis.set_yticks(range(len(ablation)), list(ablation))
    axis.invert_yaxis()
    axis.set(
        xlabel="First-action MSE difference (rad^2), symlog",
        title="ACT one-factor controls; validation only",
    )
    axis = axes[1, 1]
    for model in [selection["selected_family"], "bc"]:
        rows = [r for r in test["records"] if r["model"] == model]
        joints = np.array([r["joint_rmse_rad"] for r in rows])
        axis.plot(range(1, 17), np.sqrt((joints * joints).mean(0)), "o-", label=model)
    axis.set(
        xlabel="Dataset joint coordinate (1..16)",
        ylabel="Root macro episode joint MSE (rad)",
        title="Final test; right1..7,grip,left1..7,grip",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "research_diagnostics.png", dpi=160)
    plt.close(fig)
    rows = [
        "# Báo cáo nghiên cứu OpenArm — 08/11/2026",
        "",
        "Đã chạy thật trên GTX 1660 Ti bằng CUDA. Đây là nghiên cứu offline quy mô nhỏ trên một task, kèm thí nghiệm controller trong scene độc lập. Không phải tái lập thành tích foundation model hoặc xác nhận robot gắp cube thành công.",
        "",
        "## Thiết kế và khả năng tái lập",
        "",
        "- 30 episode / 5.405 frame; dataset v2.1 SHA `c5f7d41a37f9d9456b74eff673870997545c2d25`.",
        "- Split 22 train / 4 validation / 4 test; seed 7. Hai camera ceiling+wrist_right, 32×32, toàn bộ 16 absolute joint positions.",
        "- Bảy họ model × ba seed (7/17/27) = 21 run; 12 epochs/run. ACT one-factor có thêm 18 run; transfer có 15 run gồm source pretrain và bốn target arms cho mỗi seed.",
        f"- Tổng **{len(training)} run / {sum(len(r['epochs']) for r in training)} epoch**. FP32, width64, chunk4, effective batch64; official ACT batch32×accumulation2, ResNet18 không pretrained, reduced Transformer.",
        "- Vectorized windows đặt cache trên CUDA, chia sẻ tensor pool giữa train và validation. Không thay drivers, ROS hoặc system Python.",
        "- Protocol khóa trước training. Checkpoint chọn trên validation objective; family chọn bằng raw macro-episode MSE trung bình ba seed. Không chọn seed đẹp nhất.",
        "- Cả ba seed của family đã chọn và BC reference được đánh giá final test sau selection lock; persistence và train-mean là hai đối chứng. Tập test này **đã được sử dụng**; không còn được gọi là untouched cho nghiên cứu sau.",
        "",
        "Định nghĩa metric, toán học từng phương pháp và confounds: [methodology](research_methodology.md). Artifact thô: [protocol](results/research_20261108/protocol.json), [selection](results/research_20261108/selection.json), [training runs](results/research_20261108/training_runs.json).",
        "",
        "## So sánh chính",
        "",
        "MSE là mean của episode MSE trên seed/episode; CI là 10.000 crossed-bootstrap draws. Với chỉ bốn episode, interval mang tính mô tả và có thể under-cover. sqrt(MSE) không phải mean episode RMSE.",
        "",
        "| Family | Validation MSE rad² [CI95] | sqrt(MSE), rad |",
        "|---|---:|---:|",
    ]
    for name, value in validation.items():
        rows.append(f"| {name} | {interval(value)} | {value['sqrt_macro_mse_rad']:.5f} |")
    rows.extend(
        [
            "",
            "![Comparison](figures/research_comparison.png)",
            "",
            f"Family được chọn: **{selection['selected_family']}**. Đây là lựa chọn trong bảy cấu hình/budget đã đăng ký, không phải kết luận rằng family đó luôn tốt nhất.",
            "",
            "| Final test | MSE rad² [CI95] | sqrt(MSE), rad |",
            "|---|---:|---:|",
        ]
    )
    for name, value in final.items():
        rows.append(f"| {name} | {interval(value)} | {value['sqrt_macro_mse_rad']:.5f} |")
    chosen = final[selection["selected_family"]]
    delta = chosen["paired_difference_vs_bc"]
    reduction = 100 * (1 - chosen["mean"] / final["bc"]["mean"])
    rows.extend(
        [
            "",
            f"Trên đúng split này, selected family có MSE thấp hơn BC {reduction:.1f}%; paired difference={interval(delta)} rad². Đây là so sánh offline trên bốn test episodes; không chuyển thành manipulation success rate hoặc claim population-level significance.",
            "",
            "## Ablation và phân tích thất bại",
            "",
            "| ACT control | First-action MSE rad² | Paired difference vs base [CI95] |",
            "|---|---:|---:|",
        ]
    )
    for name, value in ablation.items():
        rows.append(
            f"| {name} | {value['mean']:.7f} | {interval(value['paired_difference_vs_base'])} |"
        )
    rows.extend(
        [
            "",
            "Dùng action +0 để so K=1/4/8 trên cùng endpoint. Đây là sáu one-factor controls, chưa phải full factorial và không có multiple-comparison correction. Half-data giữ cùng epochs nhưng có ít optimizer updates hơn; hiệu ứng data quantity và update count chưa được tách hoàn toàn.",
            "",
            "- β=0 tạo train/inference mismatch nghiêm trọng trong budget này: posterior có thể dùng latent trong train nhưng inference dùng z=0. Kết quả không chứng minh mọi CVAE cần cùng một β.",
            "- Diffusion chưa học được sampler đủ tốt; sampled joint MSE còn rất lớn. Không clip output để che lỗi. Flow tốt hơn trong budget này nhưng vẫn thua BC; không chứng minh diffusion/flow nói chung yếu hơn BC.",
            "- Sampler sensitivity dùng cùng checkpoint, validation only, 1/5/10/20/50 bước, một sample/window; không best-of-N. Latency là batch action-sampling latency CUDA-synchronized sau một warm-up, không phải end-to-end control latency.",
        ]
    )
    vm = np.mean([r["future_frame_mse"] for r in video])
    vp = np.mean([r["persistence_frame_mse"] for r in video])
    rows.extend(
        [
            f"- Video head pixel MSE={vm:.6f}, baseline giữ ảnh={vp:.6f}. Aggregate model thua persistence trên mọi seed/episode; riêng offset +4 có trung bình thấp hơn persistence. Không có bằng chứng video head đủ tốt cho planning; chưa chạy autoregressive drift/refresh protocol.",
            "",
            "![Diagnostics](figures/research_diagnostics.png)",
            "",
            "## VA prior cùng task",
            "",
            "| Target arm | Raw MSE rad² [CI95] | Difference vs random [CI95] |",
            "|---|---:|---:|",
        ]
    )
    for name, value in transfer["summary"].items():
        rows.append(
            f"| {name} | {interval(value)} | {interval(value['paired_difference_vs_random'])} |"
        )
    rows.extend(
        [
            "",
            "Source và target là hai pool rời nhau từ global train. Global validation/test không đi vào source/target training. Chỉ có **một target validation episode**; ba seed đo variability của initialization, không đủ đo generalization. Target head khởi tạo mới; frozen/partial/full chỉ copy encoder. Normalization source/target khác nhau chưa affine-compensated, là confound rõ ràng. Không gọi đây là cross-task transfer, APT reproduction hoặc sample-efficiency advantage.",
            "",
            "## Điều khiển trong OpenArm scene độc lập",
            "",
            "Mỗi case có 10 reset seeds giống nhau, 300 observation ticks/episode (~10 simulated seconds), checkpoint seed7 cố định. Success là cube lift ≥8cm trong 10 ticks; đây là criterion của lab. Scene/cameras/reset/gains chưa được xác minh giống demonstration dataset.",
            "",
            "| Case | Success | Wilson CI95 | Mean tracking error, rad | Policy latency, ms | Inference deadline misses |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in control:
        lo, hi = r["success_ci95_wilson"]
        rows.append(
            f"| {r['case']} ({r['model']}, {r['policy_hz']}Hz) | {round(r['success_rate'] * r['episodes'])}/{r['episodes']} | [{lo:.3f}, {hi:.3f}] | {r['joint_tracking_rmse_rad']:.5f} | {r['policy_latency_ms']:.3f} | {r['policy_deadline_misses']} |"
        )
    rows.extend(
        [
            "",
            "Tracking metric là mean của per-tick joint RMSE. Clipping counter của rollout reset ở mỗi episode; followup summary ghi rõ count của **episode cuối**, không gọi đó là tổng 10 episode. Inference deadline chỉ đo policy call, chưa gồm render/IPC/physics. Không đạt task trong các trial này không chứng minh mọi policy đều không thể học task; mismatch scene vẫn là yếu tố chưa kiểm soát.",
            "",
            "Chín controller fault cases đã chạy: fresh, đúng timeout, stale, NaN target, Inf target, NaN clock, negative dt, NaN state và sai shape. Hold/rejection/slew bound đều kiểm tra trực tiếp; không phải hardware safety certification.",
            "",
            "## Compute và kết luận có giới hạn",
            "",
            f"Telemetry main study: utilization trung bình {stats['gpu_util_percent']['mean']:.1f}%, p95 {stats['gpu_util_percent']['p95']:.1f}%, peak {stats['gpu_util_percent']['max']:.0f}%; total VRAM peak {stats['memory_used_mib']['max']:.0f} MiB (gồm desktop/context).",
            "Peak allocated tensor memory và thời gian từng epoch nằm trong training_runs.json. Epoch time gồm validation nhưng không gồm toàn bộ loading/checkpoint I/O. GPU telemetry không bao gồm followup transfer/rollouts; không đồng nhất tensor allocation với nvidia-smi usage.",
            "",
            "Đã hoàn tất một bộ nghiên cứu pilot có code, đối chứng, nhiều seed, test lock, measured negative results và reproduction commands. Chưa hoàn tất xác nhận rộng trên task/scene mới, full factorial sweep, genuine cross-task APT, large video world model hoặc learned hierarchical humanoid control. Các mục đó cần dataset/benchmark/compute và protocol độc lập, không thể thay bằng suy đoán.",
            "",
            "[Reproduction](reproduction.md) · [Limitations](limitations.md) · [Historical migration evidence](validation.md)",
            "",
        ]
    )
    if args.camera_study:
        camera = load(Path(args.camera_study) / "summary.json")
        extra = [
            "## Camera/độ phân giải lớn hơn — validation only",
            "",
            "Chín run fresh: ba seed cho 2view32, 2view64, 5view64, mỗi run 3 epochs và batch64/accumulation1. Dùng family đã khóa từ main validation; không evaluate lại test. Numeric trajectories và dataset revision giữa hai cache được kiểm tra equality. Đây là exploratory comparison có matched budget riêng, không so trực tiếp với main 12-epoch score.",
            "",
            "| Camera/resolution | Validation MSE rad² [CI95] | Difference vs two32 [CI95] |",
            "|---|---:|---:|",
        ]
        for name, value in camera["summary"].items():
            extra.append(
                f"| {name} | {interval(value)} | {interval(value['paired_difference_vs_two32'])} |"
            )
        extra.extend(
            [
                "",
                "Camera count thay đổi input information và model processing/capacity; ba epochs chỉ là pilot. Không dùng kết quả này để sửa family/checkpoint đã chọn hoặc retune bằng final-test labels.",
                "",
            ]
        )
        where = rows.index("## Compute và kết luận có giới hạn")
        rows[where:where] = extra
        if args.camera_telemetry:
            with Path(args.camera_telemetry).open() as handle:
                samples = list(csv.DictReader(handle))
            summary = {}
            for key in ["gpu_util_percent", "memory_used_mib", "temperature_c", "power_w"]:
                values = np.array([float(r[key]) for r in samples])
                summary[key] = {
                    "mean": float(values.mean()),
                    "p95": float(np.percentile(values, 95)),
                    "max": float(values.max()),
                }
            write_json(
                out / "camera_gpu_telemetry_summary.json",
                {
                    "samples": len(samples),
                    "scope": "camera study only; total GPU includes desktop/context",
                    **summary,
                },
            )
            (out / "camera_gpu_telemetry.csv").write_text(Path(args.camera_telemetry).read_text())
            rows.insert(
                -2,
                f"Camera study GPU utilization peak {summary['gpu_util_percent']['max']:.0f}%, p95 {summary['gpu_util_percent']['p95']:.0f}%; total VRAM peak {summary['memory_used_mib']['max']:.0f} MiB.",
            )
    Path(args.report).write_text("\n".join(rows))
    hashes = {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(
        out / "artifact_hashes.json",
        {
            "sha256": hashes,
            "report_generator_sha256": file_hash(__file__),
            "scope": "small text measurements only; raw data, model weights and environments remain ignored",
        },
    )
    print(
        json.dumps(
            {
                "training_runs": len(training),
                "selected": selection["selected_family"],
                "report": args.report,
                "artifact_sha256": hashlib.sha256(
                    json.dumps(hashes, sort_keys=True).encode()
                ).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--followups", required=True)
    p.add_argument("--telemetry", required=True)
    p.add_argument("--camera-study")
    p.add_argument("--camera-telemetry")
    p.add_argument("--output", default="docs/results/research_20261108")
    p.add_argument("--figure-dir", default="docs/figures")
    p.add_argument("--report", default="docs/research_report.md")
    run(p.parse_args())
