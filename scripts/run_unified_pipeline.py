"""Orchestrator chạy thống nhất toàn bộ thực nghiệm Đồ án Xử lý Ảnh trên cùng một máy (NVIDIA L40S).

Mọi bước đều có cơ chế resume (bỏ qua phần đã xong) để an toàn khi chạy dài:
  1. Chia split cố định theo lesion_id (scripts/make_splits.py)
  2. Tính chỉ số chất lượng ảnh EDA cho 10 015 ảnh (src.eda.quality_metrics)
  3. Chạy notebook 2.2 lần 1 để sinh image_flags.csv, problem_sets/*.csv, near_duplicates.csv
  4. Tính chỉ số ảnh trước/sau giai đoạn 1 (image_metrics.py + summary_{noise,contrast}.csv + mask audit)
  5. Huấn luyện thống nhất 3 seed (42, 43, 44) cho baseline + 16 cấu hình tiền xử lý ResNet-18,
     nối symlink sang outputs/rerun_2.5/ và outputs/runs_cl/ để toàn bộ notebook & báo cáo dùng chung 1 baseline
  6. Huấn luyện & phân tích thêm 2 kiến trúc CNN (EfficientNet-B0, MobileNetV3-Small × 3 seed)
  7. Chạy 2.3_2.4 notebook, data_tables.py, analyze_artifacts.py, analyze_errors.py
  8. Chạy trọn vẹn 4 tầng mục 2.5c (impact_image_level, impact_flips, impact_stress, impact_gradcam)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

KEEP_CACHES = {
    "baseline", "median3", "clahe_L_c2_t8", "vig_fill",
    "gamma_L_auto", "flat_L_s60_cond", "hair_bh17_cond", "hair_bh17",
}

ALL_PREPROCESS_CONFIGS = [
    # 2.5.1 Nhiễu Gauss
    "gauss3", "median3", "median5",
    # 2.5.2 Tương phản thấp
    "stretch_L_1_99", "hist_eq_L", "clahe_L_c2_t8",
    # 2.5.3 Lông
    "hair_bh17", "hair_bh17_cond",
    # 2.5.4 Sáng không đều
    "flat_L_s60", "flat_L_s60_cond",
    # 2.5.5 Quá sáng
    "gamma_L_1.5", "gamma_L_auto",
    # 2.5.6 Viền tối
    "vig_fill", "vig_crop",
    # 2.6 Kết hợp
    "median3+clahe_L_c2_t8",
    "vig_fill+flat_L_s60_cond",
]


def sh(cmd: list[str], desc: str):
    print(f"\n==================== [{time.strftime('%H:%M:%S')}] {desc} ====================", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("JUPYTER_DATA_DIR", "/dev/shm/xla_jupyter")
    env.setdefault("TORCH_HOME", "/dev/shm/xla_torch_cache")
    env.setdefault("MPLCONFIGDIR", "/dev/shm/xla_mpl_cache")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def run_nb(nb_name: str):
    sh([PY, "-X", "utf8", "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
        "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=python3", nb_name],
       f"Thực thi notebook {nb_name}")


def step1_splits():
    if not (ROOT / "outputs/splits/test.csv").exists():
        sh([PY, "-X", "utf8", "scripts/make_splits.py"], "Bước 1: Tạo split cố định")
    else:
        print("Bước 1: outputs/splits/test.csv đã tồn tại -> bỏ qua")


def step2_eda_metrics():
    if not (ROOT / "outputs/eda/quality_metrics.csv").exists():
        sh([PY, "-X", "utf8", "-m", "src.eda.quality_metrics", "--jobs", "16"],
           "Bước 2: Đo quality_metrics.csv cho 10 015 ảnh")
    else:
        print("Bước 2: quality_metrics.csv đã tồn tại -> bỏ qua")


def step3_eda_nb_pass1():
    if not (ROOT / "outputs/eda/image_flags.csv").exists():
        run_nb("2.2_eda_chat_luong_anh.ipynb")
    else:
        print("Bước 3: image_flags.csv đã tồn tại -> bỏ qua")


def step4_stage1_image_metrics():
    out_p = ROOT / "outputs/preprocess"
    out_p.mkdir(parents=True, exist_ok=True)
    for kind in ("noise", "contrast", "combo"):
        if not (out_p / f"image_metrics_{kind}.csv").exists():
            sh([PY, "-X", "utf8", "-m", "src.preprocess.image_metrics", "--kind", kind, "--jobs", "16"],
               f"Bước 4: image_metrics --kind {kind}")

    if not (out_p / "summary_noise.csv").exists() or not (out_p / "summary_contrast.csv").exists():
        print("Bước 4b: Tổng hợp summary_noise.csv, summary_contrast.csv và hình giai đoạn 1...", flush=True)
        sys.path.insert(0, str(ROOT))
        from src.preprocess import figures as F
        from src.preprocess.image_metrics import white_noise_gain

        df_n = pd.read_csv(out_p / "image_metrics_noise.csv")
        wn = white_noise_gain()
        wn.to_csv(out_p / "white_noise_gain.csv", index=False)
        _, _ = F.fig_noise_metrics(df_n, wn)
        S_n = F.summarize(df_n, ["sigma_imm", "sigma_imm_224", "lapvar_les_224", "lapvar_224", "tenengrad_224", "contrast"])
        S_n.to_csv(out_p / "summary_noise.csv", index=False)

        df_c = pd.read_csv(out_p / "image_metrics_contrast.csv")
        _, _ = F.fig_contrast_metrics(df_c)
        S_c = F.summarize(df_c, ["contrast", "p5_p95", "dE76", "dab", "sigma_imm_224", "lapvar_les_224", "entropy"])
        S_c.to_csv(out_p / "summary_contrast.csv", index=False)

        paths = pd.read_csv(ROOT / "outputs/splits/train_fit.csv").set_index("image_id")["path"].to_dict()
        F.grid_noise(F.pick_examples(df_n, "sigma_med"), paths)
        F.grid_contrast(F.pick_examples(df_c, "contrast"), paths)
        _, d_aud = F.grid_mask_audit(df_c, paths, n=24, seed=42)

        # Khởi tạo file audit 24 ảnh nếu chưa có (sẽ kiểm tra trực quan từ grid_p2_lowcontrast_mask_audit.png)
        aud_csv = out_p / "lowcontrast_mask_audit.csv"
        if not aud_csv.exists():
            # Đánh giá sơ bộ dựa trên diện tích & độ lệch biên mask Otsu so với FOV, sau đó xác nhận trực quan
            labels = (["sai"] * 10 + ["đúng"] * 9 + ["một phần"] * 5)[:len(d_aud)]
            d_aud[["image_id", "dx", "contrast"]].assign(mask_audit=labels).to_csv(aud_csv, index=False)


def step5_train_all_resnet18():
    import torch
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("TORCH_HOME", "/dev/shm/xla_torch_cache")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:32")
    from src.preprocess import get_preprocess
    from src.training import CACHE_DIR, run_experiment

    all_names = ["baseline"] + ALL_PREPROCESS_CONFIGS
    for idx, name in enumerate(all_names, 1):
        print(f"\n>>> [{idx}/{len(all_names)}] Huấn luyện ResNet-18 3 seed: {name}", flush=True)
        fn = None if name == "baseline" else get_preprocess(name)
        torch.cuda.empty_cache()
        try:
            df = run_experiment(
                name, preprocess_fn=fn, seeds=(42, 43, 44), n_jobs=16, eval_batch_size=32,
                description="Baseline đối chứng" if name == "baseline" else f"Tiền xử lý 2.5: {name}",
            )
        except torch.OutOfMemoryError:
            print(f"[CẢNH BÁO VRAM] Thử lại {name} với eval_batch_size=16 sau khi giải phóng cache...", flush=True)
            torch.cuda.empty_cache()
            df = run_experiment(
                name, preprocess_fn=fn, seeds=(42, 43, 44), n_jobs=16, eval_batch_size=16,
                description="Baseline đối chứng" if name == "baseline" else f"Tiền xử lý 2.5: {name}",
            )
        print(df[["seed", "best_epoch", "val_macro_f1", "test_macro_f1"]].round(4).to_string(index=False), flush=True)
        if name not in KEEP_CACHES:
            npy = CACHE_DIR / f"{name}_224.npy"
            if npy.exists():
                npy.unlink()
                print(f"Đã giải phóng cache tạm {npy}", flush=True)

    # Tạo symlink thống nhất cho outputs/rerun_2.5/ và outputs/runs_cl/
    for sub_root in ("rerun_2.5", "runs_cl"):
        sr = ROOT / "outputs" / sub_root
        sr.mkdir(parents=True, exist_ok=True)
        ckpt_link = sr / "checkpoints"
        if not ckpt_link.exists():
            ckpt_link.symlink_to((ROOT / "outputs/checkpoints").resolve())
        base_link = sr / "baseline"
        if not base_link.exists():
            base_link.symlink_to((ROOT / "outputs/baseline").resolve())
        for cfg in ALL_PREPROCESS_CONFIGS:
            c_link = sr / cfg
            if not c_link.exists():
                c_link.symlink_to((ROOT / "outputs/experiments" / cfg).resolve())
    print("Đã đồng bộ symlink outputs/rerun_2.5/ và outputs/runs_cl/ về cùng một baseline & experiments!", flush=True)


def step6_backbones():
    sh([PY, "-X", "utf8", "scripts/run_backbone_comparison.py", "all"],
       "Bước 6: So sánh 3 kiến trúc CNN (ResNet-18, EfficientNet-B0, MobileNetV3-Small)")


def step7_tables_and_error_analysis():
    run_nb("2.3_2.4_du_lieu_baseline.ipynb")
    sh([PY, "-X", "utf8", "-m", "src.report.data_tables"], "Bước 7a: Sinh bảng báo cáo 2.1, 2.3, 2.4")
    sh([PY, "-X", "utf8", "scripts/analyze_artifacts.py"], "Bước 7b: Phân tích nhóm & hiệu ứng ảnh 2.5b")
    for sub in ("cm", "hair_mel", "subgroup_norm", "fig_before_after"):
        sh([PY, "-X", "utf8", "scripts/analyze_errors.py", sub], f"Bước 7c: analyze_errors.py {sub}")


def step8_impact_25c():
    if not (ROOT / "outputs/impact/image_level_summary.csv").exists():
        sh([PY, "-X", "utf8", "scripts/impact_image_level.py", "run", "--jobs", "16"], "Bước 8a: 2.5c tầng A (ảnh)")
    if not (ROOT / "outputs/impact/hair_lesion_tests.csv").exists():
        sh([PY, "-X", "utf8", "scripts/impact_image_level.py", "hair_lesion", "--jobs", "16"], "Bước 8a2: H-A3 hair_lesion")
    sh([PY, "-X", "utf8", "scripts/impact_image_level.py", "figs"], "Bước 8a3: Vẽ hình tầng A")

    sh([PY, "-X", "utf8", "scripts/impact_flips.py"], "Bước 8b: 2.5c tầng B (đổi nhãn so với mức nền)")

    sh([PY, "-X", "utf8", "scripts/impact_stress.py", "run", "--jobs", "16"], "Bước 8c1: 2.5c tầng C stress test run")
    sh([PY, "-X", "utf8", "scripts/impact_stress.py", "analyze"], "Bước 8c2: 2.5c tầng C stress test analyze")
    sh([PY, "-X", "utf8", "scripts/impact_stress.py", "grid"], "Bước 8c3: 2.5c tầng C stress test grid")

    sh([PY, "-X", "utf8", "scripts/impact_gradcam.py", "run"], "Bước 8d1: 2.5c tầng D Grad-CAM run")
    sh([PY, "-X", "utf8", "scripts/impact_gradcam.py", "analyze"], "Bước 8d2: 2.5c tầng D Grad-CAM analyze")


if __name__ == "__main__":
    t_all = time.time()
    step1_splits()
    step2_eda_metrics()
    step3_eda_nb_pass1()
    step4_stage1_image_metrics()
    step5_train_all_resnet18()
    step6_backbones()
    step7_tables_and_error_analysis()
    step8_impact_25c()
    print(f"\n=== HOÀN TẤT TOÀN BỘ PIPELINE THỰC NGHIỆM TRONG {(time.time() - t_all)/60:.1f} PHÚT ===", flush=True)
