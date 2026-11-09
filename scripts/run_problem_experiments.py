"""Train lại 2.5 cho nhiễu Gauss và tương phản thấp: baseline + 6 cấu hình × 3 seed, ghi vào outputs/rerun_2.5/.

Chạy từ gốc repo, tuần tự để RAM chỉ giữ một cache ảnh. Ctrl+C an toàn: chạy lại sẽ bỏ qua seed đã có metrics.json.
Thiết lập GPU giữ theo src.training (AMP, cudnn.benchmark, ảnh trong RAM); batch 32 thuộc cấu hình đóng băng nên không đổi.

    python -X utf8 scripts/run_problem_experiments.py smoke     # 64 ảnh, 1 epoch, kiểm tra pipeline
    python -X utf8 scripts/run_problem_experiments.py all
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess.problem_report import TRAIN_ORDER, train  # noqa: E402

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "smoke":
        print(train(TRAIN_ORDER, seeds=(42,), subset=64, epochs=1).round(4).to_string())
    elif cmd == "all":
        print(train(TRAIN_ORDER).round(4).to_string())
    else:
        raise SystemExit("dùng: smoke | all")
