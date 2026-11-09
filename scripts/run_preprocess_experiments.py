"""Giai đoạn 2 của 2.5: thí nghiệm CNN cho khử nhiễu / tăng tương phản (kế hoạch ở 2.5a_nhieu_tuong_phan.ipynb, mục E).

Chạy (từ thư mục gốc repo, python của .venv):
    python -X utf8 scripts/run_preprocess_experiments.py smoke              # chạy thử subset nhỏ, 1 epoch
    python -X utf8 scripts/run_preprocess_experiments.py screen             # bước 1: seed 42, 4 cấu hình
    python -X utf8 scripts/run_preprocess_experiments.py confirm A B        # bước 2: seed 42–44 cho cấu hình A, B
    python -X utf8 scripts/run_preprocess_experiments.py run NAME [seeds]   # 1 cấu hình bất kỳ (vd. kết hợp, tùy chọn)

Tên thí nghiệm = tên trong src.preprocess.CANDIDATES (kết hợp: "a+b", áp dụng trái → phải).
Chỉ đọc val macro-F1 khi chọn; test chỉ để báo cáo.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import get_preprocess  # noqa: E402
from src.training import run_experiment  # noqa: E402

SCREEN = ["gauss3", "median3", "stretch_L_1_99", "clahe_L_c2_t8"]
LOG = Path("outputs/preprocess/experiments.log")


def run(name, seeds=(42,), **kw):
    fn = get_preprocess(name)
    df = run_experiment(name, preprocess_fn=fn, seeds=seeds, log_file=LOG,
                        description=f"2.5 tiền xử lý: {name} (src.preprocess)", **kw)
    print(name, df[["seed", "val_macro_f1"]].to_dict("records"), flush=True)
    return df


if __name__ == "__main__":
    if __package__ is None:
        pass
    LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "smoke":
        for n in ["median3", "clahe_L_c2_t8", "median3+clahe_L_c2_t8"]:
            run(n, seeds=(42,), subset=64, epochs=1)
    elif cmd == "screen":
        for n in SCREEN:
            run(n, seeds=(42,))
    elif cmd == "confirm":
        for n in args:
            run(n, seeds=(42, 43, 44))
    elif cmd == "run":
        seeds = tuple(int(s) for s in args[1:]) or (42, 43, 44)
        run(args[0], seeds=seeds)
    else:
        raise SystemExit(f"lệnh không hợp lệ: {cmd}")
