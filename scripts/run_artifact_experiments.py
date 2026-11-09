"""2.5 (phần 2): thí nghiệm CNN cho 4 vấn đề lông / viền tối / quá sáng / sáng không đều (src/preprocess/artifacts.py).

Khác phần nhiễu–tương phản: KHÔNG sàng lọc 1 seed (2.5.6 cho thấy sàng lọc 1 seed lạc quan) → mọi cấu hình chạy
thẳng 3 seed (42, 43, 44). Baseline được chạy lại trên CÙNG máy để so sánh ghép cặp công bằng.
Chạy lại lệnh sẽ bỏ qua seed đã có metrics.json (Ctrl+C an toàn, chạy lại để tiếp tục).

    python -X utf8 scripts/run_artifact_experiments.py smoke          # subset 64 ảnh, 1 epoch, kiểm tra pipeline
    python -X utf8 scripts/run_artifact_experiments.py all            # baseline + 8 cấu hình × 3 seed
    python -X utf8 scripts/run_artifact_experiments.py run NAME [seeds]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import ARTIFACT_CANDIDATES, get_preprocess  # noqa: E402
from src.training import run_experiment  # noqa: E402

LOG = Path("outputs/preprocess_artifacts/experiments.log")
ORDER = ["baseline", *ARTIFACT_CANDIDATES]


def run(name, seeds=(42, 43, 44), **kw):
    fn = None if name == "baseline" else get_preprocess(name)
    df = run_experiment(name, preprocess_fn=fn, seeds=seeds, log_file=LOG, n_jobs=24,
                        description=f"2.5 lông/viền/quá sáng/sáng không đều: {name}", **kw)
    print(name, df[["seed", "val_macro_f1", "test_macro_f1"]].round(4).to_dict("records"), flush=True)


if __name__ == "__main__":
    LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "smoke":
        for n in ORDER:
            run(n, seeds=(42,), subset=64, epochs=1)
    elif cmd == "all":
        for n in ORDER:
            run(n)
    elif cmd == "run":
        run(args[0], seeds=tuple(int(s) for s in args[1:]) or (42, 43, 44))
    else:
        raise SystemExit(f"lệnh không hợp lệ: {cmd}")
