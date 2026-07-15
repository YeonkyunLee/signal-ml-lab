"""혼합 분포 학습으로 도메인 시프트 완화.

원분포로만 학습한 모델 vs 원+변형 혼합 학습 모델을, 두 테스트셋(원/변형) 모두에서
비교한다. 일반화 비용을 학습 데이터 다양성으로 얼마나 되사는지 본다.

    python scripts/05_mixed_training.py --epochs 30 --n 4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_ml_lab import dataset, metrics  # noqa: E402
from signal_ml_lab.models import DnCNN1D  # noqa: E402
from signal_ml_lab.train import TrainConfig, denoise, train_model  # noqa: E402


def eval_on(model, clean, noisy):
    est = denoise(model, noisy)
    return float(np.mean([metrics.snr_db(c, e) for c, e in zip(clean, est)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    # 공통 테스트셋 두 개 (원분포 / 변형분포), 학습과 다른 시드
    te_orig_c, te_orig_n = dataset.make_windows(
        400, noise_scale_range=(0.5, 1.5), seed=9001
    )
    te_var_c, te_var_n = dataset.make_mixed_windows(
        400, variant_frac=1.0, noise_scale_range=(0.5, 1.5), seed=9002
    )

    def train_one(mixed: bool):
        if mixed:
            c, n = dataset.make_mixed_windows(args.n, variant_frac=0.5, seed=0)
        else:
            c, n = dataset.make_windows(args.n, seed=0)
        data = dataset.split(c, n)
        model = DnCNN1D(depth=8, channels=32, kernel=9)
        print(f"\n--- {'MIXED' if mixed else 'ORIGINAL-ONLY'} 학습 ---")
        train_model(model, data, TrainConfig(epochs=args.epochs, batch_size=64, lr=1e-3))
        return model

    m_orig = train_one(mixed=False)
    m_mixed = train_one(mixed=True)

    rows = [
        {
            "model": "original-only",
            "test=original": round(eval_on(m_orig, te_orig_c, te_orig_n), 2),
            "test=variant": round(eval_on(m_orig, te_var_c, te_var_n), 2),
        },
        {
            "model": "mixed",
            "test=original": round(eval_on(m_mixed, te_orig_c, te_orig_n), 2),
            "test=variant": round(eval_on(m_mixed, te_var_c, te_var_n), 2),
        },
    ]
    df = pd.DataFrame(rows)
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    df.to_csv(outdir / "05_mixed_training.csv", index=False)
    torch.save(m_mixed.state_dict(), outdir / "dncnn1d_mixed.pt")

    print("\n=== 혼합 학습 효과 (출력 SNR dB) ===\n")
    print(df.to_string(index=False))
    print(f"\n[csv] {outdir / '05_mixed_training.csv'}")
    print(f"[model] {outdir / 'dncnn1d_mixed.pt'}")


if __name__ == "__main__":
    main()
