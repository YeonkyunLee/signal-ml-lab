"""실데이터 검증: 합성으로 학습한 모델이 실제 ECG+실제 노이즈에 통하는가?

MIT-BIH 실 ECG(기준) + NSTDB 실 노이즈(bw/ma/em)로 테스트셋을 만들고 비교한다:
  - classical (bandpass+notch)
  - ML(합성 학습)  ← 전이(transfer) 확인
  - ML(실데이터 학습) ← 상한선

    python scripts/08_real_data.py --train-real --epochs 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_ml_lab import classical, dataset, metrics, realdata  # noqa: E402
from signal_ml_lab.models import DnCNN1D  # noqa: E402
from signal_ml_lab.train import TrainConfig, denoise, train_model  # noqa: E402

FS = 360


def evalset(clean, est):
    return float(np.mean([metrics.snr_db(c, e) for c, e in zip(clean, est)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-real", action="store_true", help="실데이터로도 학습(상한선)")
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()

    # 실데이터 테스트셋 (학습에 안 쓴 레코드로 분리)
    print("[data] 실 ECG + 실 노이즈 테스트셋 생성(캐시)...")
    te_c, te_n = realdata.make_real_pairs(
        400, records=("100", "103", "115", "215"), seed=7
    )
    snr_in = evalset(te_c, te_n)
    print(f"[data] 입력 SNR(실노이즈) 평균 = {snr_in:.2f} dB")

    # 1) classical
    cls = np.stack([classical.bandpass_notch(x, fs=FS) for x in te_n])

    # 2) 합성 학습 모델 (기존 체크포인트) — 전이 확인
    outdir = Path("outputs")
    m_syn = DnCNN1D(depth=8, channels=32, kernel=9)
    ckpt = outdir / "dncnn1d.pt"
    have_syn = ckpt.exists()
    if have_syn:
        m_syn.load_state_dict(torch.load(ckpt, map_location="cpu"))
        syn = denoise(m_syn, te_n)

    rows = [
        ("classical (bp+notch)", evalset(te_c, cls)),
    ]
    if have_syn:
        rows.append(("ML (synthetic-trained)", evalset(te_c, syn)))

    # 3) 실데이터 학습 모델 (상한선)
    real = None
    if args.train_real:
        print("[train] 실데이터로 학습(다른 레코드)...")
        tr_c, tr_n = realdata.make_real_pairs(
            4000, records=("101", "106", "119", "208", "230"), seed=0
        )
        data = dataset.split(tr_c, tr_n)
        m_real = DnCNN1D(depth=8, channels=32, kernel=9)
        train_model(m_real, data, TrainConfig(epochs=args.epochs))
        torch.save(m_real.state_dict(), outdir / "dncnn1d_real.pt")
        real = denoise(m_real, te_n)
        rows.append(("ML (real-trained)", evalset(te_c, real)))

    print("\n=== 실데이터 디노이징 (실 ECG + 실 노이즈) ===")
    print(f"{'method':26s} {'output SNR':>10s} {'gain':>8s}")
    print(f"{'noisy (input)':26s} {snr_in:10.2f} {'-':>8s}")
    for name, snr in rows:
        print(f"{name:26s} {snr:10.2f} {snr - snr_in:+8.2f}")

    # 정성 그림
    fig, axes = plt.subplots(2 + len(rows), 1, figsize=(10, 2 + 1.5 * (2 + len(rows))), sharex=True)
    w = slice(0, 720)
    axes[0].plot(te_n[0][w], lw=0.7); axes[0].set_ylabel("noisy", fontsize=8)
    axes[1].plot(te_c[0][w], lw=0.7, color="green"); axes[1].set_ylabel("real ECG", fontsize=8)
    series = {"classical": cls[0]}
    if have_syn:
        series["ML synthetic"] = syn[0]
    if real is not None:
        series["ML real"] = real[0]
    for ax, (name, sig) in zip(axes[2:], series.items()):
        ax.plot(sig[w], lw=0.7, color="C1"); ax.set_ylabel(name, fontsize=8)
    axes[-1].set_xlabel("sample")
    fig.suptitle("Real data: MIT-BIH ECG + NSTDB noise")
    fig.tight_layout()
    fig.savefig(outdir / "08_real_data.png", dpi=130)
    print(f"\n[plot] {outdir / '08_real_data.png'}")


if __name__ == "__main__":
    main()
