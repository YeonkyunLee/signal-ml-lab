"""ML 디노이저를 학습하고 가중치를 저장한다.

    python scripts/02_train_ml_denoise.py --epochs 30 --n 4000
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

from signal_ml_lab import dataset, metrics  # noqa: E402
from signal_ml_lab.models import DnCNN1D  # noqa: E402
from signal_ml_lab.train import TrainConfig, denoise, train_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="총 윈도우 수")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    print(f"[data] {args.n} windows x {args.win} samples 생성 중...")
    clean, noisy = dataset.make_windows(args.n, win_len=args.win, seed=0)
    data = dataset.split(clean, noisy)
    print(
        f"[data] train={data['train'][0].shape[0]} "
        f"val={data['val'][0].shape[0]} test={data['test'][0].shape[0]}"
    )

    model = DnCNN1D(depth=8, channels=32, kernel=9)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] DnCNN1D  params={n_params:,}")

    cfg = TrainConfig(epochs=args.epochs, batch_size=64, lr=1e-3)
    history = train_model(model, data, cfg)

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    ckpt = outdir / "dncnn1d.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"[model] 저장: {ckpt}")

    # 테스트셋에서 ML 성능 요약
    te_c, te_n = data["test"]
    est = denoise(model, te_n)
    snr_in = np.mean([metrics.snr_db(c, n) for c, n in zip(te_c, te_n)])
    snr_ml = np.mean([metrics.snr_db(c, e) for c, e in zip(te_c, est)])
    print(f"\n[test] noisy  SNR={snr_in:6.2f} dB")
    print(f"[test] ML     SNR={snr_ml:6.2f} dB  (gain +{snr_ml - snr_in:.2f} dB)")

    # 학습 곡선
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("DnCNN1D training")
    fig.tight_layout()
    fig.savefig(outdir / "02_training_curve.png", dpi=130)
    print(f"[plot] 저장: {outdir / '02_training_curve.png'}")


if __name__ == "__main__":
    main()
