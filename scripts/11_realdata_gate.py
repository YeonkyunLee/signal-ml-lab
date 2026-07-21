"""실데이터 안전게이트: 합성 학습 앙상블이 '실데이터를 낯설다'고 플래그하는가?

08에서 합성 학습 모델은 실데이터로 잘 전이되지 못했다(sim-to-real 격차). 07의
앙상블 불확실도가 그 실패를 *사전에* 감지할 수 있다면, 배포 시 실데이터 입력을
자동으로 '고전 폴백/사람 검토'로 돌릴 수 있다.

합성으로 K개 모델을 학습 → 합성(in-dist) vs 실(OOD) 입력의 앙상블 불일치 비교.

    python scripts/11_realdata_gate.py --members 4 --epochs 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_ml_lab import dataset, metrics, realdata  # noqa: E402
from signal_ml_lab.models import DnCNN1D  # noqa: E402
from signal_ml_lab.train import TrainConfig, denoise, train_model  # noqa: E402


def ensemble_std(models, noisy):
    preds = np.stack([denoise(m, noisy) for m in models])
    return preds.std(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()

    # 합성 학습 앙상블
    members = []
    for k in range(args.members):
        c, n = dataset.make_windows(4000, seed=200 + k)
        m = DnCNN1D(depth=8, channels=32, kernel=9)
        print(f"[train] synthetic ensemble member {k+1}/{args.members}")
        train_model(m, dataset.split(c, n), TrainConfig(epochs=args.epochs, seed=k))
        members.append(m)

    # in-dist(합성) vs OOD(실데이터)
    id_c, id_n = dataset.make_windows(300, seed=9001)
    rd_c, rd_n = realdata.make_real_pairs(300, records=("100", "103", "115", "215"), seed=7)

    id_u = ensemble_std(members, id_n).max(axis=1)
    rd_u = ensemble_std(members, rd_n).max(axis=1)
    a = metrics.auroc(rd_u, id_u)
    thr = np.quantile(id_u, 0.95)
    det = (rd_u > thr).mean()

    print("\n=== 실데이터 OOD 게이트 (합성 학습 앙상블) ===")
    print(f"in-dist(합성) 불일치 mean = {id_u.mean():.4f}")
    print(f"OOD(실데이터) 불일치 mean = {rd_u.mean():.4f}  ({rd_u.mean()/id_u.mean():.2f}x)")
    print(f"실데이터 판별 AUROC = {a:.3f}")
    print(f"게이트(in-dist 95%)로 실데이터 탐지율 = {det*100:.1f}%  (오탐 5%)")

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(id_u, bins=30, alpha=0.6, density=True, label="in-dist (synthetic)")
    ax.hist(rd_u, bins=30, alpha=0.6, density=True, label="OOD (real data)")
    ax.axvline(thr, color="k", ls="--", lw=1, label="gate (in-dist 95%)")
    ax.set_xlabel("ensemble disagreement (peak std)")
    ax.set_ylabel("density")
    ax.set_title(f"Real data flagged as OOD by synthetic ensemble (AUROC={a:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "11_realdata_gate.png", dpi=130)
    print(f"\n[plot] {outdir / '11_realdata_gate.png'}")


if __name__ == "__main__":
    main()
