"""불확실도 추정으로 OOD 출력에 플래그 달기.

도메인 시프트(04)에서 ML은 '깨끗해 보이는 오답'을 냈다 — 틀렸다는 신호 없이.
여기서는 두 불확실도 추정법을 비교한다:
  (a) MC-dropout  — dropout을 추론 시에도 켜 예측 분산을 얻음
  (b) deep ensemble — 서로 다른 시드로 학습한 K개 모델의 불일치

불확실도가 OOD 입력(변형 형태)에서 실제로 커지면 = 모델이 '자기가 모를 때를 안다'
= 안전 게이트로 쓸 수 있다.

    python scripts/07_uncertainty.py --members 4 --epochs 25
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

from signal_ml_lab import dataset  # noqa: E402
from signal_ml_lab.metrics import auroc  # noqa: E402
from signal_ml_lab.models import DnCNN1D  # noqa: E402
from signal_ml_lab.train import TrainConfig, denoise, train_model  # noqa: E402


@torch.no_grad()
def mc_dropout_std(model, noisy, T=30):
    model.train()
    x = torch.from_numpy(np.asarray(noisy, np.float32)).unsqueeze(1)
    preds = torch.stack([model(x).squeeze(1) for _ in range(T)])
    return preds.std(0).numpy()


def ensemble_std(models, noisy):
    preds = np.stack([denoise(m, noisy) for m in models])  # (K, N, L)
    return preds.std(axis=0)


def report(name, id_std, ood_std):
    # 윈도우당 점수: 최대 불일치(peak)가 미묘한 OOD에 더 민감
    id_u = id_std.max(axis=1)
    ood_u = ood_std.max(axis=1)
    a = auroc(ood_u, id_u)
    thr = np.quantile(id_u, 0.95)
    det = (ood_u > thr).mean()
    print(f"\n[{name}]")
    print(f"  in-dist  peak-std mean = {id_u.mean():.4f}")
    print(f"  OOD      peak-std mean = {ood_u.mean():.4f}  ({ood_u.mean()/id_u.mean():.2f}x)")
    print(f"  OOD AUROC = {a:.3f}   탐지율@5%오탐 = {det*100:.1f}%")
    return id_u, ood_u, a, thr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()

    # 테스트셋: in-dist vs OOD(변형)
    id_c, id_n = dataset.make_windows(300, seed=9001)
    ood_c, ood_n = dataset.make_mixed_windows(300, variant_frac=1.0, seed=9002)

    # (a) MC-dropout 모델 1개
    m_drop = DnCNN1D(depth=8, channels=32, kernel=9, dropout=0.1)
    print("[train] MC-dropout 모델")
    data0 = dataset.split(*dataset.make_windows(args.n, seed=0))
    train_model(m_drop, data0, TrainConfig(epochs=args.epochs))
    mc_id = mc_dropout_std(m_drop, id_n)
    mc_ood = mc_dropout_std(m_drop, ood_n)

    # (b) deep ensemble: 서로 다른 시드로 K개 학습
    members = []
    for k in range(args.members):
        data_k = dataset.split(*dataset.make_windows(args.n, seed=100 + k))
        m = DnCNN1D(depth=8, channels=32, kernel=9)
        print(f"[train] ensemble member {k+1}/{args.members}")
        train_model(m, data_k, TrainConfig(epochs=args.epochs, seed=k))
        members.append(m)
    en_id = ensemble_std(members, id_n)
    en_ood = ensemble_std(members, ood_n)

    print("\n=== OOD 탐지: 불확실도 방법 비교 ===")
    report("MC-dropout", mc_id, mc_ood)
    id_u, ood_u, a_en, thr = report(f"deep ensemble (K={args.members})", en_id, en_ood)

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(id_u, bins=30, alpha=0.6, label="in-distribution", density=True)
    ax.hist(ood_u, bins=30, alpha=0.6, label="OOD (variant)", density=True)
    ax.axvline(thr, color="k", ls="--", lw=1, label="gate (in-dist 95%)")
    ax.set_xlabel("ensemble disagreement (peak std)")
    ax.set_ylabel("density")
    ax.set_title(f"Deep ensemble flags OOD  (AUROC={a_en:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "07_uncertainty.png", dpi=130)
    print(f"\n[plot] {outdir / '07_uncertainty.png'}")


if __name__ == "__main__":
    main()
