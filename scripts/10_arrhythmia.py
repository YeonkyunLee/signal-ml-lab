"""부정맥 분류: MIT-BIH 심박 → AAMI 클래스 (환자 독립 분할).

디노이징을 넘어 '진단' 쪽으로 한 걸음. R-피크 주석으로 박동을 잘라 5클래스(N/S/V/F/Q)
로 분류한다. train/test를 레코드(환자)로 분리해 누수를 막는다 — 실전 성능의 정직한 척도.

    python scripts/10_arrhythmia.py --epochs 20
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_ml_lab import realdata  # noqa: E402
from signal_ml_lab.models import ECGBeatClassifier  # noqa: E402

CLASSES = ["N", "S", "V", "F", "Q"]
# 환자 독립 분할 (서로 다른 레코드)
TRAIN_RECS = ("101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "201", "203", "205", "208", "215")
TEST_RECS = ("100", "103", "105", "111", "113", "121", "200", "202", "210", "213", "219", "221", "223", "230")


def to_xy(beats, labels):
    y = np.array([CLASSES.index(l) for l in labels], dtype=np.int64)
    x = torch.from_numpy(beats).unsqueeze(1)
    return x, torch.from_numpy(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--win", type=int, default=256)
    args = ap.parse_args()

    print("[data] 박동 추출 (환자 독립 분할, 캐시)...")
    tr_b, tr_l = realdata.load_beats(TRAIN_RECS, win=args.win)
    te_b, te_l = realdata.load_beats(TEST_RECS, win=args.win)
    print(f"[data] train={len(tr_l)} beats  test={len(te_l)} beats")
    print(f"[data] train 분포: {dict(Counter(tr_l))}")
    print(f"[data] test  분포: {dict(Counter(te_l))}")

    xtr, ytr = to_xy(tr_b, tr_l)
    xte, yte = to_xy(te_b, te_l)

    # 클래스 불균형 → 가중 손실
    counts = np.bincount(ytr.numpy(), minlength=len(CLASSES)).astype(float)
    weights = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32)

    model = ECGBeatClassifier(n_classes=len(CLASSES), win=args.win)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] ECGBeatClassifier params={n_params:,}")

    torch.manual_seed(0)
    dl = DataLoader(TensorDataset(xtr, ytr), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1:3d}/{args.epochs}  train loss={tot/len(dl.dataset):.4f}")

    # 평가
    model.eval()
    with torch.no_grad():
        logits = []
        for i in range(0, xte.size(0), 512):
            logits.append(model(xte[i : i + 512]))
        pred = torch.cat(logits).argmax(1).numpy()
    true = yte.numpy()

    acc = (pred == true).mean()
    print(f"\n=== 부정맥 분류 (환자 독립 테스트) ===")
    print(f"전체 정확도: {acc*100:.2f}%")
    cm = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    print("\n혼동행렬 (행=참, 열=예측):")
    print("      " + "  ".join(f"{c:>5s}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>4s}  " + "  ".join(f"{cm[i,j]:5d}" for j in range(len(CLASSES))))
    print("\n클래스별 recall / precision:")
    for i, c in enumerate(CLASSES):
        rec = cm[i, i] / max(cm[i].sum(), 1)
        prec = cm[i, i] / max(cm[:, i].sum(), 1)
        print(f"  {c}: recall={rec*100:5.1f}%  precision={prec*100:5.1f}%  (n={cm[i].sum()})")

    # 혼동행렬 그림 (행 정규화)
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"MIT-BIH beat classification (acc={acc*100:.1f}%, patient-independent)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(outdir / "10_arrhythmia_cm.png", dpi=130)
    torch.save(model.state_dict(), outdir / "beat_classifier.pt")
    print(f"\n[plot] {outdir / '10_arrhythmia_cm.png'}")


if __name__ == "__main__":
    main()
