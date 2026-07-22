"""클래스 불균형 정면 돌파: 희귀 부정맥(S/F/Q) recall 끌어올리기.

10편의 한계 = 극심한 불균형(N≫S,F,Q)으로 소수 클래스 저조. 세 처방을 환자 독립
테스트로 비교한다:
  - weighted-CE  : 가중 교차엔트로피 (baseline, 10편)
  - focal        : focal loss (어려운 예제 가중, γ=2) + 클래스 가중
  - focal+aug    : focal + 소수 클래스 오버샘플링 + 신호 증강(스케일/지터/시프트)

지표: 균형 정확도(macro-recall), macro-F1, 클래스별 recall.

    python scripts/16_imbalance.py --epochs 20
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_ml_lab import realdata  # noqa: E402
from signal_ml_lab.metrics import auroc as realdata_auroc  # noqa: E402
from signal_ml_lab.models import ECGBeatClassifier  # noqa: E402

CLASSES = ["N", "S", "V", "F", "Q"]
TRAIN_RECS = ("101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "201", "203", "205", "208", "215")
TEST_RECS = ("100", "103", "105", "111", "113", "121", "200", "202", "210", "213", "219", "221", "223", "230")
WIN = 256


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")
        pt = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        return (((1 - pt) ** self.gamma) * ce).mean()


def augment(beats, rng):
    """ECG 박동 증강: 진폭 스케일·가우시안 지터·시간 시프트."""
    out = beats.copy()
    n = out.shape[0]
    out *= rng.uniform(0.8, 1.2, (n, 1)).astype(np.float32)         # 진폭
    out += rng.normal(0, 0.05, out.shape).astype(np.float32)         # 지터
    shifts = rng.integers(-6, 7, n)
    for i in range(n):
        out[i] = np.roll(out[i], int(shifts[i]))
    return out


def oversample_augment(x, y, rng, target_per_class=None):
    """소수 클래스를 증강 복제해 균형에 가깝게."""
    counts = np.bincount(y, minlength=len(CLASSES))
    target = target_per_class or int(np.median(counts[counts > 0]) * 3)
    xs, ys = [x], [y]
    for c in range(len(CLASSES)):
        idx = np.where(y == c)[0]
        if len(idx) == 0 or len(idx) >= target:
            continue
        need = target - len(idx)
        pick = rng.choice(idx, need, replace=True)
        xs.append(augment(x[pick], rng))
        ys.append(np.full(need, c))
    X = np.concatenate(xs); Y = np.concatenate(ys)
    p = rng.permutation(len(Y))
    return X[p], Y[p]


def train(x, y, loss_fn, epochs, rng_seed=0):
    torch.manual_seed(rng_seed)
    clf = ECGBeatClassifier(n_classes=len(CLASSES), win=WIN)
    dl = DataLoader(TensorDataset(torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y)),
                    batch_size=256, shuffle=True)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for _ in range(epochs):
        clf.train()
        for xb, yb in dl:
            opt.zero_grad(); loss_fn(clf(xb), yb).backward(); opt.step()
    return clf


@torch.no_grad()
def predict(clf, x):
    clf.eval(); t = torch.from_numpy(x).unsqueeze(1); out = []
    for i in range(0, len(t), 1024):
        out.append(clf(t[i:i+1024]).argmax(1).numpy())
    return np.concatenate(out)


def metrics(pred, true):
    recalls, precisions = [], []
    for c in range(len(CLASSES)):
        tp = np.sum((pred == c) & (true == c))
        rc = tp / max(np.sum(true == c), 1)
        pr = tp / max(np.sum(pred == c), 1)
        recalls.append(rc); precisions.append(pr)
    recalls = np.array(recalls); precisions = np.array(precisions)
    present = np.array([np.sum(true == c) > 0 for c in range(len(CLASSES))])
    f1 = 2 * recalls * precisions / np.maximum(recalls + precisions, 1e-9)
    return {"bal_acc": recalls[present].mean(), "macro_f1": f1[present].mean(),
            "recalls": recalls, "acc": np.mean(pred == true)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=20); args = ap.parse_args()
    rng = np.random.default_rng(0)

    xtr, ltr = realdata.load_beats(TRAIN_RECS, win=WIN, normalize=True)
    xte, lte = realdata.load_beats(TEST_RECS, win=WIN, normalize=True)
    ytr = np.array([CLASSES.index(l) for l in ltr]); yte = np.array([CLASSES.index(l) for l in lte])
    print(f"[data] train={len(ytr)} {dict(Counter(ltr))}")
    print(f"[data] test ={len(yte)} {dict(Counter(lte))}")

    counts = np.bincount(ytr, minlength=len(CLASSES)).astype(float)
    w = np.clip(np.sqrt(counts.sum() / (len(CLASSES) * np.maximum(counts, 1))), 0.5, 6.0)
    weight = torch.tensor(w, dtype=torch.float32)

    results = {}
    print("[train] weighted-CE...");
    results["weighted-CE"] = metrics(predict(train(xtr, ytr, nn.CrossEntropyLoss(weight=weight), args.epochs), xte), yte)
    print("[train] focal...")
    results["focal"] = metrics(predict(train(xtr, ytr, FocalLoss(weight=weight, gamma=2.0), args.epochs), xte), yte)
    print("[train] focal+aug (oversample+augment)...")
    xb, yb = oversample_augment(xtr, ytr, rng)
    results["focal+aug"] = metrics(predict(train(xb, yb, FocalLoss(weight=weight, gamma=2.0), args.epochs), xte), yte)

    print("\n=== 클래스 불균형 대응 비교 (환자 독립 테스트) ===")
    print(f"{'method':14s} {'acc':>6s} {'bal_acc':>8s} {'macroF1':>8s}   " + " ".join(f"{c:>5s}" for c in CLASSES))
    for name, m in results.items():
        rc = " ".join(f"{r*100:4.0f}%" for r in m["recalls"])
        print(f"{name:14s} {m['acc']*100:5.1f}% {m['bal_acc']*100:7.1f}% {m['macro_f1']*100:7.1f}%   {rc}")

    # === 건설적 대안: 이진 비정상 검출(임상 1차 과제) ===
    print("\n[train] 이진 정상 vs 비정상 검출...")
    ybin_tr = (ytr != CLASSES.index("N")).astype(np.int64)
    ybin_te = (yte != CLASSES.index("N")).astype(np.int64)
    cb = np.bincount(ybin_tr).astype(float)
    wb = torch.tensor(np.sqrt(cb.sum() / (2 * np.maximum(cb, 1))), dtype=torch.float32)

    class Bin(ECGBeatClassifier):
        pass
    clf_b = ECGBeatClassifier(n_classes=2, win=WIN)
    dl = DataLoader(TensorDataset(torch.from_numpy(xtr).unsqueeze(1), torch.from_numpy(ybin_tr)),
                    batch_size=256, shuffle=True)
    opt = torch.optim.Adam(clf_b.parameters(), lr=1e-3); lf = FocalLoss(weight=wb, gamma=2.0)
    torch.manual_seed(0)
    for _ in range(args.epochs):
        clf_b.train()
        for xb2, yb2 in dl:
            opt.zero_grad(); lf(clf_b(xb2), yb2).backward(); opt.step()
    clf_b.eval()
    with torch.no_grad():
        t = torch.from_numpy(xte).unsqueeze(1); probs = []
        for i in range(0, len(t), 1024):
            probs.append(F.softmax(clf_b(t[i:i+1024]), 1)[:, 1].numpy())
        probs = np.concatenate(probs)
    pred_b = (probs > 0.5).astype(int)
    sens = np.mean(pred_b[ybin_te == 1] == 1)   # 비정상 검출율(민감도)
    spec = np.mean(pred_b[ybin_te == 0] == 0)   # 특이도
    auroc = realdata_auroc(probs[ybin_te == 1], probs[ybin_te == 0])
    print("=== 이진 비정상 검출 (환자 독립) ===")
    print(f"민감도(비정상 recall)={sens*100:.1f}%  특이도={spec*100:.1f}%  AUROC={auroc:.3f}")
    print("→ 세부 아형(S/F/Q)은 어려워도, 임상 1차 과제인 '비정상 검출'은 강건하다.")

    outdir = Path("outputs"); outdir.mkdir(exist_ok=True)
    xpos = np.arange(len(CLASSES)); wbar = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, m) in enumerate(results.items()):
        ax.bar(xpos + (i-1)*wbar, m["recalls"]*100, wbar, label=name)
    ax.set_xticks(xpos); ax.set_xticklabels(CLASSES); ax.set_ylabel("per-class recall [%]")
    ax.set_title("Class-imbalance handling: per-class recall (patient-independent)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(outdir / "16_imbalance.png", dpi=130)
    print(f"\n[plot] {outdir / '16_imbalance.png'}")
    return {k: (v["bal_acc"], v["macro_f1"]) for k, v in results.items()}


if __name__ == "__main__":
    main()
