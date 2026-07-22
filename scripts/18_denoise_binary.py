"""디노이징은 이진 비정상 검출에 도움이 되는가? (12편의 확장)

12편: SNR 디노이저가 5클래스 세부 진단을 해쳤다(비정상 형태 뭉갬). 그렇다면 더 거친
과제인 '정상 vs 비정상' 이진 검출에서는? 노이즈 세기별로 noisy vs denoised의 검출
AUROC를 비교해, 전처리 가치가 과제 세밀도·노이즈 수준에 따라 달라지는지 본다.

    python scripts/18_denoise_binary.py --epochs 18
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_ml_lab import realdata  # noqa: E402
from signal_ml_lab.metrics import auroc  # noqa: E402
from signal_ml_lab.models import DnCNN1D, ECGBeatClassifier  # noqa: E402
from signal_ml_lab.train import denoise  # noqa: E402

CLASSES = ["N", "S", "V", "F", "Q"]
TRAIN_RECS = ("101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "201", "203", "205", "208", "215")
TEST_RECS = ("100", "103", "105", "111", "113", "121", "200", "202", "210", "213", "219", "221", "223", "230")
WIN = 256


def zscore(b):
    return ((b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-6)).astype(np.float32)


@torch.no_grad()
def abn_prob(clf, bz):
    t = torch.from_numpy(bz).unsqueeze(1); out = []
    for i in range(0, len(t), 1024):
        out.append(F.softmax(clf(t[i:i+1024]), 1)[:, 1].numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=18); args = ap.parse_args()

    xtr_mv, ltr = realdata.load_beats(TRAIN_RECS, win=WIN, normalize=False)
    xte_mv, lte = realdata.load_beats(TEST_RECS, win=WIN, normalize=False)
    ytr = np.array([0 if l == "N" else 1 for l in ltr], np.int64)
    yte = np.array([0 if l == "N" else 1 for l in lte], np.int64)

    # 이진 검출기: clean 박동으로 학습
    torch.manual_seed(0)
    clf = ECGBeatClassifier(n_classes=2, win=WIN)
    cb = np.bincount(ytr).astype(float)
    w = torch.tensor(np.sqrt(cb.sum() / (2 * np.maximum(cb, 1))), dtype=torch.float32)
    dl = DataLoader(TensorDataset(torch.from_numpy(zscore(xtr_mv)).unsqueeze(1), torch.from_numpy(ytr)),
                    batch_size=256, shuffle=True)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss(weight=w)
    for _ in range(args.epochs):
        clf.train()
        for xb, yb in dl:
            opt.zero_grad(); lf(clf(xb), yb).backward(); opt.step()
    clf.eval()

    dn = DnCNN1D(depth=8, channels=32, kernel=9)
    dn.load_state_dict(torch.load(Path("outputs") / "dncnn1d_real.pt", map_location="cpu")); dn.eval()

    noise = realdata.load_noise(); nk = list(noise.keys())
    clean_auroc = auroc(abn_prob(clf, zscore(xte_mv))[yte == 1], abn_prob(clf, zscore(xte_mv))[yte == 0])

    snrs = [0.0, 3.0, 6.0, 9.0]
    res = {"noisy": [], "denoised": []}
    for snr in snrs:
        rng = np.random.default_rng(int(snr) + 1)
        noisy = np.stack([realdata.add_real_noise(xte_mv[i], noise[nk[i % len(nk)]], snr, rng)
                          for i in range(xte_mv.shape[0])])
        den = denoise(dn, noisy)
        for name, arr in [("noisy", noisy), ("denoised", den)]:
            p = abn_prob(clf, zscore(arr))
            res[name].append(auroc(p[yte == 1], p[yte == 0]))

    print("=== 디노이징이 이진 비정상 검출에 미치는 영향 (AUROC) ===")
    print(f"clean 상한: {clean_auroc:.3f}")
    print(f"{'SNR':>5s} {'noisy':>8s} {'denoised':>9s}  {'디노이징 효과':>12s}")
    for i, snr in enumerate(snrs):
        d = res["denoised"][i] - res["noisy"][i]
        print(f"{snr:4.0f} {res['noisy'][i]:8.3f} {res['denoised'][i]:9.3f}  {d:+11.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(snrs, res["noisy"], "o-", label="noisy (no denoise)")
    ax.plot(snrs, res["denoised"], "s-", label="denoised first")
    ax.axhline(clean_auroc, color="k", ls="--", lw=0.8, label=f"clean ({clean_auroc:.3f})")
    ax.set_xlabel("input SNR [dB]"); ax.set_ylabel("abnormal-detection AUROC")
    ax.set_title("Does denoising help BINARY abnormal detection?")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(Path("outputs") / "18_denoise_binary.png", dpi=130)
    print(f"\n[plot] outputs/18_denoise_binary.png")
    return res, clean_auroc


if __name__ == "__main__":
    main()
