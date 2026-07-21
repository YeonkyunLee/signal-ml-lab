"""결합 문제 해결: 어떻게 하면 전처리가 진단을 돕는가?

12편에서 'SNR 디노이저 → clean 분류기'는 진단을 해쳤다. 여기서 세 가지 해법을 겨룬다:
  - naive        : noisy → clean-trained 분류기 (12편의 baseline)
  - broken       : SNR 디노이저 → clean-trained 분류기 (12편의 실패)
  - noise-aware  : noisy 박동으로 직접 학습한 분류기 (디노이저 없음)
  - joint (E2E)  : 디노이저+분류기를 분류손실로 함께 학습 (과제 인식)

    python scripts/13_joint_training.py --epochs 20
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_ml_lab import realdata  # noqa: E402
from signal_ml_lab.models import DnCNN1D, ECGBeatClassifier, JointDenoiseClassify  # noqa: E402
from signal_ml_lab.train import denoise  # noqa: E402

CLASSES = ["N", "S", "V", "F", "Q"]
TRAIN_RECS = ("101", "106", "108", "109", "112", "114", "116", "118", "119", "201", "203", "205", "208", "215", "220")
TEST_RECS = ("100", "103", "105", "111", "113", "121", "200", "202", "210", "213", "219", "221", "223", "230")
WIN = 256
V = CLASSES.index("V")


def zscore(b):
    return ((b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-6)).astype(np.float32)


def make_noisy(beats, noise, rng, snr_range=(0.0, 10.0)):
    nk = list(noise.keys())
    out = np.empty_like(beats)
    for i in range(beats.shape[0]):
        snr = float(rng.uniform(*snr_range))
        out[i] = realdata.add_real_noise(beats[i], noise[nk[i % len(nk)]], snr, rng)
    return out


def metrics(pred, true):
    acc = float((pred == true).mean())
    m = true == V
    vr = float((pred[m] == V).mean()) if m.any() else float("nan")
    # 균형 정확도(macro-recall): 불균형 하에서 공정한 지표
    recalls = []
    for c in range(len(CLASSES)):
        mc = true == c
        if mc.any():
            recalls.append((pred[mc] == c).mean())
    bal = float(np.mean(recalls))
    return acc, vr, bal


def train_classifier(x, y, weights, epochs, denoiser_frozen=None):
    torch.manual_seed(0)
    clf = ECGBeatClassifier(n_classes=len(CLASSES), win=WIN)
    dl = DataLoader(TensorDataset(torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y)),
                    batch_size=256, shuffle=True)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(weight=weights)
    for _ in range(epochs):
        clf.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = lossf(clf(xb), yb); loss.backward(); opt.step()
    return clf


def train_joint(xn, xc, y, weights, epochs, recon=0.3):
    torch.manual_seed(0)
    model = JointDenoiseClassify(DnCNN1D(depth=8, channels=32, kernel=9),
                                 ECGBeatClassifier(n_classes=len(CLASSES), win=WIN))
    dl = DataLoader(
        TensorDataset(torch.from_numpy(xn).unsqueeze(1), torch.from_numpy(xc).unsqueeze(1), torch.from_numpy(y)),
        batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss(weight=weights)
    mse = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, cb, yb in dl:
            opt.zero_grad()
            logits, den = model(xb)
            loss = ce(logits, yb) + recon * mse(den, cb)
            loss.backward(); opt.step()
    return model


@torch.no_grad()
def clf_pred(clf, bz):
    x = torch.from_numpy(bz).unsqueeze(1); out = []
    for i in range(0, len(x), 1024):
        out.append(clf(x[i:i+1024]).argmax(1).numpy())
    return np.concatenate(out)


@torch.no_grad()
def joint_pred(model, bn):
    x = torch.from_numpy(bn).unsqueeze(1); out = []
    for i in range(0, len(x), 1024):
        out.append(model(x[i:i+1024])[0].argmax(1).numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    noise = realdata.load_noise()
    print("[data] train/test 박동 로드...")
    tr_c, tr_l = realdata.load_beats(TRAIN_RECS, win=WIN, normalize=False)
    te_c, te_l = realdata.load_beats(TEST_RECS, win=WIN, normalize=False)
    ytr = np.array([CLASSES.index(x) for x in tr_l])
    yte = np.array([CLASSES.index(x) for x in te_l])
    print(f"[data] train={len(ytr)}  test={len(yte)}  V(test)={int((yte==V).sum())}")

    counts = np.bincount(ytr, minlength=len(CLASSES)).astype(float)
    # 완화된 가중치(sqrt 역빈도)로 소수클래스 과편향 방지, 상한 캡
    w = np.sqrt(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)))
    w = np.clip(w, 0.5, 5.0)
    weights = torch.tensor(w, dtype=torch.float32)

    # 노이즈 낀 학습셋
    rng = np.random.default_rng(0)
    tr_n = make_noisy(tr_c, noise, rng)

    # 기존 자산
    outdir = Path("outputs")
    clf_clean = ECGBeatClassifier(n_classes=len(CLASSES), win=WIN)
    clf_clean.load_state_dict(torch.load(outdir / "beat_classifier.pt", map_location="cpu")); clf_clean.eval()
    dn = DnCNN1D(depth=8, channels=32, kernel=9)
    dn.load_state_dict(torch.load(outdir / "dncnn1d_real.pt", map_location="cpu")); dn.eval()

    # 해법 학습
    print("[train] noise-aware 분류기...")
    clf_na = train_classifier(tr_n, ytr, weights, args.epochs); clf_na.eval()
    print("[train] joint (end-to-end)...")
    joint = train_joint(tr_n, tr_c, ytr, weights, args.epochs); joint.eval()

    # 모델 저장(재평가 재사용)
    torch.save(clf_na.state_dict(), outdir / "clf_noise_aware.pt")
    torch.save(joint.state_dict(), outdir / "joint_denoise_classify.pt")

    # 테스트: 여러 SNR
    snrs = [0.0, 3.0, 6.0, 9.0]
    res = {k: {"bal": [], "vr": []} for k in ["naive", "broken", "noise-aware", "joint"]}
    for snr in snrs:
        rng = np.random.default_rng(int(snr) + 100)
        te_n = make_noisy(te_c, noise, rng, snr_range=(snr, snr))
        a, v, b = metrics(clf_pred(clf_clean, zscore(te_n)), yte); res["naive"]["bal"].append(b); res["naive"]["vr"].append(v)
        a, v, b = metrics(clf_pred(clf_clean, zscore(denoise(dn, te_n))), yte); res["broken"]["bal"].append(b); res["broken"]["vr"].append(v)
        a, v, b = metrics(clf_pred(clf_na, zscore(te_n)), yte); res["noise-aware"]["bal"].append(b); res["noise-aware"]["vr"].append(v)
        a, v, b = metrics(joint_pred(joint, te_n), yte); res["joint"]["bal"].append(b); res["joint"]["vr"].append(v)
        print(f"SNR={snr:4.1f} | " + "  ".join(f"{k}:bal={res[k]['bal'][-1]*100:4.1f}/PVC={res[k]['vr'][-1]*100:4.1f}" for k in res))

    # clean 상한
    ac, vc, bc = metrics(clf_pred(clf_clean, zscore(te_c)), yte)
    print(f"\nclean 상한: bal-acc={bc*100:.1f}% PVC={vc*100:.1f}%")

    import pandas as pd
    df = pd.DataFrame({"snr": snrs, **{f"{k}_bal": res[k]["bal"] for k in res}, **{f"{k}_vr": res[k]["vr"] for k in res}})
    df.to_csv(outdir / "13_joint_training.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for k, mk in zip(res, ["o", "x", "s", "D"]):
        ax1.plot(snrs, np.array(res[k]["bal"]) * 100, mk + "-", label=k)
        ax2.plot(snrs, np.array(res[k]["vr"]) * 100, mk + "-", label=k)
    ax1.axhline(bc * 100, color="k", ls="--", lw=0.8, label="clean")
    ax2.axhline(vc * 100, color="k", ls="--", lw=0.8, label="clean")
    ax1.set_title("Balanced accuracy (macro-recall)"); ax2.set_title("PVC (V) recall")
    for ax in (ax1, ax2):
        ax.set_xlabel("input SNR [dB]"); ax.set_ylabel("%"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Fixing the coupling: task-aware training beats naive denoise->classify")
    fig.tight_layout()
    fig.savefig(outdir / "13_joint_training.png", dpi=130)
    print(f"\n[plot] {outdir / '13_joint_training.png'}")


if __name__ == "__main__":
    main()
