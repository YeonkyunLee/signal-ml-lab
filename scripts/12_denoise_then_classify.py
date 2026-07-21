"""디노이징 → 분류 결합: 전처리가 진단을 돕는가?

노이즈가 낀 심박은 분류 정확도를 떨어뜨린다. 실학습 디노이저로 전처리하면 그 손실을
얼마나 되사는가? 세 조건을 같은 분류기(clean 학습)로 비교한다:
  - clean     : 깨끗한 박동 (상한선)
  - noisy     : 실 노이즈 낀 박동 (열화)
  - denoised  : 디노이저 통과 후 박동 (회복)

    python scripts/12_denoise_then_classify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_ml_lab import realdata  # noqa: E402
from signal_ml_lab.models import DnCNN1D, ECGBeatClassifier  # noqa: E402
from signal_ml_lab.train import denoise  # noqa: E402

CLASSES = ["N", "S", "V", "F", "Q"]
TEST_RECS = ("100", "103", "105", "111", "113", "121", "200", "202", "210", "213", "219", "221", "223", "230")
WIN = 256
FS = 360


def zscore(beats: np.ndarray) -> np.ndarray:
    m = beats.mean(axis=1, keepdims=True)
    s = beats.std(axis=1, keepdims=True) + 1e-6
    return ((beats - m) / s).astype(np.float32)


@torch.no_grad()
def classify(clf, beats_z: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(beats_z).unsqueeze(1)
    out = []
    for i in range(0, x.size(0), 1024):
        out.append(clf(x[i : i + 1024]).argmax(1).numpy())
    return np.concatenate(out)


def metrics_of(pred, true):
    acc = float((pred == true).mean())
    v = CLASSES.index("V")
    v_mask = true == v
    v_recall = float((pred[v_mask] == v).mean()) if v_mask.any() else float("nan")
    return acc, v_recall


def main() -> None:
    outdir = Path("outputs")
    clf = ECGBeatClassifier(n_classes=len(CLASSES), win=WIN)
    clf.load_state_dict(torch.load(outdir / "beat_classifier.pt", map_location="cpu"))
    clf.eval()
    dn = DnCNN1D(depth=8, channels=32, kernel=9)
    dn.load_state_dict(torch.load(outdir / "dncnn1d_real.pt", map_location="cpu"))
    dn.eval()

    print("[data] 테스트 박동 로드 (mV 스케일, 캐시)...")
    beats_mv, labels = realdata.load_beats(TEST_RECS, win=WIN, normalize=False)
    true = np.array([CLASSES.index(l) for l in labels])
    noise = realdata.load_noise()
    nk = list(noise.keys())
    print(f"[data] {len(true)} beats, V(PVC) {int((true==CLASSES.index('V')).sum())}개")

    # clean 상한선
    acc_c, vr_c = metrics_of(classify(clf, zscore(beats_mv)), true)

    snrs = [0.0, 3.0, 6.0, 9.0]
    rows = []
    for snr in snrs:
        rng = np.random.default_rng(int(snr) + 1)
        noisy = np.stack([
            realdata.add_real_noise(beats_mv[i], noise[nk[i % len(nk)]], snr, rng)
            for i in range(beats_mv.shape[0])
        ])
        den = denoise(dn, noisy)
        acc_n, vr_n = metrics_of(classify(clf, zscore(noisy)), true)
        acc_d, vr_d = metrics_of(classify(clf, zscore(den)), true)
        rows.append((snr, acc_n, acc_d, vr_n, vr_d))
        print(
            f"SNR={snr:4.1f}dB | acc noisy={acc_n*100:5.1f}% denoised={acc_d*100:5.1f}%"
            f" | PVC recall noisy={vr_n*100:5.1f}% denoised={vr_d*100:5.1f}%"
        )

    print(f"\nclean 상한선: acc={acc_c*100:.1f}%  PVC recall={vr_c*100:.1f}%")

    import pandas as pd

    df = pd.DataFrame(rows, columns=["snr", "acc_noisy", "acc_denoised", "vr_noisy", "vr_denoised"])
    df.to_csv(outdir / "12_denoise_then_classify.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(df.snr, df.acc_noisy * 100, "o-", label="noisy")
    ax1.plot(df.snr, df.acc_denoised * 100, "s-", label="denoised")
    ax1.axhline(acc_c * 100, color="k", ls="--", lw=0.8, label="clean (upper bound)")
    ax1.set_xlabel("input SNR [dB]"); ax1.set_ylabel("accuracy [%]"); ax1.set_title("Overall accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(df.snr, df.vr_noisy * 100, "o-", label="noisy")
    ax2.plot(df.snr, df.vr_denoised * 100, "s-", label="denoised")
    ax2.axhline(vr_c * 100, color="k", ls="--", lw=0.8, label="clean")
    ax2.set_xlabel("input SNR [dB]"); ax2.set_ylabel("PVC recall [%]"); ax2.set_title("PVC (V) recall")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("Denoising as preprocessing for beat classification")
    fig.tight_layout()
    fig.savefig(outdir / "12_denoise_then_classify.png", dpi=130)
    print(f"\n[plot] {outdir / '12_denoise_then_classify.png'}")

    # 왜 나빠지는가 — PVC 형태 뭉개짐 (노이즈 없이 디노이저만 통과)
    v = CLASSES.index("V")
    v_idx = np.where(true == v)[0][:4]
    clean_v = beats_mv[v_idx]
    den_v = denoise(dn, clean_v)
    fig2, axes = plt.subplots(1, 4, figsize=(13, 3), sharey=True)
    for ax, cv, dv in zip(axes, clean_v, den_v):
        ax.plot(cv, "g-", lw=1.2, label="clean PVC")
        ax.plot(dv, "C1-", lw=1.2, label="denoised")
    axes[0].legend(fontsize=8); axes[0].set_ylabel("amplitude")
    fig2.suptitle("SNR-optimized denoiser flattens abnormal PVC morphology (no noise added)")
    fig2.tight_layout()
    fig2.savefig(outdir / "12_pvc_morphology.png", dpi=130)
    print(f"[plot] {outdir / '12_pvc_morphology.png'}")


if __name__ == "__main__":
    main()
