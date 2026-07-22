"""실 노이즈 종류별 분해: 어떤 잡음이 고전 DSP / ML에 유리한가?

NSTDB 실 노이즈 3종을 따로 입혀 비교한다:
  - bw (baseline wander) : 저주파(~0.3Hz) → 신호대역과 분리됨
  - ma (muscle artifact) : 광대역 → QRS 대역과 겹침
  - em (electrode motion): 큰 과도 + 광대역 → 가장 험함

가설: 주파수로 분리되는 bw는 고전 고역통과가 잘 잡고, 신호와 겹치는 ma/em은
학습된 ML이 형태 사전지식으로 더 낫다.

    python scripts/17_noise_breakdown.py
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

from signal_ml_lab import classical, metrics, realdata  # noqa: E402
from signal_ml_lab.models import DnCNN1D  # noqa: E402
from signal_ml_lab.train import denoise  # noqa: E402

FS = 360
WIN = 1024
NOISE_NAMES = {"bw": "baseline wander", "ma": "muscle artifact", "em": "electrode motion"}


def clean_windows(n=300, records=("100", "103", "115", "215"), seed=5):
    rng = np.random.default_rng(seed)
    ecgs = realdata.load_ecg(records)
    C = np.empty((n, WIN), np.float32)
    for i in range(n):
        e = ecgs[int(rng.integers(len(ecgs)))]
        off = int(rng.integers(0, e.size - WIN))
        w = e[off:off + WIN]
        C[i] = (w - w.mean()).astype(np.float32)
    return C


def main():
    outdir = Path("outputs")
    dn = DnCNN1D(depth=8, channels=32, kernel=9)
    dn.load_state_dict(torch.load(outdir / "dncnn1d_real.pt", map_location="cpu")); dn.eval()

    clean = clean_windows()
    noise = realdata.load_noise()  # {bw, ma, em}
    snrs = [0.0, 4.0, 8.0]

    def out_snr(clean_arr, est):
        return float(np.mean([metrics.snr_db(c, e) for c, e in zip(clean_arr, est)]))

    rows = []
    for nk in ["bw", "ma", "em"]:
        for snr in snrs:
            rng = np.random.default_rng(int(snr) + 1)
            noisy = np.stack([realdata.add_real_noise(clean[i], noise[nk], snr, rng)
                              for i in range(clean.shape[0])])
            cls = np.stack([classical.bandpass_notch(x, fs=FS) for x in noisy])
            ml = denoise(dn, noisy)
            g_cls = out_snr(clean, cls) - snr
            g_ml = out_snr(clean, ml) - snr
            rows.append((nk, snr, g_cls, g_ml))

    print("=== 실 노이즈 종류별 SNR 이득 (dB) ===")
    print(f"{'noise':18s} {'inSNR':>6s} {'classical':>10s} {'ML':>8s} {'승자':>8s}")
    for nk, snr, gc, gm in rows:
        win = "ML" if gm > gc else "classical"
        print(f"{NOISE_NAMES[nk]:18s} {snr:5.0f}  {gc:+9.2f} {gm:+7.2f}   {win:>8s}")

    # 노이즈별 평균 이득
    print("\n노이즈별 평균 이득:")
    for nk in ["bw", "ma", "em"]:
        gs = [(gc, gm) for k, s, gc, gm in rows if k == nk]
        mc = np.mean([g[0] for g in gs]); mm = np.mean([g[1] for g in gs])
        print(f"  {NOISE_NAMES[nk]:18s}: classical {mc:+.2f} dB,  ML {mm:+.2f} dB  → "
              f"{'ML 우세' if mm > mc else '고전 우세'}")

    # 그림: 노이즈종류 × SNR 그룹 막대
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, nk in zip(axes, ["bw", "ma", "em"]):
        sub = [(s, gc, gm) for k, s, gc, gm in rows if k == nk]
        xs = np.arange(len(sub)); wbar = 0.38
        ax.bar(xs - wbar/2, [r[1] for r in sub], wbar, label="classical")
        ax.bar(xs + wbar/2, [r[2] for r in sub], wbar, label="ML")
        ax.set_xticks(xs); ax.set_xticklabels([f"{int(r[0])}dB" for r in sub])
        ax.set_title(NOISE_NAMES[nk]); ax.set_xlabel("input SNR"); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("SNR gain [dB]"); axes[0].legend()
    fig.suptitle("Denoising gain by real-noise type: classical DSP vs ML")
    fig.tight_layout(); fig.savefig(outdir / "17_noise_breakdown.png", dpi=130)
    print(f"\n[plot] {outdir / '17_noise_breakdown.png'}")
    return rows


if __name__ == "__main__":
    main()
