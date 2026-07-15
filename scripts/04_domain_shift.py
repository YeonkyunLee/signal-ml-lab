"""도메인 시프트 평가.

원분포(synth_ecg)로 학습한 ML 디노이저를, 형태가 다른 분포(synth_ecg_variant:
T파 역전·QRS 확대·이소성 박동)에서 테스트한다. 실데이터 일반화의 대리 실험.

    python scripts/04_domain_shift.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_ml_lab import classical, metrics, noise, synth  # noqa: E402
from signal_ml_lab.models import DnCNN1D  # noqa: E402
from signal_ml_lab.train import denoise  # noqa: E402

FS = 360
WIN = 1024


def make_variant_windows(n: int, noise_scale: float, seed: int):
    rng = np.random.default_rng(seed)
    gen_len = WIN + 2 * FS
    dur = gen_len / FS
    C = np.empty((n, WIN), np.float32)
    N = np.empty((n, WIN), np.float32)
    for i in range(n):
        hr = rng.uniform(55, 100)
        _, clean = synth.synth_ecg_variant(
            duration_s=dur, fs=FS, hr_bpm=hr, seed=int(rng.integers(1 << 30))
        )
        nz = noise.add_noise(
            clean,
            fs=FS,
            baseline_amp=0.30 * noise_scale,
            powerline_amp=0.10 * noise_scale,
            emg_amp=0.05 * noise_scale,
            seed=int(rng.integers(1 << 30)),
        )
        off = int(rng.integers(0, clean.size - WIN))
        C[i] = clean[off : off + WIN]
        N[i] = nz[off : off + WIN]
    return C, N


def evalset(clean, est):
    snr = float(np.mean([metrics.snr_db(c, e) for c, e in zip(clean, est)]))
    return snr


def main() -> None:
    outdir = Path("outputs")
    model = DnCNN1D(depth=8, channels=32, kernel=9)
    model.load_state_dict(torch.load(outdir / "dncnn1d.pt", map_location="cpu"))
    model.eval()

    rows = []
    example = None
    for ns in [0.5, 1.0, 1.5, 2.0]:
        clean, noisy = make_variant_windows(300, ns, seed=5000 + int(ns * 10))
        cls = np.stack([classical.bandpass_notch(x, fs=FS) for x in noisy])
        ml = denoise(model, noisy)
        snr_in = evalset(clean, noisy)
        rows.append(
            {
                "noise_scale": ns,
                "input_SNR_dB": round(snr_in, 2),
                "classical_SNR_dB": round(evalset(clean, cls), 2),
                "ML_SNR_dB": round(evalset(clean, ml), 2),
            }
        )
        if abs(ns - 1.0) < 1e-6:
            example = (clean[0], noisy[0], cls[0], ml[0])

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "04_domain_shift.csv", index=False)
    print("\n=== 도메인 시프트: 다른 형태 분포에서의 성능 ===")
    print("(모델은 원래 형태(synth_ecg)로만 학습됨)\n")
    print(df.to_string(index=False))

    if example is not None:
        clean0, noisy0, cls0, ml0 = example
        w = slice(0, 512)
        fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
        for ax, sig, lab, col in zip(
            axes,
            [noisy0, clean0, cls0, ml0],
            ["noisy (variant)", "clean (variant)", "classical", "ML (trained on original)"],
            ["C0", "green", "C1", "C1"],
        ):
            ax.plot(sig[w], lw=0.8, color=col)
            ax.set_ylabel(lab, fontsize=8)
        axes[-1].set_xlabel("sample")
        fig.suptitle("Domain shift: inverted-T / wide-QRS ECG, model trained on original")
        fig.tight_layout()
        fig.savefig(outdir / "04_domain_shift.png", dpi=130)
        print(f"\n[plot] {outdir / '04_domain_shift.png'}")


if __name__ == "__main__":
    main()
