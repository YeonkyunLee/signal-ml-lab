"""엔드투엔드 데모: 합성(또는 실제) ECG를 오염시키고 고전 DSP로 디노이징.

    python scripts/01_classical_denoise.py                 # 합성 신호
    python scripts/01_classical_denoise.py --record 100    # 실제 PhysioNet(wfdb)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로 저장
import matplotlib.pyplot as plt
import numpy as np

# src 레이아웃을 설치 없이 import 가능하게
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_ml_lab import classical, metrics, noise, synth  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=None, help="PhysioNet 레코드 ID (예: 100). 없으면 합성.")
    ap.add_argument("--wavelet", action="store_true", help="웨이블릿 디노이징도 실행")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    if args.record:
        from signal_ml_lab import data

        t, observed, fs = data.load_physionet(record=args.record)
        clean = None  # 실제 데이터엔 ground truth가 없음
        print(f"[data] PhysioNet {args.record}, fs={fs}Hz, N={observed.size}")
    else:
        fs = 360
        t, clean = synth.synth_ecg(duration_s=10.0, fs=fs, hr_bpm=60)
        observed = noise.add_noise(clean, fs=fs)
        print(f"[data] 합성 ECG, fs={fs}Hz, N={observed.size}")

    denoised = classical.bandpass_notch(observed, fs=fs)

    results = {"bandpass+notch": denoised}
    if args.wavelet:
        try:
            results["wavelet"] = classical.wavelet_denoise(denoised)
        except RuntimeError as e:
            print(f"[warn] {e}")

    # ground truth가 있으면 정량 평가
    if clean is not None:
        print("\n=== 정량 평가 (ground truth 대비) ===")
        base_snr = metrics.snr_db(clean, observed)
        base_rmse = metrics.rmse(clean, observed)
        print(f"{'noisy(입력)':22s}  SNR={base_snr:6.2f} dB  RMSE={base_rmse:.4f}")
        for name, est in results.items():
            print(
                f"{name:22s}  SNR={metrics.snr_db(clean, est):6.2f} dB  "
                f"RMSE={metrics.rmse(clean, est):.4f}"
            )

    # 그림 저장
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    win = slice(0, min(int(4 * fs), observed.size))  # 앞 4초만
    fig, axes = plt.subplots(2 + len(results), 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t[win], observed[win], lw=0.8)
    axes[0].set_ylabel("noisy")
    idx = 1
    if clean is not None:
        axes[idx].plot(t[win], clean[win], lw=0.8, color="green")
        axes[idx].set_ylabel("clean")
        idx += 1
    for name, est in results.items():
        axes[idx].plot(t[win], est[win], lw=0.8, color="C1")
        axes[idx].set_ylabel(name)
        idx += 1
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("ECG denoising - classical DSP baseline")
    fig.tight_layout()
    out = outdir / "01_classical_denoise.png"
    fig.savefig(out, dpi=130)
    print(f"\n[plot] 저장: {out}")


if __name__ == "__main__":
    main()
