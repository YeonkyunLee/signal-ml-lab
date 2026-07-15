"""고전 DSP vs ML 디노이저 벤치마크 (SNR 스윕).

같은 합성 테스트셋을 여러 노이즈 세기로 만들어, 입력 SNR 대비 각 방법의 출력
SNR/RMSE를 표와 그림으로 비교한다.

    python scripts/03_benchmark.py
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

from signal_ml_lab import classical, dataset, metrics  # noqa: E402
from signal_ml_lab.models import DnCNN1D  # noqa: E402
from signal_ml_lab.train import denoise  # noqa: E402

FS = 360


def _has_pywt() -> bool:
    try:
        import pywt  # noqa: F401

        return True
    except ImportError:
        return False


def eval_method(clean: np.ndarray, est: np.ndarray) -> tuple[float, float]:
    snr = float(np.mean([metrics.snr_db(c, e) for c, e in zip(clean, est)]))
    rmse = float(np.mean([metrics.rmse(c, e) for c, e in zip(clean, est)]))
    return snr, rmse


def main() -> None:
    outdir = Path("outputs")
    ckpt = outdir / "dncnn1d.pt"
    if not ckpt.exists():
        raise SystemExit("모델 없음. 먼저 scripts/02_train_ml_denoise.py 실행")

    model = DnCNN1D(depth=8, channels=32, kernel=9)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    use_wavelet = _has_pywt()
    noise_scales = [0.5, 1.0, 1.5, 2.0, 3.0]
    rows = []
    example = None  # 정성 비교용 한 윈도우 저장(scale=1.5)

    for ns in noise_scales:
        # 각 노이즈 세기에서 독립 테스트셋 생성(학습과 다른 시드)
        clean, noisy = dataset.make_windows(
            300, win_len=1024, noise_scale_range=(ns, ns), seed=1000 + int(ns * 10)
        )
        results = {"noisy": noisy}
        results["classical (bp+notch)"] = np.stack(
            [classical.bandpass_notch(x, fs=FS) for x in noisy]
        )
        if use_wavelet:
            bp = results["classical (bp+notch)"]
            results["classical (+wavelet)"] = np.stack(
                [classical.wavelet_denoise(x) for x in bp]
            )
        results["ML (DnCNN1D)"] = denoise(model, noisy)

        snr_in, _ = eval_method(clean, noisy)
        for name, est in results.items():
            if name == "noisy":
                continue
            snr, rmse = eval_method(clean, est)
            rows.append(
                {
                    "noise_scale": ns,
                    "input_SNR_dB": round(snr_in, 2),
                    "method": name,
                    "output_SNR_dB": round(snr, 2),
                    "gain_dB": round(snr - snr_in, 2),
                    "RMSE": round(rmse, 4),
                }
            )
        if abs(ns - 1.5) < 1e-6:
            example = (clean[0], results)

    df = pd.DataFrame(rows)
    outdir.mkdir(exist_ok=True)
    csv = outdir / "03_benchmark.csv"
    df.to_csv(csv, index=False)

    print("\n=== 벤치마크: 고전 DSP vs ML (SNR 스윕) ===\n")
    print(df.to_string(index=False))
    print(f"\n[csv] {csv}")

    # (1) 방법별 출력 SNR vs 입력 SNR
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in df["method"].unique():
        sub = df[df["method"] == name].sort_values("input_SNR_dB")
        ax.plot(sub["input_SNR_dB"], sub["output_SNR_dB"], marker="o", label=name)
    lo = df["input_SNR_dB"].min()
    hi = df["output_SNR_dB"].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="no-op (y=x)")
    ax.set_xlabel("input SNR [dB]")
    ax.set_ylabel("output SNR [dB]")
    ax.set_title("ECG denoising: classical DSP vs ML")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "03_benchmark_snr.png", dpi=130)
    print(f"[plot] {outdir / '03_benchmark_snr.png'}")

    # (2) 정성 비교: 한 윈도우에서 모든 방법
    if example is not None:
        clean0, results = example
        methods = [k for k in results if k != "noisy"]
        fig, axes = plt.subplots(len(methods) + 2, 1, figsize=(10, 2 + 1.6 * len(methods)), sharex=True)
        win = slice(0, 512)
        axes[0].plot(results["noisy"][0][win], lw=0.8)
        axes[0].set_ylabel("noisy")
        axes[1].plot(clean0[win], lw=0.8, color="green")
        axes[1].set_ylabel("clean")
        for i, name in enumerate(methods, start=2):
            axes[i].plot(results[name][0][win], lw=0.8, color="C1")
            axes[i].set_ylabel(name, fontsize=8)
        axes[-1].set_xlabel("sample")
        fig.suptitle("Qualitative comparison (input SNR ~ noise_scale 1.5)")
        fig.tight_layout()
        fig.savefig(outdir / "03_qualitative.png", dpi=130)
        print(f"[plot] {outdir / '03_qualitative.png'}")


if __name__ == "__main__":
    main()
