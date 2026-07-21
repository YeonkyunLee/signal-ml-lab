"""합성 사전학습 → 실데이터 미세조정: 데이터 효율 곡선.

질문: 실데이터가 적을 때, 합성으로 사전학습하면 처음부터 실학습하는 것보다 얼마나
유리한가? 실 학습샘플 수 N을 바꿔가며 두 곡선을 비교한다.
  - scratch      : N개 실데이터로 처음부터 학습
  - pretrained   : 합성으로 사전학습 후 N개 실데이터로 미세조정

    python scripts/09_finetune_curve.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_ml_lab import dataset, metrics, realdata  # noqa: E402
from signal_ml_lab.models import DnCNN1D  # noqa: E402
from signal_ml_lab.train import TrainConfig, denoise, train_model  # noqa: E402

TRAIN_RECS = ("101", "106", "119", "208", "230")
TEST_RECS = ("100", "103", "115", "215")


def snr(clean, est):
    return float(np.mean([metrics.snr_db(c, e) for c, e in zip(clean, est)]))


def main() -> None:
    # 고정 실 테스트셋
    te_c, te_n = realdata.make_real_pairs(400, records=TEST_RECS, seed=7)
    snr_in = snr(te_c, te_n)
    print(f"[data] 실 테스트 입력 SNR = {snr_in:.2f} dB")

    # 큰 실 학습 풀
    pool_c, pool_n = realdata.make_real_pairs(8000, records=TRAIN_RECS, seed=0)

    # 합성 사전학습 모델 (기존 체크포인트 재사용; 없으면 학습)
    outdir = Path("outputs")
    pre = DnCNN1D(depth=8, channels=32, kernel=9)
    ck = outdir / "dncnn1d.pt"
    if ck.exists():
        pre.load_state_dict(torch.load(ck, map_location="cpu"))
    else:
        sc, sn = dataset.make_windows(4000, seed=0)
        train_model(pre, dataset.split(sc, sn), TrainConfig(epochs=25))
    snr_syn = snr(te_c, denoise(pre, te_n))
    print(f"[baseline] 합성 전용 ML = {snr_syn:.2f} dB, 고전은 별도(README 6.70)")

    Ns = [100, 300, 1000, 3000, 8000]
    res = {"scratch": [], "pretrained": []}
    for N in Ns:
        tr = dataset.split(pool_c[:N], pool_n[:N], val_frac=0.15, test_frac=0.0)

        m_s = DnCNN1D(depth=8, channels=32, kernel=9)
        train_model(m_s, tr, TrainConfig(epochs=25))
        s_scratch = snr(te_c, denoise(m_s, te_n))

        m_p = copy.deepcopy(pre)  # 합성 사전학습에서 출발
        train_model(m_p, tr, TrainConfig(epochs=15, lr=5e-4))  # 미세조정: 낮은 lr
        s_pre = snr(te_c, denoise(m_p, te_n))

        res["scratch"].append(s_scratch)
        res["pretrained"].append(s_pre)
        print(f"N={N:5d}  scratch={s_scratch:6.2f}  pretrained={s_pre:6.2f} dB")

    outdir.mkdir(exist_ok=True)
    import pandas as pd

    df = pd.DataFrame({"N_real": Ns, **res})
    df.to_csv(outdir / "09_finetune_curve.csv", index=False)
    print("\n" + df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Ns, res["scratch"], "o-", label="from scratch")
    ax.plot(Ns, res["pretrained"], "s-", label="synthetic pretrain + finetune")
    ax.axhline(snr_syn, color="gray", ls=":", label=f"synthetic-only ({snr_syn:.1f} dB)")
    ax.axhline(6.70, color="k", ls="--", lw=0.8, label="classical (6.70 dB)")
    ax.set_xscale("log")
    ax.set_xlabel("# real training windows")
    ax.set_ylabel("real test SNR [dB]")
    ax.set_title("Sim-to-real: data efficiency of synthetic pretraining")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "09_finetune_curve.png", dpi=130)
    print(f"\n[plot] {outdir / '09_finetune_curve.png'}")


if __name__ == "__main__":
    main()
