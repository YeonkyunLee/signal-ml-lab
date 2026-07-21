"""선택적 진단 (selective classification): 확신 없으면 사람에게 넘긴다.

배포된 진단 시스템은 모든 박동을 자동 판정할 필요가 없다. 불확실한 박동은 보류하고
심장전문의에게 넘기면, 자동 판정하는 나머지의 정확도는 올라간다. 이 트레이드오프를
risk-coverage 곡선으로 정량화한다. (3편 불확실도 게이트를 진단에 적용.)

    python scripts/15_selective_diagnosis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_ml_lab import realdata  # noqa: E402
from signal_ml_lab.models import ECGBeatClassifier  # noqa: E402

CLASSES = ["N", "S", "V", "F", "Q"]
TEST_RECS = ("100", "103", "105", "111", "113", "121", "200", "202", "210", "213", "219", "221", "223", "230")
WIN = 256


def zscore(b):
    return ((b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-6)).astype(np.float32)


@torch.no_grad()
def mc_predict(clf, beats_z, T=30):
    """MC-dropout: dropout 켠 채 T회 → 평균 확률분포로 예측·신뢰도."""
    clf.train()  # dropout 활성
    x = torch.from_numpy(beats_z).unsqueeze(1)
    probs = torch.zeros(x.size(0), len(CLASSES))
    for _ in range(T):
        out = []
        for i in range(0, x.size(0), 2048):
            out.append(F.softmax(clf(x[i:i+2048]), dim=1))
        probs += torch.cat(out)
    probs /= T
    conf, pred = probs.max(1)
    return pred.numpy(), conf.numpy()


def risk_coverage(pred, true, conf):
    """신뢰도 내림차순으로 커버리지를 늘리며 자동판정분 정확도 계산."""
    order = np.argsort(-conf)
    p, t = pred[order], true[order]
    correct = (p == t).astype(float)
    cov = np.arange(1, len(t) + 1) / len(t)
    acc = np.cumsum(correct) / np.arange(1, len(t) + 1)
    return cov, acc


def main() -> None:
    outdir = Path("outputs")
    clf = ECGBeatClassifier(n_classes=len(CLASSES), win=WIN)
    clf.load_state_dict(torch.load(outdir / "beat_classifier.pt", map_location="cpu"))

    beats_mv, labels = realdata.load_beats(TEST_RECS, win=WIN, normalize=False)
    true = np.array([CLASSES.index(l) for l in labels])
    noise = realdata.load_noise()

    conditions = {}
    # clean
    pred, conf = mc_predict(clf, zscore(beats_mv))
    conditions["clean"] = risk_coverage(pred, true, conf)
    full_acc_clean = (pred == true).mean()
    # noisy 6 dB
    rng = np.random.default_rng(6)
    nk = list(noise.keys())
    noisy = np.stack([realdata.add_real_noise(beats_mv[i], noise[nk[i % len(nk)]], 6.0, rng)
                      for i in range(beats_mv.shape[0])])
    pred_n, conf_n = mc_predict(clf, zscore(noisy))
    conditions["noisy (6 dB)"] = risk_coverage(pred_n, true, conf_n)
    full_acc_noisy = (pred_n == true).mean()

    print("=== 선택적 진단: 커버리지별 자동판정 정확도 ===")
    for name, (cov, acc) in conditions.items():
        def at(c):
            idx = np.searchsorted(cov, c)
            return acc[min(idx, len(acc) - 1)]
        print(f"\n[{name}]  전체(100%) 정확도 = {acc[-1]*100:.1f}%")
        for c in [0.5, 0.7, 0.8, 0.9]:
            print(f"  커버리지 {int(c*100)}% → 자동판정 정확도 {at(c)*100:.1f}%  (보류 {int((1-c)*100)}%)")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (cov, acc) in conditions.items():
        ax.plot(cov * 100, acc * 100, label=name)
    ax.axhline(full_acc_clean * 100, color="C0", ls=":", lw=0.8)
    ax.axhline(full_acc_noisy * 100, color="C1", ls=":", lw=0.8)
    ax.set_xlabel("coverage [%] (auto-decided beats)")
    ax.set_ylabel("accuracy on auto-decided [%]")
    ax.set_title("Selective diagnosis: defer uncertain beats to a cardiologist")
    ax.legend(); ax.grid(alpha=0.3)
    ax.invert_xaxis()  # 왼쪽=보류 많음(고신뢰만), 오른쪽=전량 자동
    fig.tight_layout()
    fig.savefig(outdir / "15_selective_diagnosis.png", dpi=130)
    print(f"\n[plot] {outdir / '15_selective_diagnosis.png'}")


if __name__ == "__main__":
    main()
