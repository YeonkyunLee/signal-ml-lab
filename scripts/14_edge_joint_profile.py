"""엣지 배포 프로파일: 진단 파이프라인(디노이저+분류기)이 실시간으로 도는가?

06편은 디노이저 단독을 쟀다. 여기서는 joint(디노이즈→분류) 전체 파이프라인의 지연·
풋프린트·실시간 여유를 잰다. "정확할 뿐 아니라 배포 가능한가"에 답한다.

    python scripts/14_edge_joint_profile.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_ml_lab.models import DnCNN1D, ECGBeatClassifier, JointDenoiseClassify  # noqa: E402

FS = 360
WIN = 256


def footprint(model):
    n = sum(p.numel() for p in model.parameters())
    return n, n * 4 / 1024, n * 1 / 1024  # params, KB(fp32), KB(int8)


def bench(fn, x, iters=300, warmup=30):
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(x)
        return (time.perf_counter() - t0) / iters


def main() -> None:
    torch.set_num_threads(1)  # 보수적: 임베디드 단일 코어
    outdir = Path("outputs")

    dn = DnCNN1D(depth=8, channels=32, kernel=9)
    dn.load_state_dict(torch.load(outdir / "dncnn1d_real.pt", map_location="cpu")); dn.eval()
    clf = ECGBeatClassifier(n_classes=5, win=WIN)
    clf.load_state_dict(torch.load(outdir / "beat_classifier.pt", map_location="cpu")); clf.eval()
    joint = JointDenoiseClassify(DnCNN1D(depth=8, channels=32, kernel=9),
                                 ECGBeatClassifier(n_classes=5, win=WIN))
    joint.load_state_dict(torch.load(outdir / "joint_denoise_classify.pt", map_location="cpu")); joint.eval()

    x = torch.randn(1, 1, WIN)
    beat_sec = WIN / FS  # 한 박동 윈도우가 담는 신호 길이 [s]

    comps = [
        ("denoiser", dn, lambda m, x: m(x)),
        ("classifier", clf, lambda m, x: m(x)),
        ("joint (denoise+classify)", joint, lambda m, x: m(x)[0]),
    ]

    print(f"{'component':28s} {'params':>9s} {'KB(fp32/int8)':>16s} {'latency':>10s} {'beats/s':>9s} {'RTF':>8s}")
    rows = []
    for name, model, fn in comps:
        n, kb32, kb8 = footprint(model)
        lat = bench(lambda xx: fn(model, xx), x)
        rtf = lat / beat_sec
        print(f"{name:28s} {n:9,d} {kb32:7.0f}/{kb8:<7.0f} {lat*1e3:8.2f}ms {1/lat:9.0f} {rtf:8.4f}")
        rows.append((name, n, kb32, lat * 1e3, 1 / lat, rtf))

    # 실시간 여유 판정 (심박 ~1-3 Hz)
    lat_joint = rows[-1][3] / 1e3
    print("\n=== 판정 (전체 진단 파이프라인) ===")
    print(f"박동당 지연: {lat_joint*1e3:.2f} ms  (한 박동 신호 {beat_sec:.2f}s)")
    print(f"처리량: {1/lat_joint:.0f} beats/s — 심박 최대 ~3 beats/s 대비 {1/lat_joint/3:.0f}x 여유")
    print(f"실시간 계수(RTF): {lat_joint/beat_sec:.4f} → 단일 코어로 실시간 진단 가능")
    print("함의: 디노이즈+분류를 합쳐도 초당 수백 박동. int8·경량 MPU면 웨어러블 상시 진단 현실적.")

    # 그림: 지연 & 파라미터
    names = [r[0].split(" (")[0] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(names, [r[3] for r in rows], color=["C0", "C1", "C2"])
    ax1.set_ylabel("latency / beat [ms]"); ax1.set_title("Inference latency (1 core)")
    for i, r in enumerate(rows):
        ax1.text(i, r[3], f"{r[3]:.2f}ms", ha="center", va="bottom", fontsize=8)
    ax2.bar(names, [r[1] / 1000 for r in rows], color=["C0", "C1", "C2"])
    ax2.set_ylabel("parameters [k]"); ax2.set_title("Model size")
    for i, r in enumerate(rows):
        ax2.text(i, r[1] / 1000, f"{r[1]/1000:.0f}k", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Edge profile: full diagnostic pipeline (denoise + classify)")
    fig.tight_layout()
    fig.savefig(outdir / "14_edge_joint.png", dpi=130)
    print(f"\n[plot] {outdir / '14_edge_joint.png'}")


if __name__ == "__main__":
    main()
