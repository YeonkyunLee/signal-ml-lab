"""엣지 배포 프로파일: 이 디노이저가 실시간으로 돌 수 있는가?

지연시간(윈도우당), 처리량, 실시간 계수(RTF), 모델 메모리 풋프린트를 잰다.
임베디드/엣지 실시간 추론 타당성을 판단하기 위한 분석.

    python scripts/06_edge_profile.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_ml_lab.models import DnCNN1D  # noqa: E402

FS = 360
WIN = 1024


def count_macs(model: DnCNN1D, length: int) -> int:
    """1D conv의 대략적 MAC 수(길이는 same-padding으로 보존 가정)."""
    macs = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d):
            cin = m.in_channels
            cout = m.out_channels
            k = m.kernel_size[0]
            macs += cin * cout * k * length
    return macs


def bench_latency(model, x, iters=200, warmup=20):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        # perf_counter는 벽시계 — Date.now 제한과 무관(런타임 허용)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        t1 = time.perf_counter()
    return (t1 - t0) / iters


def main() -> None:
    torch.set_num_threads(1)  # 임베디드 단일코어 가정(보수적)
    model = DnCNN1D(depth=8, channels=32, kernel=9)
    ckpt = Path("outputs/dncnn1d.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    n_params = sum(p.numel() for p in model.parameters())
    size_kb = n_params * 4 / 1024  # float32
    size_int8_kb = n_params * 1 / 1024
    macs = count_macs(model, WIN)

    print("=== 모델 풋프린트 ===")
    print(f"파라미터        : {n_params:,}")
    print(f"메모리(float32) : {size_kb:8.1f} KB")
    print(f"메모리(int8 가정): {size_int8_kb:8.1f} KB")
    print(f"MAC/윈도우      : {macs/1e6:8.2f} M  ({macs/WIN/1e3:.1f} kMAC/sample)")

    print("\n=== 지연시간 (단일 스레드, batch=1) ===")
    for win in [512, 1024, 2048]:
        x = torch.randn(1, 1, win)
        lat = bench_latency(model, x)
        sig_sec = win / FS  # 이 윈도우가 담는 신호 길이[s]
        rtf = lat / sig_sec  # 실시간 계수(<1 이면 실시간 가능)
        print(
            f"win={win:5d} ({sig_sec:4.2f}s 신호)  "
            f"지연={lat*1e3:6.2f} ms  처리량={1/lat:7.1f} win/s  RTF={rtf:.4f}"
        )

    x = torch.randn(1, 1, WIN)
    lat = bench_latency(model, x)
    rtf = lat / (WIN / FS)
    print("\n=== 판정 ===")
    print(f"RTF={rtf:.4f} → 1초 신호를 {rtf*1000:.1f} ms에 처리")
    print(f"이 CPU 단일코어 여유배수: {1/rtf:.0f}x 실시간")
    print(
        "함의: 파라미터 5.6만·수십 MMAC 규모라 int8 양자화 시 "
        "Cortex-M/저가 MPU급에서도 실시간 목표가 현실적."
    )


if __name__ == "__main__":
    main()
