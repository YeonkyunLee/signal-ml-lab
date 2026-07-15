"""합성 ECG 생성.

각 심박을 여러 가우시안(P, Q, R, S, T 파형)의 합으로 근사한다.
다운로드 없이 재현 가능한 깨끗한 기준 신호(ground truth)를 제공하는 것이 목적.
"""

from __future__ import annotations

import numpy as np

# (중심 위상[rad], 진폭[mV], 폭) — 전형적 정상 동율동 형태를 근사
_WAVES = [
    (-1.10, 0.10, 0.25),  # P
    (-0.15, -0.15, 0.10),  # Q
    (0.00, 1.30, 0.10),  # R
    (0.15, -0.30, 0.10),  # S
    (1.20, 0.35, 0.40),  # T
]


def synth_ecg(
    duration_s: float = 10.0,
    fs: int = 360,
    hr_bpm: float = 60.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """깨끗한 합성 ECG를 만든다.

    Returns:
        t: 시간축 [s], shape (N,)
        clean: 깨끗한 ECG [mV], shape (N,)
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    # 심박마다 살짝 흔들리는 RR 간격(생리적 변동)
    rr = 60.0 / hr_bpm
    beat_times = []
    cur = 0.5
    while cur < duration_s:
        beat_times.append(cur)
        cur += rr * (1.0 + rng.normal(0, 0.03))

    clean = np.zeros(n)
    for bt in beat_times:
        # 한 심박 안의 위상: R을 위상 0으로 두고 ±pi를 심박 주기에 매핑
        phase = 2 * np.pi * (t - bt) / rr
        for center, amp, width in _WAVES:
            clean += amp * np.exp(-((phase - center) ** 2) / (2 * width**2))
    return t, clean


# 형태가 다른 "다른 환자군": T파 역전·QRS 폭 확대·P파 약화 등으로 분포를 이동시킨다.
# 도메인 시프트(학습분포 ≠ 테스트분포) 일반화를 실측 없이 검증하기 위한 변형.
_WAVES_VARIANT = [
    (-1.05, 0.04, 0.30),  # P (약함)
    (-0.20, -0.20, 0.14),  # Q (깊음)
    (0.00, 1.10, 0.16),  # R (넓음)
    (0.20, -0.40, 0.14),  # S (깊음)
    (1.25, -0.30, 0.45),  # T (역전)
]


def synth_ecg_variant(
    duration_s: float = 10.0,
    fs: int = 360,
    hr_bpm: float = 75.0,
    ectopic_prob: float = 0.08,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """형태가 다른 ECG(도메인 시프트용). 가끔 이소성 박동(ectopic)을 섞는다."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    rr = 60.0 / hr_bpm
    beat_times = []
    cur = 0.5
    while cur < duration_s:
        beat_times.append(cur)
        cur += rr * (1.0 + rng.normal(0, 0.05))

    clean = np.zeros(n)
    for bt in beat_times:
        phase = 2 * np.pi * (t - bt) / rr
        ectopic = rng.random() < ectopic_prob
        for center, amp, width in _WAVES_VARIANT:
            a = amp * (1.6 if ectopic else 1.0)  # 이소성 박동은 진폭 왜곡
            w = width * (1.4 if ectopic else 1.0)
            clean += a * np.exp(-((phase - center) ** 2) / (2 * w**2))
    return t, clean
