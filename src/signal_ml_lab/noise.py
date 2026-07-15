"""현실적 ECG 노이즈 모델.

실제 ECG를 오염시키는 세 가지 대표 노이즈를 물리적 근거대로 합성한다:
- 기저선 변동(baseline wander): 호흡·전극 움직임에 의한 저주파(~0.3Hz)
- 전원선 간섭(powerline): 50/60Hz 정현파
- 근전도(EMG): 광대역 고주파 잡음
"""

from __future__ import annotations

import numpy as np


def add_noise(
    clean: np.ndarray,
    fs: int,
    baseline_amp: float = 0.30,
    powerline_hz: float = 60.0,
    powerline_amp: float = 0.10,
    emg_amp: float = 0.05,
    seed: int = 1,
) -> np.ndarray:
    """깨끗한 신호에 현실적 노이즈를 더한 관측 신호를 만든다."""
    rng = np.random.default_rng(seed)
    n = clean.size
    t = np.arange(n) / fs

    baseline = baseline_amp * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 2 * np.pi))
    powerline = powerline_amp * np.sin(2 * np.pi * powerline_hz * t)
    emg = emg_amp * rng.standard_normal(n)

    return clean + baseline + powerline + emg
