"""디노이징 정량 평가 지표."""

from __future__ import annotations

import numpy as np


def rmse(clean: np.ndarray, est: np.ndarray) -> float:
    """제곱근평균제곱오차 (낮을수록 좋음)."""
    return float(np.sqrt(np.mean((clean - est) ** 2)))


def snr_db(clean: np.ndarray, est: np.ndarray) -> float:
    """신호 대 잡음비 [dB] (높을수록 좋음).

    잡음 = 추정신호 - 깨끗한신호.
    """
    signal_power = float(np.mean(clean**2))
    noise_power = float(np.mean((est - clean) ** 2))
    if noise_power == 0:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)
