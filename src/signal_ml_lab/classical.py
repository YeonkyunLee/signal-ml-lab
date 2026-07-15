"""고전 DSP 디노이징.

기저선 변동은 고역통과로, 전원선은 노치로, 고주파 EMG는 저역통과로 제거한다.
웨이블릿 임계처리는 PyWavelets가 있으면 추가로 제공한다(선택).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_notch(
    x: np.ndarray,
    fs: int,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    notch_hz: float = 60.0,
    notch_q: float = 30.0,
) -> np.ndarray:
    """대역통과(기저선+고주파 제거) 후 노치(전원선 제거).

    filtfilt로 영위상(zero-phase) 처리 → ECG의 파형 위치를 왜곡하지 않는다.
    """
    nyq = fs / 2.0
    b, a = butter(4, [low_hz / nyq, high_hz / nyq], btype="band")
    y = filtfilt(b, a, x)

    if notch_hz < nyq:
        bn, an = iirnotch(notch_hz / nyq, notch_q)
        y = filtfilt(bn, an, y)
    return y


def wavelet_denoise(x: np.ndarray, wavelet: str = "db6", level: int = 4) -> np.ndarray:
    """웨이블릿 소프트 임계처리(범용 임계값). PyWavelets 필요."""
    try:
        import pywt
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("PyWavelets 미설치: pip install PyWavelets") from e

    coeffs = pywt.wavedec(x, wavelet, level=level)
    # 최고주파 대역으로 노이즈 표준편차 추정(MAD)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(x.size))
    coeffs[1:] = [pywt.threshold(c, uthresh, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[: x.size]
