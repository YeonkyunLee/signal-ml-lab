"""학습용 (clean, noisy) ECG 윈도우 쌍 생성.

합성 ECG를 심박수·시드·노이즈 세기를 바꿔가며 여러 개 만들고, 고정 길이
윈도우로 잘라 (깨끗한 신호, 오염된 신호) 쌍을 반환한다. ground truth가 있으니
지도학습 디노이징이 가능하다.
"""

from __future__ import annotations

import numpy as np

from .noise import add_noise
from .synth import synth_ecg


def make_windows(
    n_windows: int,
    win_len: int = 1024,
    fs: int = 360,
    hr_range: tuple[float, float] = (50.0, 95.0),
    noise_scale_range: tuple[float, float] = (0.5, 1.5),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """(clean, noisy) 쌍을 생성한다.

    Returns:
        clean: (n_windows, win_len) float32
        noisy: (n_windows, win_len) float32
    """
    rng = np.random.default_rng(seed)
    clean_list = np.empty((n_windows, win_len), dtype=np.float32)
    noisy_list = np.empty((n_windows, win_len), dtype=np.float32)

    # 윈도우 하나를 채우려면 여유 길이가 필요 → 넉넉히 생성 후 랜덤 오프셋 절취
    gen_len = win_len + 2 * fs
    dur = gen_len / fs

    for i in range(n_windows):
        hr = rng.uniform(*hr_range)
        ns = rng.uniform(*noise_scale_range)
        # 각 윈도우가 서로 다른 파형/노이즈를 갖도록 시드 분리
        _, clean = synth_ecg(duration_s=dur, fs=fs, hr_bpm=hr, seed=int(rng.integers(1 << 30)))
        noisy = add_noise(
            clean,
            fs=fs,
            baseline_amp=0.30 * ns,
            powerline_amp=0.10 * ns,
            emg_amp=0.05 * ns,
            seed=int(rng.integers(1 << 30)),
        )
        off = int(rng.integers(0, clean.size - win_len))
        clean_list[i] = clean[off : off + win_len]
        noisy_list[i] = noisy[off : off + win_len]

    return clean_list, noisy_list


def split(
    clean: np.ndarray, noisy: np.ndarray, val_frac: float = 0.15, test_frac: float = 0.15
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """시간·인덱스 누수 없이 단순 분할(윈도우가 이미 독립 생성됨)."""
    n = clean.shape[0]
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    return {
        "train": (clean[:n_train], noisy[:n_train]),
        "val": (clean[n_train : n_train + n_val], noisy[n_train : n_train + n_val]),
        "test": (clean[n_train + n_val :], noisy[n_train + n_val :]),
    }
