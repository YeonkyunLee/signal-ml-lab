"""핵심 모듈 단위 테스트.  실행: pytest -q"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_ml_lab import classical, dataset, metrics, noise, synth


def test_synth_shape_and_finite():
    t, x = synth.synth_ecg(duration_s=5.0, fs=360, hr_bpm=60)
    assert t.shape == x.shape
    assert x.size == 5 * 360
    assert np.all(np.isfinite(x))


def test_noise_increases_error():
    _, clean = synth.synth_ecg(duration_s=5.0, fs=360)
    noisy = noise.add_noise(clean, fs=360)
    # 노이즈는 신호를 오염 → RMSE > 0
    assert metrics.rmse(clean, noisy) > 0


def test_classical_improves_snr():
    _, clean = synth.synth_ecg(duration_s=10.0, fs=360)
    noisy = noise.add_noise(clean, fs=360)
    den = classical.bandpass_notch(noisy, fs=360)
    assert metrics.snr_db(clean, den) > metrics.snr_db(clean, noisy)


def test_metrics_perfect_reconstruction():
    _, clean = synth.synth_ecg(duration_s=2.0, fs=360)
    assert metrics.rmse(clean, clean) == 0.0
    assert np.isinf(metrics.snr_db(clean, clean))


def test_synth_variant_differs():
    _, base = synth.synth_ecg(duration_s=5.0, fs=360, hr_bpm=75, seed=0)
    _, var = synth.synth_ecg_variant(duration_s=5.0, fs=360, hr_bpm=75, seed=0)
    assert base.shape == var.shape
    assert np.all(np.isfinite(var))
    # 형태가 실제로 다른 분포여야 함
    assert metrics.rmse(base, var) > 0.05


def test_mixed_windows_shapes():
    clean, noisy = dataset.make_mixed_windows(16, win_len=256, variant_frac=0.5, seed=0)
    assert clean.shape == (16, 256) == noisy.shape
    assert np.all(np.isfinite(clean)) and np.all(np.isfinite(noisy))


def test_dataset_shapes_and_split():
    clean, noisy = dataset.make_windows(20, win_len=256, seed=0)
    assert clean.shape == (20, 256) == noisy.shape
    assert clean.dtype == np.float32
    parts = dataset.split(clean, noisy, val_frac=0.2, test_frac=0.2)
    total = sum(p[0].shape[0] for p in parts.values())
    assert total == 20
    assert parts["train"][0].shape[0] == 12
