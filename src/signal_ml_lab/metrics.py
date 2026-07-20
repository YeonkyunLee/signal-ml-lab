"""디노이징 정량 평가 지표."""

from __future__ import annotations

import numpy as np


def rmse(clean: np.ndarray, est: np.ndarray) -> float:
    """제곱근평균제곱오차 (낮을수록 좋음)."""
    return float(np.sqrt(np.mean((clean - est) ** 2)))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """pos 점수가 neg보다 큰 경향의 AUROC (순위합/Mann-Whitney U 기반).

    OOD 탐지 평가에 사용: pos=OOD 점수, neg=in-dist 점수. 0.5=무의미, 1.0=완벽.
    """
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    order = np.argsort(scores)
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1)
    n_pos, n_neg = pos.size, neg.size
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def snr_db(clean: np.ndarray, est: np.ndarray) -> float:
    """신호 대 잡음비 [dB] (높을수록 좋음).

    잡음 = 추정신호 - 깨끗한신호.
    """
    signal_power = float(np.mean(clean**2))
    noise_power = float(np.mean((est - clean) ** 2))
    if noise_power == 0:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)
