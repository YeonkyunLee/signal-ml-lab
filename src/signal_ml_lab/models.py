"""1D 디노이징 신경망.

DnCNN 스타일 잔차 학습 디노이저: 네트워크가 '노이즈'를 예측하고 입력에서 빼서
깨끗한 신호를 얻는다. 잔차 학습은 항등에 가까운 매핑을 쉽게 만들어 수렴이 빠르다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DnCNN1D(nn.Module):
    """1D DnCNN 잔차 디노이저.

    입력/출력: (batch, 1, length). forward는 깨끗한 신호 추정을 반환한다.
    """

    def __init__(
        self, depth: int = 8, channels: int = 32, kernel: int = 9, dropout: float = 0.0
    ):
        super().__init__()
        pad = kernel // 2
        self.dropout = dropout
        layers: list[nn.Module] = [
            nn.Conv1d(1, channels, kernel, padding=pad),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers += [
                nn.Conv1d(channels, channels, kernel, padding=pad, bias=False),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                # MC-dropout: 추론 시에도 켜서 예측 분산(불확실도)을 얻는다
                layers += [nn.Dropout(dropout)]
        layers += [nn.Conv1d(channels, 1, kernel, padding=pad)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.net(x)
        return x - noise  # 잔차: 입력에서 추정 노이즈를 뺌


class ECGBeatClassifier(nn.Module):
    """1D CNN 심박 분류기 (AAMI 클래스).

    입력 (batch, 1, win) → 클래스 로짓 (batch, n_classes).
    """

    def __init__(self, n_classes: int = 5, channels: int = 32, win: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, channels, 7, padding=3), nn.BatchNorm1d(channels), nn.ReLU(),
            nn.MaxPool1d(2),  # win/2
            nn.Conv1d(channels, channels * 2, 5, padding=2), nn.BatchNorm1d(channels * 2), nn.ReLU(),
            nn.MaxPool1d(2),  # win/4
            nn.Conv1d(channels * 2, channels * 4, 3, padding=1), nn.BatchNorm1d(channels * 4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 전역 평균 풀링
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(channels * 4, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
