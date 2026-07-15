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

    def __init__(self, depth: int = 8, channels: int = 32, kernel: int = 9):
        super().__init__()
        pad = kernel // 2
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
        layers += [nn.Conv1d(channels, 1, kernel, padding=pad)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.net(x)
        return x - noise  # 잔차: 입력에서 추정 노이즈를 뺌
