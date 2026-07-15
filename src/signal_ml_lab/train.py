"""디노이저 학습 루프."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 0


def _loader(clean: np.ndarray, noisy: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    # (N, L) -> (N, 1, L)
    c = torch.from_numpy(clean).unsqueeze(1)
    n = torch.from_numpy(noisy).unsqueeze(1)
    return DataLoader(TensorDataset(n, c), batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    cfg: TrainConfig,
) -> dict[str, list[float]]:
    """모델을 학습하고 epoch별 train/val 손실 이력을 반환한다."""
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    model.to(dev)

    tr_c, tr_n = data["train"]
    va_c, va_n = data["val"]
    train_dl = _loader(tr_c, tr_n, cfg.batch_size, shuffle=True)
    val_dl = _loader(va_c, va_n, cfg.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.MSELoss()

    history: dict[str, list[float]] = {"train": [], "val": []}
    for ep in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_dl.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                va_loss += loss_fn(model(xb), yb).item() * xb.size(0)
        va_loss /= len(val_dl.dataset)

        sched.step()
        history["train"].append(tr_loss)
        history["val"].append(va_loss)
        print(f"epoch {ep+1:3d}/{cfg.epochs}  train={tr_loss:.5f}  val={va_loss:.5f}")

    return history


@torch.no_grad()
def denoise(model: nn.Module, noisy: np.ndarray, device: str = "cpu") -> np.ndarray:
    """(N, L) 또는 (L,) 신호를 디노이징. 입력과 같은 shape 반환."""
    model.eval()
    x = np.asarray(noisy, dtype=np.float32)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    t = torch.from_numpy(x).unsqueeze(1).to(device)
    out = model(t).squeeze(1).cpu().numpy()
    return out[0] if single else out
