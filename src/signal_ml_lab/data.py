"""실제 PhysioNet 레코드 로더 (선택, wfdb 필요).

MIT-BIH Arrhythmia Database 등에서 단일 채널을 내려받아 반환한다.
네트워크와 wfdb가 없으면 합성 신호(synth)를 쓰면 된다.
"""

from __future__ import annotations

import numpy as np


def load_physionet(
    record: str = "100",
    db: str = "mitdb",
    channel: int = 0,
    max_samples: int | None = 3600,
) -> tuple[np.ndarray, np.ndarray, int]:
    """PhysioNet 레코드 한 개를 로드한다.

    Returns:
        t: 시간축 [s]
        sig: 신호 [mV]
        fs: 샘플링 주파수 [Hz]
    """
    try:
        import wfdb
    except ImportError as e:
        raise RuntimeError("wfdb 미설치: pip install wfdb") from e

    rec = wfdb.rdrecord(record, pn_dir=db)
    fs = int(rec.fs)
    sig = np.asarray(rec.p_signal[:, channel], dtype=float)
    if max_samples is not None:
        sig = sig[:max_samples]
    t = np.arange(sig.size) / fs
    return t, sig, fs
