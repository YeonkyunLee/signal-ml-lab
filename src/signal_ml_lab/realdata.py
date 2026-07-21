"""실제 PhysioNet 데이터 로더 + 실제 노이즈 합성.

- 실 ECG: MIT-BIH Arrhythmia DB(mitdb), MLII 리드를 기준(ground truth)으로 사용
- 실 노이즈: MIT-BIH Noise Stress Test DB(nstdb) — bw(기저선), ma(근전도), em(전극)
- 실 ECG에 실 노이즈를 목표 SNR로 입혀 (clean, noisy) 쌍을 만든다.

회사망 SSL 프록시는 truststore로 우회(Windows 인증서 저장소 주입). 다운로드는
로컬 npz로 캐시해 재실행을 빠르게 한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_CACHE = Path(__file__).resolve().parents[2] / "data_cache"


def _wfdb():
    try:
        import truststore

        truststore.inject_into_ssl()  # 회사 프록시 CA 신뢰(Windows 저장소)
    except Exception:
        pass
    import wfdb

    return wfdb


def _load_record(name: str, db: str, lead: str | None) -> tuple[np.ndarray, int]:
    _CACHE.mkdir(exist_ok=True)
    cache = _CACHE / f"{db}_{name}.npz"
    if cache.exists():
        d = np.load(cache)
        sig, fs, names = d["sig"], int(d["fs"]), list(d["names"])
    else:
        wfdb = _wfdb()
        rec = wfdb.rdrecord(name, pn_dir=db)
        sig = np.asarray(rec.p_signal, float)
        fs = int(rec.fs)
        names = list(rec.sig_name)
        np.savez_compressed(cache, sig=sig, fs=fs, names=np.array(names))
    ch = names.index(lead) if (lead and lead in names) else 0
    return sig[:, ch], fs


def load_ecg(records=("100", "103", "115", "215"), lead="MLII") -> list[np.ndarray]:
    """여러 실 ECG 레코드의 단일 리드를 반환(각각 표준화)."""
    out = []
    for r in records:
        sig, _ = _load_record(r, "mitdb", lead)
        sig = sig - np.median(sig)
        out.append(sig.astype(np.float32))
    return out


def load_noise(kinds=("bw", "ma", "em")) -> dict[str, np.ndarray]:
    """NSTDB 실 노이즈 채널들을 반환."""
    noise = {}
    for k in kinds:
        sig, _ = _load_record(k, "nstdb", None)
        noise[k] = sig.astype(np.float32)
    return noise


# MIT-BIH 주석 심볼 → AAMI 5클래스 (N 정상, S 상심실성, V 심실성, F 융합, Q 미상)
_AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}


def load_beats(
    records, win: int = 256, lead: str = "MLII"
) -> tuple[np.ndarray, list[str]]:
    """R-피크 주석 기준으로 박동 윈도우를 잘라 (beats, AAMI 라벨)로 반환.

    각 박동은 R-피크 중심 win 샘플, 개별 표준화(zero-mean/unit-std).
    """
    _CACHE.mkdir(exist_ok=True)
    half = win // 2
    beats, labels = [], []
    for rec in records:
        cache = _CACHE / f"beats_{rec}_{win}.npz"
        if cache.exists():
            d = np.load(cache, allow_pickle=True)
            beats.append(d["beats"])
            labels.extend(list(d["labels"]))
            continue
        wfdb = _wfdb()
        r = wfdb.rdrecord(rec, pn_dir="mitdb")
        ann = wfdb.rdann(rec, "atr", pn_dir="mitdb")
        names = list(r.sig_name)
        ch = names.index(lead) if lead in names else 0
        sig = np.asarray(r.p_signal[:, ch], float)
        rec_beats, rec_labels = [], []
        for s, sym in zip(ann.sample, ann.symbol):
            cls = _AAMI.get(sym)
            if cls is None:
                continue
            if s - half < 0 or s + half >= sig.size:
                continue
            w = sig[s - half : s + half]
            w = (w - w.mean()) / (w.std() + 1e-6)
            rec_beats.append(w.astype(np.float32))
            rec_labels.append(cls)
        rb = np.stack(rec_beats) if rec_beats else np.empty((0, win), np.float32)
        np.savez_compressed(cache, beats=rb, labels=np.array(rec_labels))
        beats.append(rb)
        labels.extend(rec_labels)
    return np.concatenate(beats, axis=0), labels


def add_real_noise(
    clean: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """clean에 noise를 목표 SNR로 스케일해 더한다(랜덤 구간 절취)."""
    n = clean.size
    if noise.size <= n:
        noise = np.tile(noise, int(np.ceil((n + 1) / noise.size)))
    off = int(rng.integers(0, noise.size - n))
    seg = noise[off : off + n]
    ps = float(np.mean(clean**2))
    pn = float(np.mean(seg**2)) + 1e-12
    scale = np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return (clean + scale * seg).astype(np.float32)


def make_real_pairs(
    n_windows: int,
    win_len: int = 1024,
    snr_range: tuple[float, float] = (-2.0, 8.0),
    records=("100", "103", "115", "215"),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """실 ECG + 실 노이즈로 (clean, noisy) 윈도우 쌍 생성.

    주의: 실 ECG를 '깨끗한 기준'으로 쓰지만 실제로는 소량의 잔여 잡음을 포함한다
    (관례적 근사). 추가하는 NSTDB 노이즈가 훨씬 크므로 평가에는 충분히 유효하다.
    """
    rng = np.random.default_rng(seed)
    ecgs = load_ecg(records)
    noise = load_noise()
    noise_keys = list(noise.keys())

    C = np.empty((n_windows, win_len), np.float32)
    N = np.empty((n_windows, win_len), np.float32)
    for i in range(n_windows):
        e = ecgs[int(rng.integers(len(ecgs)))]
        off = int(rng.integers(0, e.size - win_len))
        clean = e[off : off + win_len]
        clean = clean - clean.mean()
        k = noise_keys[int(rng.integers(len(noise_keys)))]
        snr = float(rng.uniform(*snr_range))
        C[i] = clean
        N[i] = add_real_noise(clean, noise[k], snr, rng)
    return C, N
