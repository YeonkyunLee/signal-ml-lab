# signal-ml-lab

**고전 DSP와 머신러닝으로 생체신호(ECG)를 디노이징하고 정량 비교하는 실험실.**

DSP 알고리즘 개발자가 ML로 넘어가며 만드는 공개 학습 프로젝트. 모든 데이터는
공개 데이터셋([PhysioNet](https://physionet.org/)) 또는 합성 신호이며, 어떤
회사의 독점 데이터·알고리즘도 포함하지 않는다.

## 왜 ECG 디노이징인가
- 생체신호 디노이징은 고전 DSP(필터·웨이블릿)와 학습 기반 방법을
  **같은 지표로 정면 비교**하기 좋은 문제다.
- 노이즈 모델(기저선 변동, 전원선 60Hz, 근전도)이 물리적으로 명확해
  합성 데이터만으로도 정직한 실험이 가능하다.
- 결과가 SNR/RMSE로 딱 떨어져 "그래서 얼마나 나아졌나"를 숫자로 말할 수 있다.

## 지금 되는 것
- 합성 ECG 생성 + 현실적 노이즈 주입 (다운로드 없이 즉시 실행)
- 고전 디노이징: 대역통과 + 60Hz 노치 + (선택)웨이블릿
- **ML 디노이저: 1D DnCNN 잔차 학습 (PyTorch)**
- **고전 vs ML 벤치마크: SNR 스윕 표 + 그림**
- SNR / RMSE 정량 평가, 단위 테스트(pytest)

## 결과 (합성 테스트셋, SNR 스윕)

| input SNR | classical (bp+notch) | ML (DnCNN1D) |
|----------:|---------------------:|-------------:|
|  +5.6 dB  | 5.2 dB               | **20.4 dB**  |
|  −0.4 dB  | 5.0 dB               | **17.3 dB**  |
|  −4.0 dB  | 4.5 dB               | **13.7 dB**  |
|  −6.6 dB  | 3.7 dB               | **9.2 dB**   |
| −10.0 dB  | 2.0 dB               | 2.8 dB       |

- ML은 저~중 노이즈에서 고전 대비 **10~15 dB** 더 좋다.
- **극한 노이즈(−10 dB)에서는 우위가 거의 사라진다** — 정직한 한계.
- 웨이블릿 임계처리는 이 노이즈 모델에선 bandpass+notch 대비 이득이 거의 없다.

![benchmark](outputs/03_benchmark_snr.png)

## 도메인 시프트 — ML의 함정

원분포(정상 형태)로만 학습한 ML을, **형태가 다른 분포**(T파 역전·QRS 확대·이소성
박동)에서 테스트하면 결과가 뒤집힌다.

| input SNR | classical | ML (원분포 학습) |
|----------:|----------:|-----------------:|
|  +5.2 dB  | **12.1 dB** | 2.4 dB         |
|  −0.8 dB  | **10.2 dB** | 2.5 dB         |
|  −4.3 dB  | **8.3 dB**  | 2.9 dB         |

- 인분포에서 15 dB 압승하던 ML이 **OOD에서는 고전에 크게 진다.**
- 원인: ML이 학습한 형태 사전지식을 강요해 **역전 T파를 정상 T파로 왜곡**한다.
  겉보기엔 깨끗하지만 임상적으로 중요한 형태가 틀린다 — 의료 응용의 안전 이슈.
- 고전 필터는 분포 가정이 없어 **강건**하다.

이것이 이 프로젝트의 핵심 교훈이다: *ML의 우위는 학습분포 안에서만 유효하며, 일반화
비용(generalization tax)을 반드시 함께 봐야 한다.*

![domain shift](outputs/04_domain_shift.png)

## 로드맵
- [x] 합성 ECG + 노이즈 모델
- [x] 고전 DSP 디노이징 베이스라인 + 지표
- [x] ML 디노이저 (1D DnCNN) + 학습 파이프라인
- [x] 고전 vs ML 벤치마크 표/그림
- [x] 단위 테스트
- [x] 도메인 시프트 평가 (형태 이동 분포에서 일반화 실패 규명)
- [x] 블로그 글 초안 2편
- [ ] 실제 PhysioNet 레코드로 평가 (`wfdb`; 현재 회사망 SSL 프록시로 보류)
- [ ] 혼합 분포 학습으로 일반화 회복 실험
- [ ] 불확실도 추정(앙상블/드롭아웃)으로 OOD 출력 플래그
- [ ] 엣지 경량화 + 실시간 추론

## 빠른 시작
```bash
pip install -r requirements.txt

# 1) 고전 DSP 베이스라인 (즉시 실행)
python scripts/01_classical_denoise.py

# 2) ML 디노이저 학습 (CPU 몇 분)
python scripts/02_train_ml_denoise.py --n 4000 --epochs 30

# 3) 고전 vs ML 벤치마크
python scripts/03_benchmark.py

pytest -q   # 테스트
```
그림/표는 `outputs/`, 지표는 콘솔에 출력된다.

## 구조
```
src/signal_ml_lab/
  synth.py      합성 ECG 생성
  noise.py      현실적 노이즈 모델
  classical.py  고전 DSP 디노이징
  metrics.py    SNR / RMSE
  data.py       PhysioNet 로더 (wfdb, 선택)
scripts/
  01_classical_denoise.py   엔드투엔드 데모
```
