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

## 지금 되는 것 (Phase 1: 고전 DSP 베이스라인)
- 합성 ECG 생성 + 현실적 노이즈 주입 (다운로드 없이 즉시 실행)
- 고전 디노이징: 대역통과 + 60Hz 노치 + (선택)웨이블릿
- SNR / RMSE 정량 평가 + before/after 그림 저장

## 로드맵
- [x] 합성 ECG + 노이즈 모델
- [x] 고전 DSP 디노이징 베이스라인 + 지표
- [ ] 실제 PhysioNet 레코드 로더 (`wfdb`)
- [ ] ML 디노이저 (1D U-Net / denoising autoencoder)
- [ ] 고전 vs ML 벤치마크 표 + 블로그 글

## 빠른 시작
```bash
pip install -r requirements.txt
python scripts/01_classical_denoise.py        # 합성 신호로 즉시 실행
python scripts/01_classical_denoise.py --record 100   # 실제 PhysioNet(wfdb 필요)
```
그림은 `outputs/`, 지표는 콘솔에 출력된다.

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
