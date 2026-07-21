# signal-ml-lab — 시리즈 개요

DSP 알고리즘 엔지니어가 ML로 넘어가며, ECG 신호처리를 소재로 **"좋아 보이는 것과
실제로 옳은 것은 다르며, 그 간극은 지표 하나로는 안 보인다"**를 실증한 기록. 모든
데이터는 공개(PhysioNet)·합성이며, 어떤 회사의 독점 데이터·알고리즘도 없다.

## 한 줄 서사

> ML은 인분포에서 고전 DSP를 압도하지만, 낯선 분포에서 **조용히** 무너진다. 그
> 무너짐은 실데이터에서도, 파이프라인 결합에서도 반복된다. 해법은 매번 같다 —
> **간극을 드러내는 실험을 설계하고, 목적을 정렬하고, 불확실할 땐 넘긴다.**

## 편별 지도

| 편 | 주제 | 핵심 결과 |
|----|------|-----------|
| [1](01_dsp_to_ml_ecg_denoising.md) | 고전 DSP vs ML 디노이징 | ML이 인분포에서 **+10~15 dB** 압승 |
| [2](02_generalization_tax.md) | 일반화 비용 | OOD에서 ML **조용히 붕괴**, 임상 형태 왜곡 |
| [3](03_knowing_when_wrong.md) | 불확실도 게이트 | 딥앙상블이 OOD 감지(AUROC 0.90); MC-dropout은 실패 |
| [4](04_sim_to_real.md) | sim-to-real 격차 | 합성학습 모델이 실 ECG서 **고전보다 못함**(4.1<6.7 dB) |
| [5](05_real_data_trilogy.md) | 데이터효율·진단 | 실데이터 100개로 고전 넘음; 부정맥 분류(환자독립) 81.8% |
| [6](06_denoising_hurts_diagnosis.md) | 파이프라인 함정 | 디노이징이 진단을 **해침**(PVC recall 87→58%) |
| [7](07_fixing_the_coupling.md) | 정렬로 해결 | 공동학습이 두 축 모두 clean 근접; 엣지 430× 실시간 |

## 관통하는 원리

1. **간극은 지표 하나로 안 보인다.** SNR만 보면 ML이 이긴다. OOD·실데이터·다운스트림
   과제로 재야 진실이 드러난다.
2. **조용한 실패가 가장 위험하다.** 깨끗해 보이는 오답은 사람이 못 잡는다 — 불확실도
   게이트·선택적 진단으로 잡는다.
3. **부분 최적화는 조립되지 않는다.** 부품을 시스템 목적에 정렬해야 파이프라인이 산다.
4. **정확함이 곧 쓸모는 아니다.** 배포 가능성(실시간·풋프린트)까지 봐야 완성이다.

## 재현

```bash
pip install -r requirements.txt
python scripts/01_classical_denoise.py      # 고전 베이스라인
python scripts/03_benchmark.py              # 고전 vs ML
python scripts/08_real_data.py --train-real # 실데이터 검증
python scripts/13_joint_training.py         # 결합 해결
python scripts/15_selective_diagnosis.py    # 선택적 진단
pytest -q
```
전체 스크립트 01~15, 코드·그림 모두 재현 가능. 상세는 [README](../README.md).
