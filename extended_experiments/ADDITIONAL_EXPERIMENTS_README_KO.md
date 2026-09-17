# 핵심 주장 추가 검증 실행법

이 스크립트는 같은 leakage-safe 데이터, 동일 모델 크기, 동일 seed를 사용해 다음 2×2 실험을 수행한다.

| 실험 폴더 | 학습 target | 유량 표현 |
|---|---|---|
| `correction_logit` | baseline-relative correction | logit + softmax |
| `direct_logit` | absolute solution | logit + softmax |
| `correction_raw_ratio` | baseline-relative correction | raw flow ratio + normalization |
| `direct_raw_ratio` | absolute solution | raw flow ratio + normalization |

따라서 `correction_logit` 대 `direct_logit`은 correction formulation의 효과를, `correction_logit` 대 `correction_raw_ratio`는 logit-space parameterization의 효과를 검증한다. 네 조건 전체를 함께 보고하면 두 요인의 상호작용도 숨기지 않는다.

## 기존 결과에 이어서 실행

```bash
cd extended_experiments
PYTHON_BIN=.venv/bin/python DEVICE=cuda bash run_additional_experiments.sh
```

기존 NPZ와 checkpoint는 검증 후 재사용하며, 없는 데이터와 checkpoint만 생성한다. 기본 설정은 seed 42/43/44, epoch 최대 300, IID 및 네 가지 OOD 평가이다.

CPU만 사용할 때:

```bash
PYTHON_BIN=.venv/bin/python DEVICE=cpu bash run_additional_experiments.sh
```

파이프라인 확인용 소규모 실행:

```bash
PYTHON_BIN=.venv/bin/python DEVICE=cpu TRAIN_N=200 VAL_N=50 TEST_N=50 \
EPOCHS=3 SEEDS=42 RUN_OOD=0 RUNTIME_SAMPLES=30 RUNTIME_REPEATS=2 \
DATA_ROOT=data_factorial_smoke RESULT_ROOT=results_factorial_smoke \
bash run_additional_experiments.sh
```

## runtime 설계

제안법의 전체 시간을 baseline 생성, feature/scaling, 신경망 추론, Newton refinement를 모두 포함해 측정한다. 비교 baseline도 baseline 생성과 Newton refinement를 포함한다.

- `batch1`: baseline, feature/scaling, 신경망 추론, Newton을 한 문제씩 수행하는 온라인 latency 조건
- `batch500`: 500개를 한 번에 전달하는 throughput 조건
- CPU와 CUDA가 모두 사용 가능하면 양쪽을 자동 측정
- 각 반복의 원시 ms/sample 값, median, IQR을 CSV에 함께 저장

기본 반복 수는 30회이다. 논문에는 hardware, software version, thread 수와 함께 median [IQR]을 보고한다.

## 결과 위치

- checkpoint: `results_extended/b2/<variant>/seed_<seed>/model.pt`
- 평가: 같은 폴더의 `eval_<regime>.csv`
- runtime: `correction_logit` 폴더의 `runtime_iid_<device>_batch<size>.csv`
- 통합 표: `results_extended/paper_evaluation_summary.csv/.tex`, `paper_runtime_summary.csv/.tex`
- paired effect/95% CI: `results_extended/factorial_paired_comparisons.csv`

paired 비교표의 `paired_difference_a_minus_b`는 A−B이다. 오차·residual·iteration은 음수일수록 A가 우수하고, convergence ratio는 양수일수록 A가 우수하다. seed가 3개이면 신뢰구간과 p-value의 불확실성이 크므로 개별 seed 차이(`seed_differences`)와 효과 크기를 함께 보고한다.

기존 `results_extended/b2/full` checkpoint는 version-1 호환 경로로 계속 읽을 수 있지만, factorial 표에는 새 `correction_logit` 결과를 사용하는 편이 폴더와 실험 조건을 명확히 하는 데 유리하다.
