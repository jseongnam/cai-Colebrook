# 추가 검증 실험 전체 코드

이 디렉터리는 `cai-Colebrook` 원고의 다음 네 가지 보완 실험을 한 번에 수행한다.

1. baseline 입력 제거 ablation
2. two-branch coupled system의 end-to-end runtime 비교
3. 분포 외(OOD) 일반화 시험
4. 3-branch 및 5-branch parallel system 확장 시험

핵심 주장 자체를 추가 검증하는 correction/direct × logit/raw-ratio 2×2 ablation과
batch-1/배치 runtime은 `ADDITIONAL_EXPERIMENTS_README_KO.md` 및
`run_additional_experiments.sh`를 사용한다.

## 먼저 확인해야 할 데이터 누출 문제

기존 `scripts/data/generate_multidim_colebrook_parallel2.py`는 확인한 버전에서 `center = z_star`로 저장한다. 기존 학습 코드는 이 `center`를 입력에 연결하므로, 그 데이터로 얻은 성능은 target leakage 검토가 필요하다.

다음 명령은 기존 NPZ에서 `center`와 `target`이 같은 샘플 비율을 검사한다.

```bash
cd extended_experiments
python run_experiment.py audit-legacy ../multi_colebrook_data/*.npz \
  --output legacy_leakage_audit.json
```

누출이 발견되면 종료 코드 2를 반환한다. 새 generator는 정답으로 입력 feature를 만들지 않는다. 모든 baseline feature는 conductance/Haaland initializer에서만 계산된다.

## 설치

가장 간단한 방법은 다음 한 줄이다. 가상환경과 패키지가 없으면 자동으로 설치하고, 데이터부터 checkpoint와 평가표까지 순서대로 생성한다.

```bash
cd extended_experiments
bash run_from_zero.sh paper
```

CUDA 사용 가능 여부를 자동으로 확인한다. 강제로 지정하려면 `DEVICE=cuda` 또는 `DEVICE=cpu`를 앞에 붙인다.

전체 실행 전 작은 규모로 PT 생성까지 확인하려면:

```bash
bash run_from_zero.sh smoke
```

완료된 NPZ, PT, 평가 CSV는 다시 실행할 때 자동으로 건너뛴다. 처음부터 다시 계산해야 할 때만 `FORCE=1`을 사용한다.

수동 설치 방법:

```bash
cd extended_experiments
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_smoke_test.sh
```

GPU 사용 시 설치 환경에 맞는 PyTorch CUDA wheel을 사용한다.

## 빠른 전체 점검

작은 데이터로 파이프라인만 확인할 때:

```bash
TRAIN_N=200 VAL_N=50 TEST_N=50 EPOCHS=3 DEVICE=cpu \
  DATA_ROOT=data_smoke RESULT_ROOT=results_smoke bash run_full_suite.sh
```

이 실행은 통계적으로 의미 있는 논문 결과가 아니라 코드 연결 확인용이다.

## 논문용 전체 실행

```bash
DEVICE=cuda DATA_ROOT=data_extended RESULT_ROOT=results_extended \
  bash run_full_suite.sh
```

기본값은 train/validation/test = 20,000/4,000/4,000, epoch 최대 300, 독립 학습 seed 42/43/44이다. validation early stopping을 사용한다.

## 실험 1: baseline-input ablation

세 가지 입력 조건을 동일한 correction target과 decoder로 비교한다.

| mode | encoder가 받는 정보 | 목적 |
|---|---|---|
| `full` | 물리 파라미터 + baseline state + baseline residual | 전체 제안법 |
| `no_state` | 물리 파라미터 + baseline residual | 명시적인 q0/x0 state 제거 |
| `physics_only` | 물리 파라미터만 | baseline에서 파생된 모든 encoder 정보 제거 |

세 조건 모두 correction target과 logit-space decoder에는 동일한 baseline을 사용한다. 따라서 “baseline-relative target의 효과”와 “baseline을 encoder에 제공하는 효과”를 구분할 수 있다.

수동 실행 예:

```bash
python run_experiment.py train \
  --train data_extended/b2_iid_train.npz \
  --val data_extended/b2_iid_val.npz \
  --input-mode physics_only --seed 42 --device cuda \
  --output results_extended/b2/physics_only/seed_42/model.pt
```

## 실험 2: coupled runtime

`runtime` 명령은 checkpoint loading과 데이터 파일 I/O를 timing에서 제외한다. 두 파이프라인 모두 baseline 생성 시간을 포함한다.

- baseline pipeline: baseline 생성 + damped Newton
- neural pipeline: baseline 생성 + feature/scaling + neural forward + damped Newton

동일한 sample, Newton tolerance, CPU/GPU 조건을 사용하고 median 및 IQR을 저장한다.

```bash
python run_experiment.py runtime \
  --checkpoint results_extended/b2/full/seed_42/model.pt \
  --data data_extended/b2_iid_test.npz --device cuda \
  --samples 500 --warmup 2 --repeats 10 \
  --output results_extended/b2/full/seed_42/runtime_iid.csv
```

GPU 결과와 별도로 `--device cpu` 결과도 보고하는 것을 권장한다.

## 실험 3: OOD 평가

학습은 `iid` 범위에서만 수행하고 동일 checkpoint를 다음 범위에 그대로 평가한다.

- `high_flow`: 학습 범위보다 큰 total flow
- `high_roughness`: 학습 범위보다 큰 roughness
- `extreme_diameter_ratio`: 한 branch가 매우 작은 직경을 갖는 조건
- `combined`: flow, roughness, diameter ratio, viscosity를 동시에 이동

각 OOD generator는 실제 해에서 모든 branch가 `Re >= 4000`인 샘플만 수락한다. OOD test를 보고할 때는 direct residual, Newton iteration, convergence ratio를 모두 제시한다.

## 실험 4: multi-branch 확장

물리 solver와 모델은 branch 개수를 고정식으로 코딩하지 않는다.

- 상태: branch flow fractions와 branch별 Colebrook x
- 제약: 모든 flow는 양수이고 합은 total flow
- correction: branch flow logit correction + additive x correction
- 모델: permutation-equivariant DeepSets branch encoder
- 평가: 2, 3, 5 branch에서 동일 metric 사용

`run_full_suite.sh`는 3-branch와 5-branch 데이터도 생성하고 각각 독립 seed 3회로 학습·평가한다. 논문에서는 이것을 topology-size validation으로 기술해야 하며, looped network 검증이라고 표현하면 안 된다.

## 결과 파일

전체 실행 후 다음 파일이 생성된다.

- `paper_evaluation_summary.csv/.tex`: seed 평균·표준편차
- `paper_runtime_summary.csv/.tex`: runtime median/IQR의 seed 요약
- `all_evaluation_rows.csv`: 모든 원시 평가 행
- `all_runtime_rows.csv`: 모든 runtime 행
- 각 checkpoint 아래 `eval_*.csv`, `runtime_*.csv`

주요 metric:

- `q_relative_rmse`
- `x_rmse`
- `direct_residual_mean`, `direct_residual_p90`
- `newton_iterations_mean`, `newton_iterations_p90`
- `convergence_ratio`
- `final_residual_mean`

## 공정한 보고 원칙

- OOD checkpoint 재학습 또는 OOD 기반 hyperparameter 선택 금지
- seed별 결과를 선택하지 말고 모두 평균±표준편차로 보고
- runtime에서 checkpoint load와 disk I/O 제외
- runtime hardware, thread 수, PyTorch/NumPy 버전 기록
- 3/5 branch 결과를 일반 loop network 결과로 과장하지 않기
- 기존 누출 가능 데이터의 수치를 새 leakage-safe 결과와 섞지 않기
