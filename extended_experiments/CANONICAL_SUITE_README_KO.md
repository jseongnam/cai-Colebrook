# 동일 데이터로 통일한 최종 재실험

`run_canonical_suite.sh`는 논문에 사용할 최종 실험을 새 데이터 및 결과 디렉터리에서 다시 실행한다. 기존 `data_extended`와 `results_extended`는 읽거나 섞지 않는다.

## 실행

```bash
cd extended_experiments
PYTHON_BIN=.venv/bin/python DEVICE=cuda bash run_canonical_suite.sh
```

기본 디렉터리:

- 고정 데이터: `data_canonical_v1`
- 최종 결과: `results_canonical_v1`
- 데이터 manifest: `data_canonical_v1/dataset_manifest.json`

## 데이터 통일 방식

처음 실행할 때 다음 파일을 한 번만 생성한 후 content fingerprint로 동결한다.

- 2-branch IID train/validation/test
- 2-branch OOD 네 종류의 test
- 3-branch IID train/validation/test
- 5-branch IID train/validation/test

이후 실행에서는 NPZ를 다시 만들지 않고 모든 파일의 SHA-256 기반 content ID를 manifest와 대조한다. 값 하나라도 달라지거나 요청한 sample 수가 달라지면 실행을 중단한다. 다른 크기의 실험이 필요하면 기존 디렉터리를 덮어쓰지 말고 새 `DATA_ROOT`를 사용한다.

```bash
DATA_ROOT=data_canonical_smoke RESULT_ROOT=results_canonical_smoke \
TRAIN_N=200 VAL_N=50 TEST_N=50 EPOCHS=3 SEEDS=42 \
PYTHON_BIN=.venv/bin/python DEVICE=cpu bash run_canonical_suite.sh
```

## 동일 데이터로 실행되는 실험

- `correction_logit`
- baseline input ablation: `no_state`, `physics_only`
- factorial: `direct_logit`, `correction_raw_ratio`, `direct_raw_ratio`
- 모든 2-branch IID/OOD 평가
- 3/5-branch 일반화 평가
- CPU/CUDA batch-1 및 batch-500 runtime

checkpoint에는 `suite_id`, train dataset ID, validation dataset ID가 저장된다. 평가 및 runtime CSV에는 여기에 test dataset ID도 추가된다. 기존 checkpoint나 CSV가 발견되면 ID가 모두 일치하는 경우에만 재사용한다.

## 재실행 규칙

학습 결과만 다시 만들려면 데이터는 그대로 두고 다음을 사용한다.

```bash
FORCE_RESULTS=1 PYTHON_BIN=.venv/bin/python DEVICE=cuda bash run_canonical_suite.sh
```

`FORCE_RESULTS=1`은 데이터 파일을 변경하지 않는다. 동결 데이터 검증에 실패하면 새 데이터 버전 이름을 사용한다.

```bash
DATA_ROOT=data_canonical_v2 RESULT_ROOT=results_canonical_v2 \
PYTHON_BIN=.venv/bin/python DEVICE=cuda bash run_canonical_suite.sh
```

논문 표에는 하나의 `suite_id`에 속한 결과만 사용한다. `paper_evaluation_summary.csv`, `paper_runtime_summary.csv`, `factorial_paired_comparisons.csv`에도 ID가 포함된다.
