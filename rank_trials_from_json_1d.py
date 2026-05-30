import os
import glob
import json
import math
from typing import Dict, List, Tuple, Any

BASE_DIR = "/home/seokjun/math_03_14/grid_runs_1d/deg25_search"

MODEL_TYPES = ["mlp", "lstm", "gru", "transformer"]

# ------------------------------------------------------------------
# 1) 점수 계산에 사용할 지표와 가중치
#    - lower is better: 음수 방향으로 반영
#    - higher is better: 양수 방향으로 반영
#
# 권장 철학:
# - 최종적으로는 plus_newton 성능과 iteration 효율이 중요
# - direct stage 품질도 중요
# - converged_ratio는 아주 중요
# ------------------------------------------------------------------

WEIGHTS = {
    # direct stage
    "direct_mae": 0.10,
    "direct_rmse": 0.15,
    "direct_residual_mean": 0.08,
    "direct_residual_p90": 0.07,
    "direct_r2": 0.05,
    "direct_valid_ratio": 0.05,

    # plus newton stage
    "plus_newton_mae": 0.08,
    "plus_newton_rmse": 0.12,
    "plus_newton_residual_mean": 0.05,
    "plus_newton_residual_p90": 0.05,
    "plus_newton_r2": 0.03,
    "plus_newton_valid_ratio": 0.03,

    # Newton efficiency
    "plus_newton_newton_iter_mean": 0.08,
    "plus_newton_newton_iter_median": 0.02,
    "plus_newton_newton_iter_p90": 0.02,
    "plus_newton_converged_ratio": 0.10,
}

LOWER_IS_BETTER = {
    "direct_mae",
    "direct_rmse",
    "direct_residual_mean",
    "direct_residual_p90",
    "plus_newton_mae",
    "plus_newton_rmse",
    "plus_newton_residual_mean",
    "plus_newton_residual_p90",
    "plus_newton_newton_iter_mean",
    "plus_newton_newton_iter_median",
    "plus_newton_newton_iter_p90",
}

HIGHER_IS_BETTER = {
    "direct_r2",
    "direct_valid_ratio",
    "plus_newton_r2",
    "plus_newton_valid_ratio",
    "plus_newton_converged_ratio",
}


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def load_json_files(base_dir: str, model_type: str) -> List[Dict[str, Any]]:
    pattern = os.path.join(base_dir, f"trial_*_{model_type}.json")
    files = sorted(glob.glob(pattern))

    records = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_file"] = fp
            data["_model_type"] = model_type
            records.append(data)
        except Exception as e:
            print(f"[WARN] Failed to load {fp}: {e}")
    return records


def get_metric_values(records: List[Dict[str, Any]], metric: str) -> List[float]:
    vals = []
    for r in records:
        v = safe_float(r.get(metric, float("nan")))
        if not math.isnan(v):
            vals.append(v)
    return vals


def min_max_normalize(value: float, vmin: float, vmax: float) -> float:
    if math.isnan(value):
        return 0.0
    if vmax - vmin < 1e-15:
        return 1.0
    return (value - vmin) / (vmax - vmin)


def compute_score_within_model(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    같은 모델 타입(예: MLP trial들끼리) 안에서만 normalize 후 점수 계산.
    이렇게 해야 MLP끼리 누가 제일 좋은지 뽑는 데 안정적.
    """

    # metric별 min/max 수집
    metric_stats: Dict[str, Tuple[float, float]] = {}
    for metric in WEIGHTS.keys():
        vals = get_metric_values(records, metric)
        if len(vals) == 0:
            metric_stats[metric] = (float("nan"), float("nan"))
        else:
            metric_stats[metric] = (min(vals), max(vals))

    scored = []
    for r in records:
        total_score = 0.0
        detail = {}

        for metric, weight in WEIGHTS.items():
            raw_val = safe_float(r.get(metric, float("nan")))
            vmin, vmax = metric_stats[metric]

            if math.isnan(raw_val) or math.isnan(vmin) or math.isnan(vmax):
                contrib = 0.0
                norm_good = 0.0
            else:
                norm = min_max_normalize(raw_val, vmin, vmax)

                # 좋은 값일수록 큰 점수가 되게 변환
                if metric in LOWER_IS_BETTER:
                    norm_good = 1.0 - norm
                elif metric in HIGHER_IS_BETTER:
                    norm_good = norm
                else:
                    norm_good = 0.0

                contrib = weight * norm_good

            detail[metric] = {
                "raw": raw_val,
                "weighted_goodness": contrib,
            }
            total_score += contrib

        out = dict(r)
        out["_score"] = total_score
        out["_score_detail"] = detail
        scored.append(out)

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored


def summarize_trial(trial: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "trial_id",
        "trial_name",
        "model",
        "best_epoch",
        "elapsed_sec",
        "direct_mae",
        "direct_rmse",
        "direct_r2",
        "direct_residual_mean",
        "direct_residual_p90",
        "plus_newton_mae",
        "plus_newton_rmse",
        "plus_newton_r2",
        "plus_newton_residual_mean",
        "plus_newton_residual_p90",
        "plus_newton_newton_iter_mean",
        "plus_newton_newton_iter_median",
        "plus_newton_newton_iter_p90",
        "plus_newton_converged_ratio",
        "_score",
        "_file",
    ]
    return {k: trial.get(k) for k in keys}


def find_best_trials(base_dir: str) -> Dict[str, Dict[str, Any]]:
    results = {}

    for model_type in MODEL_TYPES:
        records = load_json_files(base_dir, model_type)
        if not records:
            print(f"[WARN] No files found for model type: {model_type}")
            results[model_type] = {}
            continue

        scored = compute_score_within_model(records)
        best_trial = scored[0]
        results[model_type] = {
            "best": summarize_trial(best_trial),
            "top3": [summarize_trial(x) for x in scored[:3]],
            "count": len(scored),
        }

    return results


def print_results(results: Dict[str, Dict[str, Any]]) -> None:
    print("=" * 100)
    print("BEST TRIAL FOR EACH MODEL TYPE")
    print("=" * 100)

    for model_type in MODEL_TYPES:
        info = results.get(model_type, {})
        if not info:
            print(f"\n[{model_type.upper()}] No result")
            continue

        print(f"\n[{model_type.upper()}] total trials = {info['count']}")
        best = info["best"]
        print(f"  trial_name                 : {best.get('trial_name')}")
        print(f"  score                      : {best.get('_score'):.6f}")
        print(f"  best_epoch                 : {best.get('best_epoch')}")
        print(f"  elapsed_sec                : {best.get('elapsed_sec')}")
        print(f"  direct_rmse                : {best.get('direct_rmse')}")
        print(f"  direct_r2                  : {best.get('direct_r2')}")
        print(f"  direct_residual_mean       : {best.get('direct_residual_mean')}")
        print(f"  plus_newton_rmse           : {best.get('plus_newton_rmse')}")
        print(f"  plus_newton_iter_mean      : {best.get('plus_newton_newton_iter_mean')}")
        print(f"  plus_newton_iter_median    : {best.get('plus_newton_newton_iter_median')}")
        print(f"  plus_newton_iter_p90       : {best.get('plus_newton_newton_iter_p90')}")
        print(f"  plus_newton_converged_ratio: {best.get('plus_newton_converged_ratio')}")
        print(f"  file                       : {best.get('_file')}")

        print("  Top-3 trials:")
        for i, t in enumerate(info["top3"], start=1):
            print(
                f"    {i}. {t.get('trial_name')} | "
                f"score={t.get('_score'):.6f}, "
                f"direct_rmse={t.get('direct_rmse')}, "
                f"plus_newton_iter_mean={t.get('plus_newton_newton_iter_mean')}, "
                f"conv={t.get('plus_newton_converged_ratio')}"
            )


if __name__ == "__main__":
    results = find_best_trials(BASE_DIR)
    print_results(results)import os
import glob
import json
import math
from typing import Dict, List, Tuple, Any

BASE_DIR = "/home/seokjun/math_03_14/grid_runs_1d/deg25_search"

MODEL_TYPES = ["mlp", "lstm", "gru", "transformer"]

# ------------------------------------------------------------------
# 1) 점수 계산에 사용할 지표와 가중치
#    - lower is better: 음수 방향으로 반영
#    - higher is better: 양수 방향으로 반영
#
# 권장 철학:
# - 최종적으로는 plus_newton 성능과 iteration 효율이 중요
# - direct stage 품질도 중요
# - converged_ratio는 아주 중요
# ------------------------------------------------------------------

WEIGHTS = {
    # direct stage
    "direct_mae": 0.10,
    "direct_rmse": 0.15,
    "direct_residual_mean": 0.08,
    "direct_residual_p90": 0.07,
    "direct_r2": 0.05,
    "direct_valid_ratio": 0.05,

    # plus newton stage
    "plus_newton_mae": 0.08,
    "plus_newton_rmse": 0.12,
    "plus_newton_residual_mean": 0.05,
    "plus_newton_residual_p90": 0.05,
    "plus_newton_r2": 0.03,
    "plus_newton_valid_ratio": 0.03,

    # Newton efficiency
    "plus_newton_newton_iter_mean": 0.08,
    "plus_newton_newton_iter_median": 0.02,
    "plus_newton_newton_iter_p90": 0.02,
    "plus_newton_converged_ratio": 0.10,
}

LOWER_IS_BETTER = {
    "direct_mae",
    "direct_rmse",
    "direct_residual_mean",
    "direct_residual_p90",
    "plus_newton_mae",
    "plus_newton_rmse",
    "plus_newton_residual_mean",
    "plus_newton_residual_p90",
    "plus_newton_newton_iter_mean",
    "plus_newton_newton_iter_median",
    "plus_newton_newton_iter_p90",
}

HIGHER_IS_BETTER = {
    "direct_r2",
    "direct_valid_ratio",
    "plus_newton_r2",
    "plus_newton_valid_ratio",
    "plus_newton_converged_ratio",
}


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def load_json_files(base_dir: str, model_type: str) -> List[Dict[str, Any]]:
    pattern = os.path.join(base_dir, f"trial_*_{model_type}.json")
    files = sorted(glob.glob(pattern))

    records = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_file"] = fp
            data["_model_type"] = model_type
            records.append(data)
        except Exception as e:
            print(f"[WARN] Failed to load {fp}: {e}")
    return records


def get_metric_values(records: List[Dict[str, Any]], metric: str) -> List[float]:
    vals = []
    for r in records:
        v = safe_float(r.get(metric, float("nan")))
        if not math.isnan(v):
            vals.append(v)
    return vals


def min_max_normalize(value: float, vmin: float, vmax: float) -> float:
    if math.isnan(value):
        return 0.0
    if vmax - vmin < 1e-15:
        return 1.0
    return (value - vmin) / (vmax - vmin)


def compute_score_within_model(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    같은 모델 타입(예: MLP trial들끼리) 안에서만 normalize 후 점수 계산.
    이렇게 해야 MLP끼리 누가 제일 좋은지 뽑는 데 안정적.
    """

    # metric별 min/max 수집
    metric_stats: Dict[str, Tuple[float, float]] = {}
    for metric in WEIGHTS.keys():
        vals = get_metric_values(records, metric)
        if len(vals) == 0:
            metric_stats[metric] = (float("nan"), float("nan"))
        else:
            metric_stats[metric] = (min(vals), max(vals))

    scored = []
    for r in records:
        total_score = 0.0
        detail = {}

        for metric, weight in WEIGHTS.items():
            raw_val = safe_float(r.get(metric, float("nan")))
            vmin, vmax = metric_stats[metric]

            if math.isnan(raw_val) or math.isnan(vmin) or math.isnan(vmax):
                contrib = 0.0
                norm_good = 0.0
            else:
                norm = min_max_normalize(raw_val, vmin, vmax)

                # 좋은 값일수록 큰 점수가 되게 변환
                if metric in LOWER_IS_BETTER:
                    norm_good = 1.0 - norm
                elif metric in HIGHER_IS_BETTER:
                    norm_good = norm
                else:
                    norm_good = 0.0

                contrib = weight * norm_good

            detail[metric] = {
                "raw": raw_val,
                "weighted_goodness": contrib,
            }
            total_score += contrib

        out = dict(r)
        out["_score"] = total_score
        out["_score_detail"] = detail
        scored.append(out)

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored


def summarize_trial(trial: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "trial_id",
        "trial_name",
        "model",
        "best_epoch",
        "elapsed_sec",
        "direct_mae",
        "direct_rmse",
        "direct_r2",
        "direct_residual_mean",
        "direct_residual_p90",
        "plus_newton_mae",
        "plus_newton_rmse",
        "plus_newton_r2",
        "plus_newton_residual_mean",
        "plus_newton_residual_p90",
        "plus_newton_newton_iter_mean",
        "plus_newton_newton_iter_median",
        "plus_newton_newton_iter_p90",
        "plus_newton_converged_ratio",
        "_score",
        "_file",
    ]
    return {k: trial.get(k) for k in keys}


def find_best_trials(base_dir: str) -> Dict[str, Dict[str, Any]]:
    results = {}

    for model_type in MODEL_TYPES:
        records = load_json_files(base_dir, model_type)
        if not records:
            print(f"[WARN] No files found for model type: {model_type}")
            results[model_type] = {}
            continue

        scored = compute_score_within_model(records)
        best_trial = scored[0]
        results[model_type] = {
            "best": summarize_trial(best_trial),
            "top3": [summarize_trial(x) for x in scored[:3]],
            "count": len(scored),
        }

    return results


def print_results(results: Dict[str, Dict[str, Any]]) -> None:
    print("=" * 100)
    print("BEST TRIAL FOR EACH MODEL TYPE")
    print("=" * 100)

    for model_type in MODEL_TYPES:
        info = results.get(model_type, {})
        if not info:
            print(f"\n[{model_type.upper()}] No result")
            continue

        print(f"\n[{model_type.upper()}] total trials = {info['count']}")
        best = info["best"]
        print(f"  trial_name                 : {best.get('trial_name')}")
        print(f"  score                      : {best.get('_score'):.6f}")
        print(f"  best_epoch                 : {best.get('best_epoch')}")
        print(f"  elapsed_sec                : {best.get('elapsed_sec')}")
        print(f"  direct_rmse                : {best.get('direct_rmse')}")
        print(f"  direct_r2                  : {best.get('direct_r2')}")
        print(f"  direct_residual_mean       : {best.get('direct_residual_mean')}")
        print(f"  plus_newton_rmse           : {best.get('plus_newton_rmse')}")
        print(f"  plus_newton_iter_mean      : {best.get('plus_newton_newton_iter_mean')}")
        print(f"  plus_newton_iter_median    : {best.get('plus_newton_newton_iter_median')}")
        print(f"  plus_newton_iter_p90       : {best.get('plus_newton_newton_iter_p90')}")
        print(f"  plus_newton_converged_ratio: {best.get('plus_newton_converged_ratio')}")
        print(f"  file                       : {best.get('_file')}")

        print("  Top-3 trials:")
        for i, t in enumerate(info["top3"], start=1):
            print(
                f"    {i}. {t.get('trial_name')} | "
                f"score={t.get('_score'):.6f}, "
                f"direct_rmse={t.get('direct_rmse')}, "
                f"plus_newton_iter_mean={t.get('plus_newton_newton_iter_mean')}, "
                f"conv={t.get('plus_newton_converged_ratio')}"
            )


if __name__ == "__main__":
    results = find_best_trials(BASE_DIR)
    print_results(results)