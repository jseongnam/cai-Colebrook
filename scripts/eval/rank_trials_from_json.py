#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import math


# =========================
# 기본 설정
# =========================
# 클수록 좋은 지표
HIGHER_IS_BETTER = {
    "direct_r2",
    "direct_valid_ratio",
    "plus_newton_r2",
    "plus_newton_valid_ratio",
    "plus_newton_converged_ratio",
}

# 작을수록 좋은 지표
LOWER_IS_BETTER = {
    "direct_mae",
    "direct_rmse",
    "direct_residual_mean",
    "direct_residual_median",
    "direct_residual_p90",
    "plus_newton_mae",
    "plus_newton_rmse",
    "plus_newton_residual_mean",
    "plus_newton_residual_median",
    "plus_newton_residual_p90",
    "plus_newton_newton_iter_mean",
    "plus_newton_newton_iter_median",
    "plus_newton_newton_iter_p90",
    "elapsed_sec",
}

# direct 쪽 가중치
DEFAULT_DIRECT_WEIGHTS = {
    "direct_r2": 0.30,
    "direct_rmse": 0.25,
    "direct_mae": 0.10,
    "direct_residual_mean": 0.15,
    "direct_residual_median": 0.08,
    "direct_residual_p90": 0.10,
    "direct_valid_ratio": 0.02,
}

# plus_newton 쪽 가중치
DEFAULT_PLUS_WEIGHTS = {
    "plus_newton_r2": 0.20,
    "plus_newton_rmse": 0.20,
    "plus_newton_mae": 0.08,
    "plus_newton_residual_mean": 0.10,
    "plus_newton_residual_median": 0.06,
    "plus_newton_residual_p90": 0.10,
    "plus_newton_newton_iter_mean": 0.14,
    "plus_newton_newton_iter_median": 0.04,
    "plus_newton_newton_iter_p90": 0.05,
    "plus_newton_valid_ratio": 0.01,
    "plus_newton_converged_ratio": 0.02,
}

# overall = direct + plus_newton 혼합
DEFAULT_OVERALL_MIX = {
    "direct_score": 0.40,
    "plus_score": 0.60,
}


# =========================
# 유틸
# =========================
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def load_trial_jsons(input_dir: Path) -> List[Dict]:
    rows = []
    for p in sorted(input_dir.glob("trial_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["_json_path"] = str(p)
            rows.append(data)
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}")
    return rows


def collect_metric_values(rows: List[Dict], metric_names: List[str]) -> Dict[str, List[float]]:
    out = {m: [] for m in metric_names}
    for row in rows:
        for m in metric_names:
            val = safe_float(row.get(m, math.nan))
            out[m].append(val)
    return out


def min_max_score(value: float, vmin: float, vmax: float, higher_is_better: bool) -> float:
    if not math.isfinite(value):
        return 0.0
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return 0.0
    if abs(vmax - vmin) < 1e-15:
        # 전부 동일하면 구분력 없음
        return 0.0
    if higher_is_better:
        return (value - vmin) / (vmax - vmin)
    return (vmax - value) / (vmax - vmin)


def build_metric_stats(rows: List[Dict], metric_names: List[str]) -> Dict[str, Tuple[float, float]]:
    stats = {}
    for m in metric_names:
        vals = [safe_float(r.get(m, math.nan)) for r in rows]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            stats[m] = (math.nan, math.nan)
        else:
            stats[m] = (min(vals), max(vals))
    return stats


def weighted_score(row: Dict, weights: Dict[str, float], stats: Dict[str, Tuple[float, float]]) -> Tuple[float, Dict[str, float]]:
    score = 0.0
    contrib = {}

    for metric, weight in weights.items():
        val = safe_float(row.get(metric, math.nan))
        vmin, vmax = stats[metric]

        if metric in HIGHER_IS_BETTER:
            s = min_max_score(val, vmin, vmax, higher_is_better=True)
        elif metric in LOWER_IS_BETTER:
            s = min_max_score(val, vmin, vmax, higher_is_better=False)
        else:
            # 분류되지 않은 지표는 일단 무시
            s = 0.0

        contrib[metric] = s * weight
        score += contrib[metric]

    return score, contrib


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(weights.values())
    if s <= 0:
        raise ValueError("weights sum must be > 0")
    return {k: v / s for k, v in weights.items()}


def save_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    keys = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# =========================
# 메인 랭킹
# =========================
def rank_trials(
    rows: List[Dict],
    direct_weights: Dict[str, float],
    plus_weights: Dict[str, float],
    overall_mix: Dict[str, float],
) -> List[Dict]:
    direct_weights = normalize_weights(direct_weights)
    plus_weights = normalize_weights(plus_weights)
    overall_mix = normalize_weights(overall_mix)

    all_metric_names = sorted(set(direct_weights.keys()) | set(plus_weights.keys()))
    stats = build_metric_stats(rows, all_metric_names)

    ranked = []
    for row in rows:
        direct_score, direct_detail = weighted_score(row, direct_weights, stats)
        plus_score, plus_detail = weighted_score(row, plus_weights, stats)

        overall_score = (
            overall_mix["direct_score"] * direct_score
            + overall_mix["plus_score"] * plus_score
        )

        out = dict(row)
        out["direct_score"] = direct_score
        out["plus_score"] = plus_score
        out["overall_score"] = overall_score

        for k, v in direct_detail.items():
            out[f"contrib_direct__{k}"] = v
        for k, v in plus_detail.items():
            out[f"contrib_plus__{k}"] = v

        ranked.append(out)

    ranked.sort(key=lambda x: x["overall_score"], reverse=True)

    for i, row in enumerate(ranked, start=1):
        row["overall_rank"] = i

    # direct 기준 순위
    direct_sorted = sorted(ranked, key=lambda x: x["direct_score"], reverse=True)
    for i, row in enumerate(direct_sorted, start=1):
        row["direct_rank"] = i

    # plus 기준 순위
    plus_sorted = sorted(ranked, key=lambda x: x["plus_score"], reverse=True)
    for i, row in enumerate(plus_sorted, start=1):
        row["plus_rank"] = i

    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="trial_*.json 파일들이 들어있는 폴더",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="랭킹 결과 저장 폴더",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="콘솔에 출력할 상위 개수",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_trial_jsons(input_dir)
    if not rows:
        raise RuntimeError(f"No trial_*.json found in {input_dir}")

    ranked = rank_trials(
        rows=rows,
        direct_weights=DEFAULT_DIRECT_WEIGHTS,
        plus_weights=DEFAULT_PLUS_WEIGHTS,
        overall_mix=DEFAULT_OVERALL_MIX,
    )

    # overall 저장
    save_csv(output_dir / "ranking_all.csv", ranked)
    (output_dir / "ranking_all.json").write_text(
        json.dumps(ranked, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # direct / plus 별도 저장
    direct_sorted = sorted(ranked, key=lambda x: x["direct_score"], reverse=True)
    plus_sorted = sorted(ranked, key=lambda x: x["plus_score"], reverse=True)

    save_csv(output_dir / "ranking_direct.csv", direct_sorted)
    save_csv(output_dir / "ranking_plus_newton.csv", plus_sorted)

    summary = {
        "num_trials": len(ranked),
        "direct_weights": DEFAULT_DIRECT_WEIGHTS,
        "plus_weights": DEFAULT_PLUS_WEIGHTS,
        "overall_mix": DEFAULT_OVERALL_MIX,
        "best_overall": {
            "trial_id": ranked[0]["trial_id"],
            "trial_name": ranked[0]["trial_name"],
            "model": ranked[0]["model"],
            "overall_score": ranked[0]["overall_score"],
            "direct_score": ranked[0]["direct_score"],
            "plus_score": ranked[0]["plus_score"],
        },
    }
    (output_dir / "ranking_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("=== TOP OVERALL ===")
    for row in ranked[:args.topk]:
        print({
            "overall_rank": row["overall_rank"],
            "trial_id": row["trial_id"],
            "model": row["model"],
            "overall_score": row["overall_score"],
            "direct_score": row["direct_score"],
            "plus_score": row["plus_score"],
            "direct_rmse": row.get("direct_rmse"),
            "plus_newton_rmse": row.get("plus_newton_rmse"),
            "plus_newton_newton_iter_mean": row.get("plus_newton_newton_iter_mean"),
        })

    print("\nSaved files:")
    print(output_dir / "ranking_all.csv")
    print(output_dir / "ranking_all.json")
    print(output_dir / "ranking_direct.csv")
    print(output_dir / "ranking_plus_newton.csv")
    print(output_dir / "ranking_summary.json")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import math


# =========================
# 기본 설정
# =========================
# 클수록 좋은 지표
HIGHER_IS_BETTER = {
    "direct_r2",
    "direct_valid_ratio",
    "plus_newton_r2",
    "plus_newton_valid_ratio",
    "plus_newton_converged_ratio",
}

# 작을수록 좋은 지표
LOWER_IS_BETTER = {
    "direct_mae",
    "direct_rmse",
    "direct_residual_mean",
    "direct_residual_median",
    "direct_residual_p90",
    "plus_newton_mae",
    "plus_newton_rmse",
    "plus_newton_residual_mean",
    "plus_newton_residual_median",
    "plus_newton_residual_p90",
    "plus_newton_newton_iter_mean",
    "plus_newton_newton_iter_median",
    "plus_newton_newton_iter_p90",
    "elapsed_sec",
}

# direct 쪽 가중치
DEFAULT_DIRECT_WEIGHTS = {
    "direct_r2": 0.30,
    "direct_rmse": 0.25,
    "direct_mae": 0.10,
    "direct_residual_mean": 0.15,
    "direct_residual_median": 0.08,
    "direct_residual_p90": 0.10,
    "direct_valid_ratio": 0.02,
}

# plus_newton 쪽 가중치
DEFAULT_PLUS_WEIGHTS = {
    "plus_newton_r2": 0.20,
    "plus_newton_rmse": 0.20,
    "plus_newton_mae": 0.08,
    "plus_newton_residual_mean": 0.10,
    "plus_newton_residual_median": 0.06,
    "plus_newton_residual_p90": 0.10,
    "plus_newton_newton_iter_mean": 0.14,
    "plus_newton_newton_iter_median": 0.04,
    "plus_newton_newton_iter_p90": 0.05,
    "plus_newton_valid_ratio": 0.01,
    "plus_newton_converged_ratio": 0.02,
}

# overall = direct + plus_newton 혼합
DEFAULT_OVERALL_MIX = {
    "direct_score": 0.40,
    "plus_score": 0.60,
}


# =========================
# 유틸
# =========================
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def load_trial_jsons(input_dir: Path) -> List[Dict]:
    rows = []
    for p in sorted(input_dir.glob("trial_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["_json_path"] = str(p)
            rows.append(data)
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}")
    return rows


def collect_metric_values(rows: List[Dict], metric_names: List[str]) -> Dict[str, List[float]]:
    out = {m: [] for m in metric_names}
    for row in rows:
        for m in metric_names:
            val = safe_float(row.get(m, math.nan))
            out[m].append(val)
    return out


def min_max_score(value: float, vmin: float, vmax: float, higher_is_better: bool) -> float:
    if not math.isfinite(value):
        return 0.0
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return 0.0
    if abs(vmax - vmin) < 1e-15:
        # 전부 동일하면 구분력 없음
        return 0.0
    if higher_is_better:
        return (value - vmin) / (vmax - vmin)
    return (vmax - value) / (vmax - vmin)


def build_metric_stats(rows: List[Dict], metric_names: List[str]) -> Dict[str, Tuple[float, float]]:
    stats = {}
    for m in metric_names:
        vals = [safe_float(r.get(m, math.nan)) for r in rows]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            stats[m] = (math.nan, math.nan)
        else:
            stats[m] = (min(vals), max(vals))
    return stats


def weighted_score(row: Dict, weights: Dict[str, float], stats: Dict[str, Tuple[float, float]]) -> Tuple[float, Dict[str, float]]:
    score = 0.0
    contrib = {}

    for metric, weight in weights.items():
        val = safe_float(row.get(metric, math.nan))
        vmin, vmax = stats[metric]

        if metric in HIGHER_IS_BETTER:
            s = min_max_score(val, vmin, vmax, higher_is_better=True)
        elif metric in LOWER_IS_BETTER:
            s = min_max_score(val, vmin, vmax, higher_is_better=False)
        else:
            # 분류되지 않은 지표는 일단 무시
            s = 0.0

        contrib[metric] = s * weight
        score += contrib[metric]

    return score, contrib


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(weights.values())
    if s <= 0:
        raise ValueError("weights sum must be > 0")
    return {k: v / s for k, v in weights.items()}


def save_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    keys = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# =========================
# 메인 랭킹
# =========================
def rank_trials(
    rows: List[Dict],
    direct_weights: Dict[str, float],
    plus_weights: Dict[str, float],
    overall_mix: Dict[str, float],
) -> List[Dict]:
    direct_weights = normalize_weights(direct_weights)
    plus_weights = normalize_weights(plus_weights)
    overall_mix = normalize_weights(overall_mix)

    all_metric_names = sorted(set(direct_weights.keys()) | set(plus_weights.keys()))
    stats = build_metric_stats(rows, all_metric_names)

    ranked = []
    for row in rows:
        direct_score, direct_detail = weighted_score(row, direct_weights, stats)
        plus_score, plus_detail = weighted_score(row, plus_weights, stats)

        overall_score = (
            overall_mix["direct_score"] * direct_score
            + overall_mix["plus_score"] * plus_score
        )

        out = dict(row)
        out["direct_score"] = direct_score
        out["plus_score"] = plus_score
        out["overall_score"] = overall_score

        for k, v in direct_detail.items():
            out[f"contrib_direct__{k}"] = v
        for k, v in plus_detail.items():
            out[f"contrib_plus__{k}"] = v

        ranked.append(out)

    ranked.sort(key=lambda x: x["overall_score"], reverse=True)

    for i, row in enumerate(ranked, start=1):
        row["overall_rank"] = i

    # direct 기준 순위
    direct_sorted = sorted(ranked, key=lambda x: x["direct_score"], reverse=True)
    for i, row in enumerate(direct_sorted, start=1):
        row["direct_rank"] = i

    # plus 기준 순위
    plus_sorted = sorted(ranked, key=lambda x: x["plus_score"], reverse=True)
    for i, row in enumerate(plus_sorted, start=1):
        row["plus_rank"] = i

    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="trial_*.json 파일들이 들어있는 폴더",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="랭킹 결과 저장 폴더",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="콘솔에 출력할 상위 개수",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_trial_jsons(input_dir)
    if not rows:
        raise RuntimeError(f"No trial_*.json found in {input_dir}")

    ranked = rank_trials(
        rows=rows,
        direct_weights=DEFAULT_DIRECT_WEIGHTS,
        plus_weights=DEFAULT_PLUS_WEIGHTS,
        overall_mix=DEFAULT_OVERALL_MIX,
    )

    # overall 저장
    save_csv(output_dir / "ranking_all.csv", ranked)
    (output_dir / "ranking_all.json").write_text(
        json.dumps(ranked, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # direct / plus 별도 저장
    direct_sorted = sorted(ranked, key=lambda x: x["direct_score"], reverse=True)
    plus_sorted = sorted(ranked, key=lambda x: x["plus_score"], reverse=True)

    save_csv(output_dir / "ranking_direct.csv", direct_sorted)
    save_csv(output_dir / "ranking_plus_newton.csv", plus_sorted)

    summary = {
        "num_trials": len(ranked),
        "direct_weights": DEFAULT_DIRECT_WEIGHTS,
        "plus_weights": DEFAULT_PLUS_WEIGHTS,
        "overall_mix": DEFAULT_OVERALL_MIX,
        "best_overall": {
            "trial_id": ranked[0]["trial_id"],
            "trial_name": ranked[0]["trial_name"],
            "model": ranked[0]["model"],
            "overall_score": ranked[0]["overall_score"],
            "direct_score": ranked[0]["direct_score"],
            "plus_score": ranked[0]["plus_score"],
        },
    }
    (output_dir / "ranking_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("=== TOP OVERALL ===")
    for row in ranked[:args.topk]:
        print({
            "overall_rank": row["overall_rank"],
            "trial_id": row["trial_id"],
            "model": row["model"],
            "overall_score": row["overall_score"],
            "direct_score": row["direct_score"],
            "plus_score": row["plus_score"],
            "direct_rmse": row.get("direct_rmse"),
            "plus_newton_rmse": row.get("plus_newton_rmse"),
            "plus_newton_newton_iter_mean": row.get("plus_newton_newton_iter_mean"),
        })

    print("\nSaved files:")
    print(output_dir / "ranking_all.csv")
    print(output_dir / "ranking_all.json")
    print(output_dir / "ranking_direct.csv")
    print(output_dir / "ranking_plus_newton.csv")
    print(output_dir / "ranking_summary.json")


if __name__ == "__main__":
    main()