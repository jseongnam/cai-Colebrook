import json
import math
from pathlib import Path

import pandas as pd


# =========================
# 설정
# =========================

# trial_001_gru.json 등이 들어 있는 폴더
RESULT_DIR = Path("grid_runs_1d/deg25_search_mlp")

# 모델 순서
MODEL_ORDER = ["mlp", "lstm", "gru", "transformer", "baseline"]

# 각 모델별 최고 trial 선정 기준
# direct initialization 표를 채우는 목적이면 direct_rmse 최소 추천
BEST_SORT_KEY = "direct_rmse"


# =========================
# 유틸 함수
# =========================

def safe_float(x):
    if x is None:
        return math.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def fmt(x, digits=6):
    """논문 표에 넣기 좋은 숫자 포맷."""
    x = safe_float(x)
    if math.isnan(x):
        return ""

    if x != 0 and abs(x) < 1e-4:
        return f"{x:.4e}"

    return f"{x:.{digits}f}"


# =========================
# JSON 읽기
# =========================

rows = []

for path in sorted(RESULT_DIR.glob("trial_*.json")):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    d["file"] = path.name
    d["model"] = str(d.get("model", "")).lower()
    rows.append(d)

if not rows:
    raise FileNotFoundError(f"No trial_*.json files found in {RESULT_DIR.resolve()}")

df = pd.DataFrame(rows)

# 숫자 컬럼 변환
numeric_cols = [
    "direct_mae",
    "direct_rmse",
    "direct_r2",
    "direct_residual_mean",
    "plus_newton_rmse",
    "plus_newton_newton_iter_mean",
    "plus_newton_newton_iter_p90",
    "plus_newton_converged_ratio",
]

for c in numeric_cols:
    if c in df.columns:
        df[c] = df[c].apply(safe_float)


# =========================
# 모델별 최고 trial 선택
# =========================

best_rows = []

for model in MODEL_ORDER:
    sub = df[df["model"] == model].copy()

    if sub.empty:
        print(f"[Warning] No trials found for model: {model}")
        continue

    sub = sub.sort_values(
        by=[BEST_SORT_KEY, "plus_newton_newton_iter_mean"],
        ascending=[True, True],
        na_position="last"
    )

    best_rows.append(sub.iloc[0])

best = pd.DataFrame(best_rows)


# =========================
# 표 I: Direct Initialization
# =========================

table1 = pd.DataFrame({
    "Model": best["model"].map({
        "mlp": "Mlp",
        "lstm": "Lstm",
        "gru": "Gru",
        "transformer": "Transformer",
        "baseline": "baseline",
    }),
    "MAE": best["direct_mae"].map(fmt),
    "RMSE": best["direct_rmse"].map(fmt),
    "R²": best["direct_r2"].map(fmt),
    "Residual Mean": best["direct_residual_mean"].map(fmt),
})


# =========================
# 표 II: Newton Refinement
# =========================

table2 = pd.DataFrame({
    "Model": best["model"].map({
        "mlp": "Mlp",
        "lstm": "Lstm",
        "gru": "Gru",
        "transformer": "transformer",
        "baseline": "baseline",
    }),
    "RMSE": best["plus_newton_rmse"].map(fmt),
    "Iter Mean": best["plus_newton_newton_iter_mean"].map(fmt),
    "Iter p90": best["plus_newton_newton_iter_p90"].map(fmt),
    "Converged Ratio": best["plus_newton_converged_ratio"].map(fmt),
})


# =========================
# 저장
# =========================

table1.to_csv("table1_direct_initialization.csv", index=False, encoding="utf-8-sig")
table2.to_csv("table2_newton_refinement.csv", index=False, encoding="utf-8-sig")

table1.to_excel("table1_direct_initialization.xlsx", index=False)
table2.to_excel("table2_newton_refinement.xlsx", index=False)

# Word에 복붙하기 쉬운 탭 구분 텍스트
table1.to_csv("table1_direct_initialization_tsv.txt", index=False, sep="\t")
table2.to_csv("table2_newton_refinement_tsv.txt", index=False, sep="\t")

# LaTeX용
table1.to_latex(
    "table1_direct_initialization.tex",
    index=False,
    escape=False,
    column_format="lcccc"
)

table2.to_latex(
    "table2_newton_refinement.tex",
    index=False,
    escape=False,
    column_format="lcccc"
)


# =========================
# 화면 출력
# =========================

print("\nTABLE I. Direct Initialization")
print(table1.to_string(index=False))

print("\nTABLE II. Newton Refinement")
print(table2.to_string(index=False))

print("\nSelected best trials:")
print(
    best[
        [
            "model",
            "trial_name",
            "best_epoch",
            "direct_rmse",
            "plus_newton_rmse",
            "plus_newton_newton_iter_mean",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print("- table1_direct_initialization.csv")
print("- table2_newton_refinement.csv")
print("- table1_direct_initialization.xlsx")
print("- table2_newton_refinement.xlsx")
print("- table1_direct_initialization_tsv.txt")
print("- table2_newton_refinement_tsv.txt")
print("- table1_direct_initialization.tex")
print("- table2_newton_refinement.tex")