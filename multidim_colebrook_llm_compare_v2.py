#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multidim_colebrook_llm_compare_v2.py

다중 Colebrook-type 결합 시스템 전용 LLM 비교실험 스크립트

지원 방식
1) openrouter
   - GPT-5.1, claude-sonnet-4-20250514 같은 모델명을 그대로 사용 가능
2) openai_compatible
   - vLLM / OpenAI-compatible endpoint
3) manual_jsonl
   - 외부에서 받은 응답(JSONL)만 읽어서 평가
4) export_prompts
   - 프롬프트만 CSV/JSONL로 저장

비교 대상
- heuristic_direct
- heuristic_plus_newton
- llm_direct
- llm_plus_newton

LLM 출력 형식(JSON)
{"Q1": <float>, "x1": <float>, "x2": <float>}
"""

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import requests
except Exception:
    requests = None

PI = math.pi
LN10 = math.log(10.0)


# =========================================================
# Utility
# =========================================================
def sanitize_array(x: np.ndarray, clip_value: float = 1e12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x


def percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def save_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    keys = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def vector_metrics(pred, true):
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mae_Q1": float(np.mean(np.abs(err[:, 0]))),
        "mae_x1": float(np.mean(np.abs(err[:, 1]))),
        "mae_x2": float(np.mean(np.abs(err[:, 2]))),
        "max_abs_error": float(np.max(np.abs(err))),
    }


# =========================================================
# Physics / Newton
# =========================================================
def re_from_Q(Q, rho, mu, D):
    return 4.0 * rho * Q / (PI * mu * D)


def colebrook_single_x_eq(x, Re, rel_rough):
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    rel_rough = np.asarray(rel_rough, dtype=np.float64)
    z = rel_rough / 3.7 + 2.51 * x / Re
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = (Re > 0) & (z > 0)
    out[mask] = x[mask] + 2.0 * np.log10(z[mask])
    return out


def colebrook_single_x_df(x, Re, rel_rough):
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    rel_rough = np.asarray(rel_rough, dtype=np.float64)
    z = rel_rough / 3.7 + 2.51 * x / Re
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = (Re > 0) & (z > 0)
    out[mask] = 1.0 + 2.0 * ((2.51 / Re[mask]) / (z[mask] * LN10))
    return out


def solve_x_from_Q(Q, D, eps, rho, mu, x_init=7.0, tol=1e-13, max_iter=50):
    Re = re_from_Q(np.array([Q]), np.array([rho]), np.array([mu]), np.array([D]))[0]
    rr = eps / D
    x = float(x_init)
    for _ in range(max_iter):
        fx = colebrook_single_x_eq(np.array([x]), np.array([Re]), np.array([rr]))[0]
        dfx = colebrook_single_x_df(np.array([x]), np.array([Re]), np.array([rr]))[0]
        if (not np.isfinite(fx)) or (not np.isfinite(dfx)) or abs(dfx) < 1e-15:
            break
        x_new = float(np.clip(x - fx / dfx, 1e-3, 1e3))
        if abs(x_new - x) < tol and abs(fx) < tol:
            x = x_new
            break
        x = x_new
    return float(x)


def head_loss(Q, x, L, D, g):
    return 8.0 * L * (Q ** 2) / (g * (PI ** 2) * (D ** 5) * (x ** 2))


def system_F(z, params):
    Q1, x1, x2 = z[..., 0], z[..., 1], z[..., 2]
    QT = params["Q_total"]
    D1 = params["D1"]
    D2 = params["D2"]
    eps1 = params["eps1"]
    eps2 = params["eps2"]
    L1 = params["L1"]
    L2 = params["L2"]
    rho = params["rho"]
    mu = params["mu"]
    g = params["g"]

    Q2 = QT - Q1
    Re1 = re_from_Q(Q1, rho, mu, D1)
    Re2 = re_from_Q(Q2, rho, mu, D2)
    rr1 = eps1 / D1
    rr2 = eps2 / D2

    F1 = colebrook_single_x_eq(x1, Re1, rr1)
    F2 = colebrook_single_x_eq(x2, Re2, rr2)
    F3 = head_loss(Q1, x1, L1, D1, g) - head_loss(Q2, x2, L2, D2, g)
    return np.stack([F1, F2, F3], axis=-1)


def numerical_jacobian_single(z, p, eps=1e-6):
    z = np.asarray(z, dtype=np.float64)
    J = np.zeros((3, 3), dtype=np.float64)
    for j in range(3):
        zp = z.copy()
        zm = z.copy()
        step = eps * max(1.0, abs(z[j]))
        zp[j] += step
        zm[j] -= step
        fp = system_F(zp[None, :], p)[0]
        fm = system_F(zm[None, :], p)[0]
        J[:, j] = (fp - fm) / (2.0 * step)
    return J


def project_feasible(z, p):
    z = np.asarray(z, dtype=np.float64).copy()
    QT = float(p["Q_total"])
    z[0] = np.clip(z[0], max(1e-8, QT * 1e-5), QT - max(1e-8, QT * 1e-5))
    z[1] = max(z[1], 1e-3)
    z[2] = max(z[2], 1e-3)
    return z


def newton_system_single(z0, p, tol=1e-12, max_iter=20, damping=1.0):
    z = project_feasible(z0, p)
    converged = False
    used_iter = 0
    for k in range(1, max_iter + 1):
        J = numerical_jacobian_single(z, p)
        f = system_F(z[None, :], p)[0]
        if not np.all(np.isfinite(J)) or not np.all(np.isfinite(f)):
            break
        try:
            step = np.linalg.solve(J, f)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -5.0, 5.0)
        z_new = project_feasible(z - damping * step, p)
        f_new = system_F(z_new[None, :], p)[0]
        if np.linalg.norm(f_new, ord=2) > np.linalg.norm(f, ord=2):
            z_half = project_feasible(z - 0.5 * damping * step, p)
            f_half = system_F(z_half[None, :], p)[0]
            if np.linalg.norm(f_half, ord=2) < np.linalg.norm(f_new, ord=2):
                z_new = z_half
                f_new = f_half
        z = z_new
        used_iter = k
        if np.linalg.norm(f_new, ord=np.inf) <= tol:
            converged = True
            break
    return z, used_iter, converged


def refine_batch(z_init, data, tol=1e-12, max_iter=20):
    n = len(z_init)
    out = np.zeros_like(z_init, dtype=np.float64)
    iters = np.zeros(n, dtype=np.int32)
    conv = np.zeros(n, dtype=bool)
    for i in range(n):
        p = {k: float(np.asarray(data[k])[i]) for k in ["Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]}
        zf, it, ok = newton_system_single(z_init[i], p, tol=tol, max_iter=max_iter)
        out[i] = zf
        iters[i] = it
        conv[i] = ok
    return out, iters, conv


# =========================================================
# Data and baselines
# =========================================================
def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    required = ["coeffs", "center", "target", "Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {npz_path}. Available: {list(data.keys())}")
    return {k: np.asarray(data[k]) for k in required}


def build_heuristic_baseline(data: Dict[str, np.ndarray]) -> np.ndarray:
    QT = np.asarray(data["Q_total"], dtype=np.float64)
    D1 = np.asarray(data["D1"], dtype=np.float64)
    D2 = np.asarray(data["D2"], dtype=np.float64)
    eps1 = np.asarray(data["eps1"], dtype=np.float64)
    eps2 = np.asarray(data["eps2"], dtype=np.float64)
    rho = np.asarray(data["rho"], dtype=np.float64)
    mu = np.asarray(data["mu"], dtype=np.float64)

    n = len(QT)
    z0 = np.zeros((n, 3), dtype=np.float64)
    z0[:, 0] = QT / 2.0
    for i in range(n):
        qh = QT[i] / 2.0
        z0[i, 1] = solve_x_from_Q(qh, D1[i], eps1[i], rho[i], mu[i])
        z0[i, 2] = solve_x_from_Q(qh, D2[i], eps2[i], rho[i], mu[i])
    return z0


def residual_metrics(pred, data):
    params = {k: np.asarray(data[k], dtype=np.float64) for k in ["Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]}
    F = system_F(pred.astype(np.float64), params)
    norms_inf = np.max(np.abs(F), axis=1)
    valid = np.all(np.isfinite(F), axis=1)
    return {
        "valid_ratio": float(np.mean(valid)),
        "residual_mean": float(np.nanmean(norms_inf)),
        "residual_median": float(np.nanmedian(norms_inf)),
        "residual_p90": percentile(norms_inf[np.isfinite(norms_inf)], 90),
    }


# =========================================================
# Prompt and parsing
# =========================================================
SYSTEM_PROMPT = """You are solving a coupled nonlinear Colebrook-type system.
Return ONLY a JSON object with keys Q1, x1, x2.
No explanation. No markdown. No prose.
The result must satisfy:
1) branch-1 Colebrook equation
2) branch-2 Colebrook equation
3) equal head-loss coupling
4) 0 < Q1 < Q_total and x1 > 0 and x2 > 0
"""

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def format_sample_prompt(data: Dict[str, np.ndarray], idx: int, include_taylor: bool = True, coeff_round: int = 8) -> str:
    QT = float(data["Q_total"][idx])
    D1 = float(data["D1"][idx]); D2 = float(data["D2"][idx])
    eps1 = float(data["eps1"][idx]); eps2 = float(data["eps2"][idx])
    L1 = float(data["L1"][idx]); L2 = float(data["L2"][idx])
    rho = float(data["rho"][idx]); mu = float(data["mu"][idx]); g = float(data["g"][idx])
    center = np.asarray(data["center"][idx], dtype=np.float64).tolist()
    coeffs = np.asarray(data["coeffs"][idx], dtype=np.float64)

    parts = []
    parts.append("Solve the following coupled Colebrook-type nonlinear system.")
    parts.append(f"Q_total = {QT:.12g}")
    parts.append(f"D1 = {D1:.12g}, D2 = {D2:.12g}")
    parts.append(f"eps1 = {eps1:.12g}, eps2 = {eps2:.12g}")
    parts.append(f"L1 = {L1:.12g}, L2 = {L2:.12g}")
    parts.append(f"rho = {rho:.12g}, mu = {mu:.12g}, g = {g:.12g}")
    parts.append("")
    parts.append("Unknowns: Q1, x1, x2")
    parts.append("Q2 = Q_total - Q1")
    parts.append("")
    parts.append("Definitions:")
    parts.append("Re1 = 4*rho*Q1 / (pi*mu*D1)")
    parts.append("Re2 = 4*rho*Q2 / (pi*mu*D2)")
    parts.append("rr1 = eps1 / D1")
    parts.append("rr2 = eps2 / D2")
    parts.append("")
    parts.append("Equations:")
    parts.append("1) x1 + 2*log10(rr1/3.7 + 2.51*x1/Re1) = 0")
    parts.append("2) x2 + 2*log10(rr2/3.7 + 2.51*x2/Re2) = 0")
    parts.append("3) 8*L1*Q1^2 / (g*pi^2*D1^5*x1^2) = 8*L2*Q2^2 / (g*pi^2*D2^5*x2^2)")
    parts.append("")
    parts.append("Constraints:")
    parts.append("0 < Q1 < Q_total")
    parts.append("x1 > 0")
    parts.append("x2 > 0")
    parts.append("")
    if include_taylor:
        parts.append(f"Taylor centers = {center}")
        parts.append("Flattened Taylor coefficients:")
        parts.append(json.dumps([round(float(v), coeff_round) for v in coeffs.reshape(-1)], ensure_ascii=False))
        parts.append("")
    parts.append('Return ONLY JSON: {"Q1": ..., "x1": ..., "x2": ...}')
    return "\n".join(parts)


def parse_llm_json(text: str) -> Optional[Dict[str, float]]:
    if text is None:
        return None
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    for k in ["Q1", "x1", "x2"]:
        if k not in obj:
            return None
        try:
            out[k] = float(obj[k])
        except Exception:
            return None
    return out


# =========================================================
# Providers
# =========================================================
def call_openai_compatible(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 200,
    timeout: int = 120,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    if requests is None:
        raise RuntimeError("requests 패키지가 필요합니다.")
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if extra_headers:
        headers.update(extra_headers)
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    t0 = time.time()
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    latency = time.time() - t0
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    usage["latency_sec"] = latency
    return text, usage


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 200,
    timeout: int = 120,
    referer: str = "",
    title: str = "",
) -> Tuple[str, Dict[str, Any]]:
    headers = {}
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return call_openai_compatible(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_headers=headers,
    )


def load_manual_jsonl(manual_jsonl: Path) -> Dict[int, Dict[str, Any]]:
    mapping = {}
    with open(manual_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            mapping[int(obj["index"])] = obj
    return mapping


# =========================================================
# Main experiment
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--provider", choices=["openrouter", "openai_compatible", "manual_jsonl", "export_prompts"], required=True)

    # provider args
    parser.add_argument("--api_key", default="")
    parser.add_argument("--base_url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--manual_jsonl", default="")
    parser.add_argument("--openrouter_referer", default="")
    parser.add_argument("--openrouter_title", default="")

    # experiment control
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--include_taylor", action="store_true")
    parser.add_argument("--coeff_round", type=int, default=8)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.test_npz)
    y_true = np.asarray(data["target"], dtype=np.float64)
    heur_pred = build_heuristic_baseline(data)

    n_total = len(y_true)
    idxs = list(range(args.start_index, n_total))
    if args.max_samples > 0:
        idxs = idxs[: args.max_samples]

    # export prompts only
    prompt_rows = []
    for idx in idxs:
        prompt_rows.append({
            "index": idx,
            "prompt": format_sample_prompt(data, idx, include_taylor=args.include_taylor, coeff_round=args.coeff_round),
        })

    if args.provider == "export_prompts":
        save_csv(out_dir / "prompts.csv", prompt_rows)
        save_jsonl(out_dir / "prompts.jsonl", prompt_rows)
        print(f"[DONE] prompts exported to {out_dir}")
        print("  - prompts.csv")
        print("  - prompts.jsonl")
        return

    manual_map = None
    if args.provider == "manual_jsonl":
        if not args.manual_jsonl:
            raise ValueError("--manual_jsonl 필요")
        manual_map = load_manual_jsonl(Path(args.manual_jsonl))

    llm_preds = []
    resp_rows = []

    for i, idx in enumerate(idxs, start=1):
        prompt = prompt_rows[i - 1]["prompt"]
        raw_text = ""
        usage = {}
        parsed = None
        err_msg = ""

        try:
            if args.provider == "openrouter":
                raw_text, usage = call_openrouter(
                    api_key=args.api_key,
                    model=args.model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    referer=args.openrouter_referer,
                    title=args.openrouter_title,
                )
            elif args.provider == "openai_compatible":
                raw_text, usage = call_openai_compatible(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
            else:
                item = manual_map.get(idx)
                if item is None:
                    raise KeyError(f"manual_jsonl에 index={idx} 응답이 없습니다.")
                raw_text = item.get("response_text", "")
                usage = item.get("usage", {}) if isinstance(item.get("usage", {}), dict) else {}

            parsed = parse_llm_json(raw_text)
            if parsed is None:
                err_msg = "JSON parse failed"
        except Exception as e:
            err_msg = str(e)

        if parsed is None:
            pred_vec = heur_pred[idx].astype(np.float64)  # fallback
            parsed_for_save = None
        else:
            pred_vec = np.array([parsed["Q1"], parsed["x1"], parsed["x2"]], dtype=np.float64)
            p = {k: float(np.asarray(data[k])[idx]) for k in ["Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]}
            pred_vec = project_feasible(pred_vec, p)
            parsed_for_save = parsed

        llm_preds.append(pred_vec)
        resp_rows.append({
            "index": idx,
            "ok": parsed_for_save is not None,
            "error": err_msg,
            "response_text": raw_text,
            "parsed_json": json.dumps(parsed_for_save, ensure_ascii=False) if parsed_for_save is not None else "",
            "latency_sec": usage.get("latency_sec", ""),
            "prompt_tokens": usage.get("prompt_tokens", ""),
            "completion_tokens": usage.get("completion_tokens", ""),
            "total_tokens": usage.get("total_tokens", ""),
        })
        print(f"[{i}/{len(idxs)}] index={idx} ok={parsed_for_save is not None}")

    llm_pred = np.stack(llm_preds, axis=0)
    y_sub = y_true[idxs]
    heur_sub = heur_pred[idxs]
    data_sub = {k: np.asarray(v)[idxs] for k, v in data.items()}

    rows = []

    hd = vector_metrics(heur_sub, y_sub)
    hd.update(residual_metrics(heur_sub, data_sub))
    hd["name"] = "heuristic_direct"
    rows.append(hd)

    href, hit, hconv = refine_batch(heur_sub.astype(np.float64), data_sub, tol=args.tol, max_iter=args.max_newton_iter)
    hr = vector_metrics(href, y_sub.astype(np.float64))
    hr.update(residual_metrics(href, data_sub))
    hr["name"] = "heuristic_plus_newton"
    hr["newton_iter_mean"] = float(np.mean(hit))
    hr["newton_iter_median"] = float(np.median(hit))
    hr["newton_iter_p90"] = float(np.percentile(hit, 90))
    hr["newton_converged_ratio"] = float(np.mean(hconv))
    rows.append(hr)

    ld = vector_metrics(llm_pred, y_sub)
    ld.update(residual_metrics(llm_pred, data_sub))
    ld["name"] = "llm_direct"
    rows.append(ld)

    lref, lit, lconv = refine_batch(llm_pred.astype(np.float64), data_sub, tol=args.tol, max_iter=args.max_newton_iter)
    lr = vector_metrics(lref, y_sub.astype(np.float64))
    lr.update(residual_metrics(lref, data_sub))
    lr["name"] = "llm_plus_newton"
    lr["newton_iter_mean"] = float(np.mean(lit))
    lr["newton_iter_median"] = float(np.median(lit))
    lr["newton_iter_p90"] = float(np.percentile(lit, 90))
    lr["newton_converged_ratio"] = float(np.mean(lconv))
    rows.append(lr)

    save_csv(out_dir / "summary_metrics.csv", rows)
    save_csv(out_dir / "responses.csv", resp_rows)
    save_csv(out_dir / "prompts.csv", prompt_rows)

    per_rows = []
    for j, idx in enumerate(idxs):
        per_rows.append({
            "index": idx,
            "true_Q1": float(y_sub[j, 0]),
            "true_x1": float(y_sub[j, 1]),
            "true_x2": float(y_sub[j, 2]),
            "heur_Q1": float(heur_sub[j, 0]),
            "heur_x1": float(heur_sub[j, 1]),
            "heur_x2": float(heur_sub[j, 2]),
            "llm_Q1": float(llm_pred[j, 0]),
            "llm_x1": float(llm_pred[j, 1]),
            "llm_x2": float(llm_pred[j, 2]),
            "llm_ref_Q1": float(lref[j, 0]),
            "llm_ref_x1": float(lref[j, 1]),
            "llm_ref_x2": float(lref[j, 2]),
            "llm_iter": int(lit[j]),
            "llm_converged": bool(lconv[j]),
        })
    save_csv(out_dir / "per_sample_results.csv", per_rows)

    config = vars(args).copy()
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for row in rows:
        print(row)
    print(f"\n[DONE] saved to: {out_dir}")
    print("  - summary_metrics.csv")
    print("  - per_sample_results.csv")
    print("  - responses.csv")
    print("  - prompts.csv")
    print("  - config.json")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multidim_colebrook_llm_compare_v2.py

다중 Colebrook-type 결합 시스템 전용 LLM 비교실험 스크립트

지원 방식
1) openrouter
   - GPT-5.1, claude-sonnet-4-20250514 같은 모델명을 그대로 사용 가능
2) openai_compatible
   - vLLM / OpenAI-compatible endpoint
3) manual_jsonl
   - 외부에서 받은 응답(JSONL)만 읽어서 평가
4) export_prompts
   - 프롬프트만 CSV/JSONL로 저장

비교 대상
- heuristic_direct
- heuristic_plus_newton
- llm_direct
- llm_plus_newton

LLM 출력 형식(JSON)
{"Q1": <float>, "x1": <float>, "x2": <float>}
"""

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import requests
except Exception:
    requests = None

PI = math.pi
LN10 = math.log(10.0)


# =========================================================
# Utility
# =========================================================
def sanitize_array(x: np.ndarray, clip_value: float = 1e12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x


def percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def save_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    keys = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def vector_metrics(pred, true):
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mae_Q1": float(np.mean(np.abs(err[:, 0]))),
        "mae_x1": float(np.mean(np.abs(err[:, 1]))),
        "mae_x2": float(np.mean(np.abs(err[:, 2]))),
        "max_abs_error": float(np.max(np.abs(err))),
    }


# =========================================================
# Physics / Newton
# =========================================================
def re_from_Q(Q, rho, mu, D):
    return 4.0 * rho * Q / (PI * mu * D)


def colebrook_single_x_eq(x, Re, rel_rough):
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    rel_rough = np.asarray(rel_rough, dtype=np.float64)
    z = rel_rough / 3.7 + 2.51 * x / Re
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = (Re > 0) & (z > 0)
    out[mask] = x[mask] + 2.0 * np.log10(z[mask])
    return out


def colebrook_single_x_df(x, Re, rel_rough):
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    rel_rough = np.asarray(rel_rough, dtype=np.float64)
    z = rel_rough / 3.7 + 2.51 * x / Re
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = (Re > 0) & (z > 0)
    out[mask] = 1.0 + 2.0 * ((2.51 / Re[mask]) / (z[mask] * LN10))
    return out


def solve_x_from_Q(Q, D, eps, rho, mu, x_init=7.0, tol=1e-13, max_iter=50):
    Re = re_from_Q(np.array([Q]), np.array([rho]), np.array([mu]), np.array([D]))[0]
    rr = eps / D
    x = float(x_init)
    for _ in range(max_iter):
        fx = colebrook_single_x_eq(np.array([x]), np.array([Re]), np.array([rr]))[0]
        dfx = colebrook_single_x_df(np.array([x]), np.array([Re]), np.array([rr]))[0]
        if (not np.isfinite(fx)) or (not np.isfinite(dfx)) or abs(dfx) < 1e-15:
            break
        x_new = float(np.clip(x - fx / dfx, 1e-3, 1e3))
        if abs(x_new - x) < tol and abs(fx) < tol:
            x = x_new
            break
        x = x_new
    return float(x)


def head_loss(Q, x, L, D, g):
    return 8.0 * L * (Q ** 2) / (g * (PI ** 2) * (D ** 5) * (x ** 2))


def system_F(z, params):
    Q1, x1, x2 = z[..., 0], z[..., 1], z[..., 2]
    QT = params["Q_total"]
    D1 = params["D1"]
    D2 = params["D2"]
    eps1 = params["eps1"]
    eps2 = params["eps2"]
    L1 = params["L1"]
    L2 = params["L2"]
    rho = params["rho"]
    mu = params["mu"]
    g = params["g"]

    Q2 = QT - Q1
    Re1 = re_from_Q(Q1, rho, mu, D1)
    Re2 = re_from_Q(Q2, rho, mu, D2)
    rr1 = eps1 / D1
    rr2 = eps2 / D2

    F1 = colebrook_single_x_eq(x1, Re1, rr1)
    F2 = colebrook_single_x_eq(x2, Re2, rr2)
    F3 = head_loss(Q1, x1, L1, D1, g) - head_loss(Q2, x2, L2, D2, g)
    return np.stack([F1, F2, F3], axis=-1)


def numerical_jacobian_single(z, p, eps=1e-6):
    z = np.asarray(z, dtype=np.float64)
    J = np.zeros((3, 3), dtype=np.float64)
    for j in range(3):
        zp = z.copy()
        zm = z.copy()
        step = eps * max(1.0, abs(z[j]))
        zp[j] += step
        zm[j] -= step
        fp = system_F(zp[None, :], p)[0]
        fm = system_F(zm[None, :], p)[0]
        J[:, j] = (fp - fm) / (2.0 * step)
    return J


def project_feasible(z, p):
    z = np.asarray(z, dtype=np.float64).copy()
    QT = float(p["Q_total"])
    z[0] = np.clip(z[0], max(1e-8, QT * 1e-5), QT - max(1e-8, QT * 1e-5))
    z[1] = max(z[1], 1e-3)
    z[2] = max(z[2], 1e-3)
    return z


def newton_system_single(z0, p, tol=1e-12, max_iter=20, damping=1.0):
    z = project_feasible(z0, p)
    converged = False
    used_iter = 0
    for k in range(1, max_iter + 1):
        J = numerical_jacobian_single(z, p)
        f = system_F(z[None, :], p)[0]
        if not np.all(np.isfinite(J)) or not np.all(np.isfinite(f)):
            break
        try:
            step = np.linalg.solve(J, f)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -5.0, 5.0)
        z_new = project_feasible(z - damping * step, p)
        f_new = system_F(z_new[None, :], p)[0]
        if np.linalg.norm(f_new, ord=2) > np.linalg.norm(f, ord=2):
            z_half = project_feasible(z - 0.5 * damping * step, p)
            f_half = system_F(z_half[None, :], p)[0]
            if np.linalg.norm(f_half, ord=2) < np.linalg.norm(f_new, ord=2):
                z_new = z_half
                f_new = f_half
        z = z_new
        used_iter = k
        if np.linalg.norm(f_new, ord=np.inf) <= tol:
            converged = True
            break
    return z, used_iter, converged


def refine_batch(z_init, data, tol=1e-12, max_iter=20):
    n = len(z_init)
    out = np.zeros_like(z_init, dtype=np.float64)
    iters = np.zeros(n, dtype=np.int32)
    conv = np.zeros(n, dtype=bool)
    for i in range(n):
        p = {k: float(np.asarray(data[k])[i]) for k in ["Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]}
        zf, it, ok = newton_system_single(z_init[i], p, tol=tol, max_iter=max_iter)
        out[i] = zf
        iters[i] = it
        conv[i] = ok
    return out, iters, conv


# =========================================================
# Data and baselines
# =========================================================
def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    required = ["coeffs", "center", "target", "Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {npz_path}. Available: {list(data.keys())}")
    return {k: np.asarray(data[k]) for k in required}


def build_heuristic_baseline(data: Dict[str, np.ndarray]) -> np.ndarray:
    QT = np.asarray(data["Q_total"], dtype=np.float64)
    D1 = np.asarray(data["D1"], dtype=np.float64)
    D2 = np.asarray(data["D2"], dtype=np.float64)
    eps1 = np.asarray(data["eps1"], dtype=np.float64)
    eps2 = np.asarray(data["eps2"], dtype=np.float64)
    rho = np.asarray(data["rho"], dtype=np.float64)
    mu = np.asarray(data["mu"], dtype=np.float64)

    n = len(QT)
    z0 = np.zeros((n, 3), dtype=np.float64)
    z0[:, 0] = QT / 2.0
    for i in range(n):
        qh = QT[i] / 2.0
        z0[i, 1] = solve_x_from_Q(qh, D1[i], eps1[i], rho[i], mu[i])
        z0[i, 2] = solve_x_from_Q(qh, D2[i], eps2[i], rho[i], mu[i])
    return z0


def residual_metrics(pred, data):
    params = {k: np.asarray(data[k], dtype=np.float64) for k in ["Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]}
    F = system_F(pred.astype(np.float64), params)
    norms_inf = np.max(np.abs(F), axis=1)
    valid = np.all(np.isfinite(F), axis=1)
    return {
        "valid_ratio": float(np.mean(valid)),
        "residual_mean": float(np.nanmean(norms_inf)),
        "residual_median": float(np.nanmedian(norms_inf)),
        "residual_p90": percentile(norms_inf[np.isfinite(norms_inf)], 90),
    }


# =========================================================
# Prompt and parsing
# =========================================================
SYSTEM_PROMPT = """You are solving a coupled nonlinear Colebrook-type system.
Return ONLY a JSON object with keys Q1, x1, x2.
No explanation. No markdown. No prose.
The result must satisfy:
1) branch-1 Colebrook equation
2) branch-2 Colebrook equation
3) equal head-loss coupling
4) 0 < Q1 < Q_total and x1 > 0 and x2 > 0
"""

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def format_sample_prompt(data: Dict[str, np.ndarray], idx: int, include_taylor: bool = True, coeff_round: int = 8) -> str:
    QT = float(data["Q_total"][idx])
    D1 = float(data["D1"][idx]); D2 = float(data["D2"][idx])
    eps1 = float(data["eps1"][idx]); eps2 = float(data["eps2"][idx])
    L1 = float(data["L1"][idx]); L2 = float(data["L2"][idx])
    rho = float(data["rho"][idx]); mu = float(data["mu"][idx]); g = float(data["g"][idx])
    center = np.asarray(data["center"][idx], dtype=np.float64).tolist()
    coeffs = np.asarray(data["coeffs"][idx], dtype=np.float64)

    parts = []
    parts.append("Solve the following coupled Colebrook-type nonlinear system.")
    parts.append(f"Q_total = {QT:.12g}")
    parts.append(f"D1 = {D1:.12g}, D2 = {D2:.12g}")
    parts.append(f"eps1 = {eps1:.12g}, eps2 = {eps2:.12g}")
    parts.append(f"L1 = {L1:.12g}, L2 = {L2:.12g}")
    parts.append(f"rho = {rho:.12g}, mu = {mu:.12g}, g = {g:.12g}")
    parts.append("")
    parts.append("Unknowns: Q1, x1, x2")
    parts.append("Q2 = Q_total - Q1")
    parts.append("")
    parts.append("Definitions:")
    parts.append("Re1 = 4*rho*Q1 / (pi*mu*D1)")
    parts.append("Re2 = 4*rho*Q2 / (pi*mu*D2)")
    parts.append("rr1 = eps1 / D1")
    parts.append("rr2 = eps2 / D2")
    parts.append("")
    parts.append("Equations:")
    parts.append("1) x1 + 2*log10(rr1/3.7 + 2.51*x1/Re1) = 0")
    parts.append("2) x2 + 2*log10(rr2/3.7 + 2.51*x2/Re2) = 0")
    parts.append("3) 8*L1*Q1^2 / (g*pi^2*D1^5*x1^2) = 8*L2*Q2^2 / (g*pi^2*D2^5*x2^2)")
    parts.append("")
    parts.append("Constraints:")
    parts.append("0 < Q1 < Q_total")
    parts.append("x1 > 0")
    parts.append("x2 > 0")
    parts.append("")
    if include_taylor:
        parts.append(f"Taylor centers = {center}")
        parts.append("Flattened Taylor coefficients:")
        parts.append(json.dumps([round(float(v), coeff_round) for v in coeffs.reshape(-1)], ensure_ascii=False))
        parts.append("")
    parts.append('Return ONLY JSON: {"Q1": ..., "x1": ..., "x2": ...}')
    return "\n".join(parts)


def parse_llm_json(text: str) -> Optional[Dict[str, float]]:
    if text is None:
        return None
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    for k in ["Q1", "x1", "x2"]:
        if k not in obj:
            return None
        try:
            out[k] = float(obj[k])
        except Exception:
            return None
    return out


# =========================================================
# Providers
# =========================================================
def call_openai_compatible(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 200,
    timeout: int = 120,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    if requests is None:
        raise RuntimeError("requests 패키지가 필요합니다.")
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if extra_headers:
        headers.update(extra_headers)
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    t0 = time.time()
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    latency = time.time() - t0
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    usage["latency_sec"] = latency
    return text, usage


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 200,
    timeout: int = 120,
    referer: str = "",
    title: str = "",
) -> Tuple[str, Dict[str, Any]]:
    headers = {}
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return call_openai_compatible(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_headers=headers,
    )


def load_manual_jsonl(manual_jsonl: Path) -> Dict[int, Dict[str, Any]]:
    mapping = {}
    with open(manual_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            mapping[int(obj["index"])] = obj
    return mapping


# =========================================================
# Main experiment
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--provider", choices=["openrouter", "openai_compatible", "manual_jsonl", "export_prompts"], required=True)

    # provider args
    parser.add_argument("--api_key", default="")
    parser.add_argument("--base_url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--manual_jsonl", default="")
    parser.add_argument("--openrouter_referer", default="")
    parser.add_argument("--openrouter_title", default="")

    # experiment control
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--include_taylor", action="store_true")
    parser.add_argument("--coeff_round", type=int, default=8)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max_newton_iter", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.test_npz)
    y_true = np.asarray(data["target"], dtype=np.float64)
    heur_pred = build_heuristic_baseline(data)

    n_total = len(y_true)
    idxs = list(range(args.start_index, n_total))
    if args.max_samples > 0:
        idxs = idxs[: args.max_samples]

    # export prompts only
    prompt_rows = []
    for idx in idxs:
        prompt_rows.append({
            "index": idx,
            "prompt": format_sample_prompt(data, idx, include_taylor=args.include_taylor, coeff_round=args.coeff_round),
        })

    if args.provider == "export_prompts":
        save_csv(out_dir / "prompts.csv", prompt_rows)
        save_jsonl(out_dir / "prompts.jsonl", prompt_rows)
        print(f"[DONE] prompts exported to {out_dir}")
        print("  - prompts.csv")
        print("  - prompts.jsonl")
        return

    manual_map = None
    if args.provider == "manual_jsonl":
        if not args.manual_jsonl:
            raise ValueError("--manual_jsonl 필요")
        manual_map = load_manual_jsonl(Path(args.manual_jsonl))

    llm_preds = []
    resp_rows = []

    for i, idx in enumerate(idxs, start=1):
        prompt = prompt_rows[i - 1]["prompt"]
        raw_text = ""
        usage = {}
        parsed = None
        err_msg = ""

        try:
            if args.provider == "openrouter":
                raw_text, usage = call_openrouter(
                    api_key=args.api_key,
                    model=args.model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    referer=args.openrouter_referer,
                    title=args.openrouter_title,
                )
            elif args.provider == "openai_compatible":
                raw_text, usage = call_openai_compatible(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
            else:
                item = manual_map.get(idx)
                if item is None:
                    raise KeyError(f"manual_jsonl에 index={idx} 응답이 없습니다.")
                raw_text = item.get("response_text", "")
                usage = item.get("usage", {}) if isinstance(item.get("usage", {}), dict) else {}

            parsed = parse_llm_json(raw_text)
            if parsed is None:
                err_msg = "JSON parse failed"
        except Exception as e:
            err_msg = str(e)

        if parsed is None:
            pred_vec = heur_pred[idx].astype(np.float64)  # fallback
            parsed_for_save = None
        else:
            pred_vec = np.array([parsed["Q1"], parsed["x1"], parsed["x2"]], dtype=np.float64)
            p = {k: float(np.asarray(data[k])[idx]) for k in ["Q_total", "D1", "D2", "eps1", "eps2", "L1", "L2", "rho", "mu", "g"]}
            pred_vec = project_feasible(pred_vec, p)
            parsed_for_save = parsed

        llm_preds.append(pred_vec)
        resp_rows.append({
            "index": idx,
            "ok": parsed_for_save is not None,
            "error": err_msg,
            "response_text": raw_text,
            "parsed_json": json.dumps(parsed_for_save, ensure_ascii=False) if parsed_for_save is not None else "",
            "latency_sec": usage.get("latency_sec", ""),
            "prompt_tokens": usage.get("prompt_tokens", ""),
            "completion_tokens": usage.get("completion_tokens", ""),
            "total_tokens": usage.get("total_tokens", ""),
        })
        print(f"[{i}/{len(idxs)}] index={idx} ok={parsed_for_save is not None}")

    llm_pred = np.stack(llm_preds, axis=0)
    y_sub = y_true[idxs]
    heur_sub = heur_pred[idxs]
    data_sub = {k: np.asarray(v)[idxs] for k, v in data.items()}

    rows = []

    hd = vector_metrics(heur_sub, y_sub)
    hd.update(residual_metrics(heur_sub, data_sub))
    hd["name"] = "heuristic_direct"
    rows.append(hd)

    href, hit, hconv = refine_batch(heur_sub.astype(np.float64), data_sub, tol=args.tol, max_iter=args.max_newton_iter)
    hr = vector_metrics(href, y_sub.astype(np.float64))
    hr.update(residual_metrics(href, data_sub))
    hr["name"] = "heuristic_plus_newton"
    hr["newton_iter_mean"] = float(np.mean(hit))
    hr["newton_iter_median"] = float(np.median(hit))
    hr["newton_iter_p90"] = float(np.percentile(hit, 90))
    hr["newton_converged_ratio"] = float(np.mean(hconv))
    rows.append(hr)

    ld = vector_metrics(llm_pred, y_sub)
    ld.update(residual_metrics(llm_pred, data_sub))
    ld["name"] = "llm_direct"
    rows.append(ld)

    lref, lit, lconv = refine_batch(llm_pred.astype(np.float64), data_sub, tol=args.tol, max_iter=args.max_newton_iter)
    lr = vector_metrics(lref, y_sub.astype(np.float64))
    lr.update(residual_metrics(lref, data_sub))
    lr["name"] = "llm_plus_newton"
    lr["newton_iter_mean"] = float(np.mean(lit))
    lr["newton_iter_median"] = float(np.median(lit))
    lr["newton_iter_p90"] = float(np.percentile(lit, 90))
    lr["newton_converged_ratio"] = float(np.mean(lconv))
    rows.append(lr)

    save_csv(out_dir / "summary_metrics.csv", rows)
    save_csv(out_dir / "responses.csv", resp_rows)
    save_csv(out_dir / "prompts.csv", prompt_rows)

    per_rows = []
    for j, idx in enumerate(idxs):
        per_rows.append({
            "index": idx,
            "true_Q1": float(y_sub[j, 0]),
            "true_x1": float(y_sub[j, 1]),
            "true_x2": float(y_sub[j, 2]),
            "heur_Q1": float(heur_sub[j, 0]),
            "heur_x1": float(heur_sub[j, 1]),
            "heur_x2": float(heur_sub[j, 2]),
            "llm_Q1": float(llm_pred[j, 0]),
            "llm_x1": float(llm_pred[j, 1]),
            "llm_x2": float(llm_pred[j, 2]),
            "llm_ref_Q1": float(lref[j, 0]),
            "llm_ref_x1": float(lref[j, 1]),
            "llm_ref_x2": float(lref[j, 2]),
            "llm_iter": int(lit[j]),
            "llm_converged": bool(lconv[j]),
        })
    save_csv(out_dir / "per_sample_results.csv", per_rows)

    config = vars(args).copy()
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for row in rows:
        print(row)
    print(f"\n[DONE] saved to: {out_dir}")
    print("  - summary_metrics.csv")
    print("  - per_sample_results.csv")
    print("  - responses.csv")
    print("  - prompts.csv")
    print("  - config.json")


if __name__ == "__main__":
    main()
