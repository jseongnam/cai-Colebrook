from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .physics import (
    baseline_initializer,
    colebrook_residual,
    head_loss,
    minimum_reynolds,
    reynolds,
    solve_reference,
)


ID_RANGES = {
    "Q_total": (0.01, 0.08),
    "D": (0.06, 0.35),
    "eps": (1e-5, 1.0e-3),
    "L": (50.0, 600.0),
    "rho": (990.0, 1005.0),
    "mu": (8e-4, 1.3e-3),
}


def _uniform(rng: np.random.Generator, bounds, size=None):
    return rng.uniform(bounds[0], bounds[1], size=size)


def sample_params(rng: np.random.Generator, branches: int, regime: str) -> dict:
    r = ID_RANGES
    qt = float(_uniform(rng, r["Q_total"]))
    d = _uniform(rng, r["D"], branches)
    eps = _uniform(rng, r["eps"], branches)
    length = _uniform(rng, r["L"], branches)
    rho = float(_uniform(rng, r["rho"]))
    mu = float(_uniform(rng, r["mu"]))

    if regime == "high_flow":
        qt = float(_uniform(rng, (0.085, 0.14)))
    elif regime == "high_roughness":
        eps = _uniform(rng, (1.2e-3, 3.0e-3), branches)
    elif regime == "extreme_diameter_ratio":
        d = _uniform(rng, (0.22, 0.45), branches)
        small = int(rng.integers(0, branches))
        d[small] = float(_uniform(rng, (0.035, 0.055)))
    elif regime == "combined":
        qt = float(_uniform(rng, (0.085, 0.14)))
        eps = _uniform(rng, (1.2e-3, 3.0e-3), branches)
        d = _uniform(rng, (0.20, 0.45), branches)
        d[int(rng.integers(0, branches))] = float(_uniform(rng, (0.04, 0.06)))
        mu = float(_uniform(rng, (5e-4, 7.5e-4)))
    elif regime != "iid":
        raise ValueError(f"Unknown regime: {regime}")

    return {
        "Q_total": qt,
        "D": np.asarray(d, dtype=np.float64),
        "eps": np.asarray(eps, dtype=np.float64),
        "L": np.asarray(length, dtype=np.float64),
        "rho": rho,
        "mu": mu,
        "g": 9.81,
    }


def generate_dataset(n: int, branches: int, regime: str, seed: int,
                     min_re: float = 4000.0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    attempts = 0
    while len(rows) < n:
        attempts += 1
        if attempts > max(500, 100 * n):
            raise RuntimeError(f"Only generated {len(rows)}/{n} valid samples")
        p = sample_params(rng, branches, regime)
        ref = solve_reference(p)
        if not ref.converged or minimum_reynolds(ref.q, p) < min_re:
            continue
        q0, x0 = baseline_initializer(p)
        rows.append({**p, "q_target": ref.q, "x_target": ref.x,
                     "q_baseline": q0, "x_baseline": x0,
                     "reference_residual": ref.residual_inf})

    scalar = ["Q_total", "rho", "mu", "g", "reference_residual"]
    vector = ["D", "eps", "L", "q_target", "x_target", "q_baseline", "x_baseline"]
    out = {k: np.asarray([row[k] for row in rows], dtype=np.float64) for k in scalar + vector}
    out["branches"] = np.asarray(branches, dtype=np.int64)
    out["regime"] = np.asarray(regime)
    out["seed"] = np.asarray(seed, dtype=np.int64)
    return out


def save_dataset(path: str | Path, data: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def sample_dict(data: Mapping[str, np.ndarray], index: int) -> dict:
    return {
        "Q_total": float(data["Q_total"][index]),
        "D": np.asarray(data["D"][index], dtype=np.float64),
        "eps": np.asarray(data["eps"][index], dtype=np.float64),
        "L": np.asarray(data["L"][index], dtype=np.float64),
        "rho": float(data["rho"][index]),
        "mu": float(data["mu"][index]),
        "g": float(data["g"][index]),
    }


def build_inference_features(data: Mapping[str, np.ndarray], input_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return leakage-safe branch and global encoder features.

    full: physical + explicit baseline state + baseline-derived residual features
    no_state: physical + baseline-derived residuals, but no q0/x0 state
    physics_only: no information computed from the baseline enters the encoder
    """
    if input_mode not in {"full", "no_state", "physics_only"}:
        raise ValueError(input_mode)
    n, branches = np.asarray(data["D"]).shape
    branch_rows, global_rows = [], []
    for i in range(n):
        p = sample_dict(data, i)
        q0 = np.asarray(data["q_baseline"][i], dtype=np.float64)
        x0 = np.asarray(data["x_baseline"][i], dtype=np.float64)
        qt = p["Q_total"]
        re0 = reynolds(q0, p["rho"], p["mu"], p["D"])
        static = np.column_stack([np.log(p["D"]), np.log(p["eps"]), np.log(p["L"])])
        if input_mode != "physics_only":
            cb = colebrook_residual(x0, re0, p["eps"] / p["D"])
            h = head_loss(q0, x0, p["L"], p["D"], p["g"])
            h_rel = (h - h.mean()) / max(float(np.max(np.abs(h))), 1e-9)
            residual_features = np.column_stack([np.log(np.maximum(re0, 1.0)), cb, h_rel])
            static = np.concatenate([static, residual_features], axis=1)
        if input_mode == "full":
            baseline_state = np.column_stack([
                np.log(np.clip(q0 / qt, 1e-10, 1.0)), np.log(np.maximum(x0, 1e-6))
            ])
            static = np.concatenate([static, baseline_state], axis=1)
        branch_rows.append(static)
        global_rows.append([
            np.log(qt), np.log(p["rho"]), np.log(p["mu"]), p["g"], float(branches)
        ])
    return np.asarray(branch_rows), np.asarray(global_rows)


def build_targets(data: Mapping[str, np.ndarray], target_mode: str = "correction",
                  flow_parameterization: str = "logit") -> np.ndarray:
    if target_mode not in {"correction", "direct"}:
        raise ValueError(target_mode)
    if flow_parameterization not in {"logit", "raw_ratio"}:
        raise ValueError(flow_parameterization)
    targets = []
    for i in range(len(np.asarray(data["Q_total"]))):
        qt = float(data["Q_total"][i])
        q0 = np.asarray(data["q_baseline"][i], dtype=np.float64)
        x0 = np.asarray(data["x_baseline"][i], dtype=np.float64)
        q_true = np.asarray(data["q_target"][i], dtype=np.float64)
        x_true = np.asarray(data["x_target"][i], dtype=np.float64)
        r_true = np.clip(q_true / qt, 1e-10, 1.0)
        r0 = np.clip(q0 / qt, 1e-10, 1.0)
        if flow_parameterization == "logit":
            flow_target = np.log(r_true)
            if target_mode == "correction":
                flow_target = flow_target - np.log(r0)
            flow_target -= flow_target.mean()
        elif target_mode == "correction":
            flow_target = r_true - r0
        else:
            flow_target = r_true
        x_target = x_true - x0 if target_mode == "correction" else x_true
        targets.append(np.column_stack([flow_target, x_target]))
    return np.asarray(targets)


def build_features(data: Mapping[str, np.ndarray], input_mode: str,
                   target_mode: str = "correction",
                   flow_parameterization: str = "logit") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    branch_x, global_x = build_inference_features(data, input_mode)
    target = build_targets(data, target_mode, flow_parameterization)
    return branch_x, global_x, target


def audit_legacy_npz(path: str | Path) -> dict:
    """Detect the target-as-center leakage present in some legacy generators."""
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        result = {"path": str(path), "keys": sorted(keys), "unsafe": False}
        if {"center", "target"} <= keys:
            center = np.asarray(z["center"], dtype=np.float64)
            target = np.asarray(z["target"], dtype=np.float64)
            same_shape = center.shape == target.shape
            exact_fraction = float(np.mean(np.all(np.isclose(center, target, rtol=1e-10, atol=1e-12), axis=-1))) if same_shape else 0.0
            result.update({"same_shape": same_shape, "center_equals_target_fraction": exact_fraction,
                           "unsafe": bool(exact_fraction > 0.01)})
        return result


def write_manifest(path: str | Path, payload: Mapping) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
