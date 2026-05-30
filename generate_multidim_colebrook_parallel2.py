#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

PI = math.pi
LN10 = math.log(10.0)


def re_from_Q(Q, rho, mu, D):
    return 4.0 * rho * Q / (PI * mu * D)


def colebrook_single_x_eq(x, Re, rel_rough):
    z = rel_rough / 3.7 + 2.51 * x / Re
    out = np.full_like(np.asarray(x, dtype=np.float64), np.nan, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    rel_rough = np.asarray(rel_rough, dtype=np.float64)
    mask = (Re > 0) & (z > 0)
    out[mask] = x[mask] + 2.0 * np.log10(z[mask])
    return out


def colebrook_single_x_df(x, Re, rel_rough):
    z = rel_rough / 3.7 + 2.51 * x / Re
    out = np.full_like(np.asarray(x, dtype=np.float64), np.nan, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    rel_rough = np.asarray(rel_rough, dtype=np.float64)
    mask = (Re > 0) & (z > 0)
    out[mask] = 1.0 + 2.0 * ((2.51 / Re[mask]) / (z[mask] * LN10))
    return out


def head_loss(Q, x, L, D, g):
    return 8.0 * L * (Q ** 2) / (g * (PI ** 2) * (D ** 5) * (x ** 2))


def solve_x_from_Q(Q, D, eps, rho, mu, x_init=7.0, tol=1e-13, max_iter=50):
    Qv = np.array([Q], dtype=np.float64)
    Dv = np.array([D], dtype=np.float64)
    epsv = np.array([eps], dtype=np.float64)
    rhov = np.array([rho], dtype=np.float64)
    muv = np.array([mu], dtype=np.float64)

    Re = re_from_Q(Qv, rhov, muv, Dv)
    rr = epsv / Dv

    x = np.array([x_init], dtype=np.float64)
    converged = False
    for _ in range(max_iter):
        fx = colebrook_single_x_eq(x, Re, rr)
        dfx = colebrook_single_x_df(x, Re, rr)
        if not np.isfinite(fx[0]) or not np.isfinite(dfx[0]) or abs(dfx[0]) < 1e-15:
            break
        x_new = x - fx / dfx
        x_new = np.clip(x_new, 1e-3, 1e3)
        if abs(x_new[0] - x[0]) < tol and abs(fx[0]) < tol:
            x = x_new
            converged = True
            break
        x = x_new

    resid = float(abs(colebrook_single_x_eq(x, Re, rr)[0]))
    return float(x[0]), resid, converged


def solve_parallel_system(QT, D1, D2, eps1, eps2, L1, L2, rho, mu, g, tol=1e-11, max_iter=80):
    def balance(q1):
        q2 = QT - q1
        if q1 <= 0 or q2 <= 0:
            return np.nan, np.nan, np.nan, False
        x1, _, ok1 = solve_x_from_Q(q1, D1, eps1, rho, mu)
        x2, _, ok2 = solve_x_from_Q(q2, D2, eps2, rho, mu)
        if not (ok1 and ok2):
            return np.nan, np.nan, np.nan, False
        h1 = head_loss(np.array([q1]), np.array([x1]), np.array([L1]), np.array([D1]), np.array([g]))[0]
        h2 = head_loss(np.array([q2]), np.array([x2]), np.array([L2]), np.array([D2]), np.array([g]))[0]
        return float(h1 - h2), float(x1), float(x2), True

    left = max(1e-8, QT * 1e-4)
    right = QT - max(1e-8, QT * 1e-4)

    qs = np.linspace(left, right, 120)
    vals, oks = [], []
    for q in qs:
        v, _, _, ok = balance(float(q))
        vals.append(v)
        oks.append(ok)
    vals = np.array(vals, dtype=np.float64)
    oks = np.array(oks, dtype=bool)

    bracket = None
    for i in range(len(qs)-1):
        if oks[i] and oks[i+1] and np.isfinite(vals[i]) and np.isfinite(vals[i+1]):
            if vals[i] == 0:
                bracket = (qs[i], qs[i])
                break
            if vals[i] * vals[i+1] < 0:
                bracket = (qs[i], qs[i+1])
                break

    if bracket is None:
        good = np.where(oks & np.isfinite(vals))[0]
        if len(good) == 0:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float64), np.nan, False
        j = good[np.argmin(np.abs(vals[good]))]
        q1 = float(qs[j])
        q2 = QT - q1
        x1, _, ok1 = solve_x_from_Q(q1, D1, eps1, rho, mu)
        x2, _, ok2 = solve_x_from_Q(q2, D2, eps2, rho, mu)
        if not (ok1 and ok2):
            return np.array([np.nan, np.nan, np.nan], dtype=np.float64), np.nan, False
        hbal = balance(q1)[0]
        Re1 = re_from_Q(np.array([q1]), np.array([rho]), np.array([mu]), np.array([D1]))[0]
        Re2 = re_from_Q(np.array([q2]), np.array([rho]), np.array([mu]), np.array([D2]))[0]
        f1 = colebrook_single_x_eq(np.array([x1]), np.array([Re1]), np.array([eps1 / D1]))[0]
        f2 = colebrook_single_x_eq(np.array([x2]), np.array([Re2]), np.array([eps2 / D2]))[0]
        res = max(abs(f1), abs(f2), abs(hbal))
        return np.array([q1, x1, x2], dtype=np.float64), float(res), True

    a, b = bracket
    fa, _, _, oka = balance(a)
    fb, _, _, okb = balance(b)
    if not (oka and okb):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64), np.nan, False

    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm, xm1, xm2, okm = balance(m)
        if not okm or not np.isfinite(fm):
            break
        if abs(fm) < tol or abs(b - a) < tol:
            q1 = m
            q2 = QT - q1
            x1, x2 = xm1, xm2
            Re1 = re_from_Q(np.array([q1]), np.array([rho]), np.array([mu]), np.array([D1]))[0]
            Re2 = re_from_Q(np.array([q2]), np.array([rho]), np.array([mu]), np.array([D2]))[0]
            f1 = colebrook_single_x_eq(np.array([x1]), np.array([Re1]), np.array([eps1 / D1]))[0]
            f2 = colebrook_single_x_eq(np.array([x2]), np.array([Re2]), np.array([eps2 / D2]))[0]
            res = max(abs(f1), abs(f2), abs(fm))
            return np.array([q1, x1, x2], dtype=np.float64), float(res), True
        if fa * fm < 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm

    q1 = 0.5 * (a + b)
    q2 = QT - q1
    x1, _, ok1 = solve_x_from_Q(q1, D1, eps1, rho, mu)
    x2, _, ok2 = solve_x_from_Q(q2, D2, eps2, rho, mu)
    if not (ok1 and ok2):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64), np.nan, False
    hbal = balance(q1)[0]
    Re1 = re_from_Q(np.array([q1]), np.array([rho]), np.array([mu]), np.array([D1]))[0]
    Re2 = re_from_Q(np.array([q2]), np.array([rho]), np.array([mu]), np.array([D2]))[0]
    f1 = colebrook_single_x_eq(np.array([x1]), np.array([Re1]), np.array([eps1 / D1]))[0]
    f2 = colebrook_single_x_eq(np.array([x2]), np.array([Re2]), np.array([eps2 / D2]))[0]
    res = max(abs(f1), abs(f2), abs(hbal))
    return np.array([q1, x1, x2], dtype=np.float64), float(res), True


def finite_diff_coeffs_1d(func, c, degree=25, h=1e-4):
    m = max(2 * degree + 3, 61)
    xs = np.linspace(c - h, c + h, m)
    ys = func(xs)
    if not np.all(np.isfinite(ys)):
        return np.full(degree + 1, np.nan, dtype=np.float64)
    u = xs - c
    p = np.polyfit(u, ys, degree)
    return p[::-1].astype(np.float64)


def build_coeffs_for_sample(z_star, QT, D1, D2, eps1, eps2, L1, L2, rho, mu, g, degree):
    Q1s, x1s, x2s = z_star
    Q2s = QT - Q1s
    rr1 = eps1 / D1
    rr2 = eps2 / D2

    def eq1_wrt_Q1(q_arr):
        q_arr = np.asarray(q_arr, dtype=np.float64)
        Re1 = re_from_Q(q_arr, np.full_like(q_arr, rho), np.full_like(q_arr, mu), np.full_like(q_arr, D1))
        return colebrook_single_x_eq(np.full_like(q_arr, x1s), Re1, np.full_like(q_arr, rr1))

    def eq2_wrt_x1(x_arr):
        x_arr = np.asarray(x_arr, dtype=np.float64)
        Re1s = re_from_Q(np.array([Q1s]), np.array([rho]), np.array([mu]), np.array([D1]))[0]
        return colebrook_single_x_eq(x_arr, np.full_like(x_arr, Re1s), np.full_like(x_arr, rr1))

    def eq3_wrt_x2(x_arr):
        x_arr = np.asarray(x_arr, dtype=np.float64)
        h1 = head_loss(np.full_like(x_arr, Q1s), np.full_like(x_arr, x1s), np.full_like(x_arr, L1), np.full_like(x_arr, D1), np.full_like(x_arr, g))
        h2 = head_loss(np.full_like(x_arr, Q2s), x_arr, np.full_like(x_arr, L2), np.full_like(x_arr, D2), np.full_like(x_arr, g))
        return h1 - h2

    coeff1 = finite_diff_coeffs_1d(eq1_wrt_Q1, Q1s, degree=degree, h=max(1e-5, 0.03 * max(Q1s, 1e-3)))
    coeff2 = finite_diff_coeffs_1d(eq2_wrt_x1, x1s, degree=degree, h=max(1e-5, 0.03 * max(abs(x1s), 1e-3)))
    coeff3 = finite_diff_coeffs_1d(eq3_wrt_x2, x2s, degree=degree, h=max(1e-5, 0.03 * max(abs(x2s), 1e-3)))

    coeffs = np.stack([coeff1, coeff2, coeff3], axis=0)
    center = np.array([Q1s, x1s, x2s], dtype=np.float64)
    return coeffs, center


def sample_one(rng):
    QT = float(rng.uniform(0.002, 0.08))
    D1 = float(rng.uniform(0.05, 0.50))
    D2 = float(rng.uniform(0.05, 0.50))
    eps1 = float(rng.uniform(1e-5, 3e-3))
    eps2 = float(rng.uniform(1e-5, 3e-3))
    L1 = float(rng.uniform(50.0, 800.0))
    L2 = float(rng.uniform(50.0, 800.0))
    rho = float(rng.uniform(980.0, 1005.0))
    mu = float(rng.uniform(7e-4, 1.4e-3))
    g = 9.81

    z_star, residual, ok = solve_parallel_system(QT, D1, D2, eps1, eps2, L1, L2, rho, mu, g)
    return {
        "ok": ok,
        "QT": QT, "D1": D1, "D2": D2, "eps1": eps1, "eps2": eps2,
        "L1": L1, "L2": L2, "rho": rho, "mu": mu, "g": g,
        "target": z_star, "residual": residual,
    }


def build_split(n_samples, degree, seed):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ["coeffs","center","target","Q_total","D1","D2","eps1","eps2","L1","L2","rho","mu","g","residual","expr_str"]}
    tries = 0
    while len(out["target"]) < n_samples:
        tries += 1
        if tries > n_samples * 80:
            raise RuntimeError("Too many failed samples. Relax parameter ranges.")
        sample = sample_one(rng)
        if not sample["ok"]:
            continue
        z_star = sample["target"]
        if not np.all(np.isfinite(z_star)):
            continue
        if sample["residual"] > 1e-8:
            continue
        coeffs, center = build_coeffs_for_sample(
            z_star, sample["QT"], sample["D1"], sample["D2"], sample["eps1"], sample["eps2"],
            sample["L1"], sample["L2"], sample["rho"], sample["mu"], sample["g"], degree
        )
        if not np.all(np.isfinite(coeffs)):
            continue
        out["coeffs"].append(coeffs)
        out["center"].append(center)
        out["target"].append(z_star)
        out["Q_total"].append(sample["QT"])
        out["D1"].append(sample["D1"])
        out["D2"].append(sample["D2"])
        out["eps1"].append(sample["eps1"])
        out["eps2"].append(sample["eps2"])
        out["L1"].append(sample["L1"])
        out["L2"].append(sample["L2"])
        out["rho"].append(sample["rho"])
        out["mu"].append(sample["mu"])
        out["g"].append(sample["g"])
        out["residual"].append(sample["residual"])
        out["expr_str"].append("Parallel-2-pipe Colebrook system: [F1(Q1,x1)=0, F2(Q1,x2)=0, F3(Q1,x1,x2)=0]")
        if len(out["target"]) % 1000 == 0:
            print(f"built {len(out['target'])}/{n_samples}")
    return {
        "coeffs": np.stack(out["coeffs"], axis=0).astype(np.float64),
        "center": np.stack(out["center"], axis=0).astype(np.float64),
        "target": np.stack(out["target"], axis=0).astype(np.float64),
        "Q_total": np.array(out["Q_total"], dtype=np.float64),
        "D1": np.array(out["D1"], dtype=np.float64),
        "D2": np.array(out["D2"], dtype=np.float64),
        "eps1": np.array(out["eps1"], dtype=np.float64),
        "eps2": np.array(out["eps2"], dtype=np.float64),
        "L1": np.array(out["L1"], dtype=np.float64),
        "L2": np.array(out["L2"], dtype=np.float64),
        "rho": np.array(out["rho"], dtype=np.float64),
        "mu": np.array(out["mu"], dtype=np.float64),
        "g": np.array(out["g"], dtype=np.float64),
        "residual": np.array(out["residual"], dtype=np.float64),
        "expr_str": np.array(out["expr_str"], dtype=object),
        "feature_desc": np.array(["coeffs shape=(N,3,degree+1): eq1 wrt Q1, eq2 wrt x1, eq3 wrt x2; center=(Q1_center,x1_center,x2_center); target=(Q1*,x1*,x2*)"], dtype=object),
    }


def save_npz(path: Path, data: Dict[str, np.ndarray]):
    np.savez_compressed(path, **data)
    print(f"[saved] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=4000)
    ap.add_argument("--degree", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="./multi_colebrook_data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating train split...")
    train = build_split(args.n_train, args.degree, seed=args.seed)
    save_npz(out_dir / f"parallel2_colebrook_deg{args.degree}_train.npz", train)

    print("Generating val split...")
    val = build_split(args.n_val, args.degree, seed=args.seed + 1)
    save_npz(out_dir / f"parallel2_colebrook_deg{args.degree}_val.npz", val)

    print("Generating test split...")
    test = build_split(args.n_test, args.degree, seed=args.seed + 2)
    save_npz(out_dir / f"parallel2_colebrook_deg{args.degree}_test.npz", test)

    print("[DONE]")
    print(f"Generated: parallel2_colebrook_deg{args.degree}_train/val/test.npz")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple
from numpy.polynomial import Chebyshev, Polynomial
import numpy as np

PI = math.pi
LN10 = math.log(10.0)


def re_from_Q(Q, rho, mu, D):
    return 4.0 * rho * Q / (PI * mu * D)


def colebrook_single_x_eq(x, Re, rel_rough):
    z = rel_rough / 3.7 + 2.51 * x / Re
    out = np.full_like(np.asarray(x, dtype=np.float64), np.nan, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    rel_rough = np.asarray(rel_rough, dtype=np.float64)
    mask = (Re > 0) & (z > 0)
    out[mask] = x[mask] + 2.0 * np.log10(z[mask])
    return out


def colebrook_single_x_df(x, Re, rel_rough):
    z = rel_rough / 3.7 + 2.51 * x / Re
    out = np.full_like(np.asarray(x, dtype=np.float64), np.nan, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    Re = np.asarray(Re, dtype=np.float64)
    rel_rough = np.asarray(rel_rough, dtype=np.float64)
    mask = (Re > 0) & (z > 0)
    out[mask] = 1.0 + 2.0 * ((2.51 / Re[mask]) / (z[mask] * LN10))
    return out


def head_loss(Q, x, L, D, g):
    return 8.0 * L * (Q ** 2) / (g * (PI ** 2) * (D ** 5) * (x ** 2))


def solve_x_from_Q(Q, D, eps, rho, mu, x_init=7.0, tol=1e-13, max_iter=50):
    Qv = np.array([Q], dtype=np.float64)
    Dv = np.array([D], dtype=np.float64)
    epsv = np.array([eps], dtype=np.float64)
    rhov = np.array([rho], dtype=np.float64)
    muv = np.array([mu], dtype=np.float64)

    Re = re_from_Q(Qv, rhov, muv, Dv)
    rr = epsv / Dv

    x = np.array([x_init], dtype=np.float64)
    converged = False
    for _ in range(max_iter):
        fx = colebrook_single_x_eq(x, Re, rr)
        dfx = colebrook_single_x_df(x, Re, rr)
        if not np.isfinite(fx[0]) or not np.isfinite(dfx[0]) or abs(dfx[0]) < 1e-15:
            break
        x_new = x - fx / dfx
        x_new = np.clip(x_new, 1e-3, 1e3)
        if abs(x_new[0] - x[0]) < tol and abs(fx[0]) < tol:
            x = x_new
            converged = True
            break
        x = x_new

    resid = float(abs(colebrook_single_x_eq(x, Re, rr)[0]))
    return float(x[0]), resid, converged


def solve_parallel_system(QT, D1, D2, eps1, eps2, L1, L2, rho, mu, g, tol=1e-11, max_iter=80):
    def balance(q1):
        q2 = QT - q1
        if q1 <= 0 or q2 <= 0:
            return np.nan, np.nan, np.nan, False
        x1, _, ok1 = solve_x_from_Q(q1, D1, eps1, rho, mu)
        x2, _, ok2 = solve_x_from_Q(q2, D2, eps2, rho, mu)
        if not (ok1 and ok2):
            return np.nan, np.nan, np.nan, False
        h1 = head_loss(np.array([q1]), np.array([x1]), np.array([L1]), np.array([D1]), np.array([g]))[0]
        h2 = head_loss(np.array([q2]), np.array([x2]), np.array([L2]), np.array([D2]), np.array([g]))[0]
        return float(h1 - h2), float(x1), float(x2), True

    left = max(1e-8, QT * 1e-4)
    right = QT - max(1e-8, QT * 1e-4)

    qs = np.linspace(left, right, 120)
    vals, oks = [], []
    for q in qs:
        v, _, _, ok = balance(float(q))
        vals.append(v)
        oks.append(ok)
    vals = np.array(vals, dtype=np.float64)
    oks = np.array(oks, dtype=bool)

    bracket = None
    for i in range(len(qs)-1):
        if oks[i] and oks[i+1] and np.isfinite(vals[i]) and np.isfinite(vals[i+1]):
            if vals[i] == 0:
                bracket = (qs[i], qs[i])
                break
            if vals[i] * vals[i+1] < 0:
                bracket = (qs[i], qs[i+1])
                break

    if bracket is None:
        good = np.where(oks & np.isfinite(vals))[0]
        if len(good) == 0:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float64), np.nan, False
        j = good[np.argmin(np.abs(vals[good]))]
        q1 = float(qs[j])
        q2 = QT - q1
        x1, _, ok1 = solve_x_from_Q(q1, D1, eps1, rho, mu)
        x2, _, ok2 = solve_x_from_Q(q2, D2, eps2, rho, mu)
        if not (ok1 and ok2):
            return np.array([np.nan, np.nan, np.nan], dtype=np.float64), np.nan, False
        hbal = balance(q1)[0]
        Re1 = re_from_Q(np.array([q1]), np.array([rho]), np.array([mu]), np.array([D1]))[0]
        Re2 = re_from_Q(np.array([q2]), np.array([rho]), np.array([mu]), np.array([D2]))[0]
        f1 = colebrook_single_x_eq(np.array([x1]), np.array([Re1]), np.array([eps1 / D1]))[0]
        f2 = colebrook_single_x_eq(np.array([x2]), np.array([Re2]), np.array([eps2 / D2]))[0]
        res = max(abs(f1), abs(f2), abs(hbal))
        return np.array([q1, x1, x2], dtype=np.float64), float(res), True

    a, b = bracket
    fa, _, _, oka = balance(a)
    fb, _, _, okb = balance(b)
    if not (oka and okb):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64), np.nan, False

    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm, xm1, xm2, okm = balance(m)
        if not okm or not np.isfinite(fm):
            break
        if abs(fm) < tol or abs(b - a) < tol:
            q1 = m
            q2 = QT - q1
            x1, x2 = xm1, xm2
            Re1 = re_from_Q(np.array([q1]), np.array([rho]), np.array([mu]), np.array([D1]))[0]
            Re2 = re_from_Q(np.array([q2]), np.array([rho]), np.array([mu]), np.array([D2]))[0]
            f1 = colebrook_single_x_eq(np.array([x1]), np.array([Re1]), np.array([eps1 / D1]))[0]
            f2 = colebrook_single_x_eq(np.array([x2]), np.array([Re2]), np.array([eps2 / D2]))[0]
            res = max(abs(f1), abs(f2), abs(fm))
            return np.array([q1, x1, x2], dtype=np.float64), float(res), True
        if fa * fm < 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm

    q1 = 0.5 * (a + b)
    q2 = QT - q1
    x1, _, ok1 = solve_x_from_Q(q1, D1, eps1, rho, mu)
    x2, _, ok2 = solve_x_from_Q(q2, D2, eps2, rho, mu)
    if not (ok1 and ok2):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64), np.nan, False
    hbal = balance(q1)[0]
    Re1 = re_from_Q(np.array([q1]), np.array([rho]), np.array([mu]), np.array([D1]))[0]
    Re2 = re_from_Q(np.array([q2]), np.array([rho]), np.array([mu]), np.array([D2]))[0]
    f1 = colebrook_single_x_eq(np.array([x1]), np.array([Re1]), np.array([eps1 / D1]))[0]
    f2 = colebrook_single_x_eq(np.array([x2]), np.array([Re2]), np.array([eps2 / D2]))[0]
    res = max(abs(f1), abs(f2), abs(hbal))
    return np.array([q1, x1, x2], dtype=np.float64), float(res), True

def finite_diff_coeffs_1d(func, c, degree=25, h=1e-4):
    m = max(3 * degree + 5, 121)

    xs = np.linspace(c - h, c + h, m)
    ys = func(xs)

    if not np.all(np.isfinite(ys)):
        return np.full(degree + 1, np.nan, dtype=np.float64)

    u = xs - c
    scale = float(np.max(np.abs(u)))
    if not np.isfinite(scale) or scale < 1e-12:
        return np.full(degree + 1, np.nan, dtype=np.float64)

    us = u / scale

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cheb = Chebyshev.fit(us, ys, deg=degree, domain=[-1, 1])
            poly_scaled = cheb.convert(kind=Polynomial)
            p_scaled_asc = np.asarray(poly_scaled.coef, dtype=np.float64)
    except Exception:
        return np.full(degree + 1, np.nan, dtype=np.float64)

    if p_scaled_asc.shape[0] < degree + 1:
        p_scaled_asc = np.pad(p_scaled_asc, (0, degree + 1 - p_scaled_asc.shape[0]))

    p_scaled_asc = p_scaled_asc[: degree + 1]

    coeff_u = np.zeros(degree + 1, dtype=np.float64)
    for k in range(degree + 1):
        denom = scale ** k
        if not np.isfinite(denom) or denom == 0:
            return np.full(degree + 1, np.nan, dtype=np.float64)
        coeff_u[k] = p_scaled_asc[k] / denom

    if not np.all(np.isfinite(coeff_u)):
        return np.full(degree + 1, np.nan, dtype=np.float64)

    return coeff_u
def build_coeffs_for_sample(z_star, QT, D1, D2, eps1, eps2, L1, L2, rho, mu, g, degree):
    Q1s, x1s, x2s = z_star
    Q2s = QT - Q1s
    rr1 = eps1 / D1
    rr2 = eps2 / D2

    def eq1_wrt_Q1(q_arr):
        q_arr = np.asarray(q_arr, dtype=np.float64)
        Re1 = re_from_Q(q_arr, np.full_like(q_arr, rho), np.full_like(q_arr, mu), np.full_like(q_arr, D1))
        return colebrook_single_x_eq(np.full_like(q_arr, x1s), Re1, np.full_like(q_arr, rr1))

    def eq2_wrt_x1(x_arr):
        x_arr = np.asarray(x_arr, dtype=np.float64)
        Re1s = re_from_Q(np.array([Q1s]), np.array([rho]), np.array([mu]), np.array([D1]))[0]
        return colebrook_single_x_eq(x_arr, np.full_like(x_arr, Re1s), np.full_like(x_arr, rr1))

    def eq3_wrt_x2(x_arr):
        x_arr = np.asarray(x_arr, dtype=np.float64)
        h1 = head_loss(np.full_like(x_arr, Q1s), np.full_like(x_arr, x1s), np.full_like(x_arr, L1), np.full_like(x_arr, D1), np.full_like(x_arr, g))
        h2 = head_loss(np.full_like(x_arr, Q2s), x_arr, np.full_like(x_arr, L2), np.full_like(x_arr, D2), np.full_like(x_arr, g))
        return h1 - h2

    coeff1 = finite_diff_coeffs_1d(eq1_wrt_Q1, Q1s, degree=degree, h=max(1e-5, 0.03 * max(Q1s, 1e-3)))
    coeff2 = finite_diff_coeffs_1d(eq2_wrt_x1, x1s, degree=degree, h=max(1e-5, 0.03 * max(abs(x1s), 1e-3)))
    coeff3 = finite_diff_coeffs_1d(eq3_wrt_x2, x2s, degree=degree, h=max(1e-5, 0.03 * max(abs(x2s), 1e-3)))

    coeffs = np.stack([coeff1, coeff2, coeff3], axis=0)
    center = np.array([Q1s, x1s, x2s], dtype=np.float64)
    return coeffs, center


def sample_one(rng):
    QT = float(rng.uniform(0.002, 0.08))
    D1 = float(rng.uniform(0.05, 0.50))
    D2 = float(rng.uniform(0.05, 0.50))
    eps1 = float(rng.uniform(1e-5, 3e-3))
    eps2 = float(rng.uniform(1e-5, 3e-3))
    L1 = float(rng.uniform(50.0, 800.0))
    L2 = float(rng.uniform(50.0, 800.0))
    rho = float(rng.uniform(980.0, 1005.0))
    mu = float(rng.uniform(7e-4, 1.4e-3))
    g = 9.81

    z_star, residual, ok = solve_parallel_system(QT, D1, D2, eps1, eps2, L1, L2, rho, mu, g)
    return {
        "ok": ok,
        "QT": QT, "D1": D1, "D2": D2, "eps1": eps1, "eps2": eps2,
        "L1": L1, "L2": L2, "rho": rho, "mu": mu, "g": g,
        "target": z_star, "residual": residual,
    }


def build_split(n_samples, degree, seed):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ["coeffs","center","target","Q_total","D1","D2","eps1","eps2","L1","L2","rho","mu","g","residual","expr_str"]}
    tries = 0
    while len(out["target"]) < n_samples:
        tries += 1
        if tries > n_samples * 80:
            raise RuntimeError("Too many failed samples. Relax parameter ranges.")
        sample = sample_one(rng)
        if not sample["ok"]:
            continue
        z_star = sample["target"]
        if not np.all(np.isfinite(z_star)):
            continue
        if sample["residual"] > 1e-8:
            continue
        coeffs, center = build_coeffs_for_sample(
            z_star, sample["QT"], sample["D1"], sample["D2"], sample["eps1"], sample["eps2"],
            sample["L1"], sample["L2"], sample["rho"], sample["mu"], sample["g"], degree
        )
        if not np.all(np.isfinite(coeffs)):
            continue
        out["coeffs"].append(coeffs)
        out["center"].append(center)
        out["target"].append(z_star)
        out["Q_total"].append(sample["QT"])
        out["D1"].append(sample["D1"])
        out["D2"].append(sample["D2"])
        out["eps1"].append(sample["eps1"])
        out["eps2"].append(sample["eps2"])
        out["L1"].append(sample["L1"])
        out["L2"].append(sample["L2"])
        out["rho"].append(sample["rho"])
        out["mu"].append(sample["mu"])
        out["g"].append(sample["g"])
        out["residual"].append(sample["residual"])
        out["expr_str"].append("Parallel-2-pipe Colebrook system: [F1(Q1,x1)=0, F2(Q1,x2)=0, F3(Q1,x1,x2)=0]")
        if len(out["target"]) % 1000 == 0:
            print(f"built {len(out['target'])}/{n_samples}")
    return {
        "coeffs": np.stack(out["coeffs"], axis=0).astype(np.float64),
        "center": np.stack(out["center"], axis=0).astype(np.float64),
        "target": np.stack(out["target"], axis=0).astype(np.float64),
        "Q_total": np.array(out["Q_total"], dtype=np.float64),
        "D1": np.array(out["D1"], dtype=np.float64),
        "D2": np.array(out["D2"], dtype=np.float64),
        "eps1": np.array(out["eps1"], dtype=np.float64),
        "eps2": np.array(out["eps2"], dtype=np.float64),
        "L1": np.array(out["L1"], dtype=np.float64),
        "L2": np.array(out["L2"], dtype=np.float64),
        "rho": np.array(out["rho"], dtype=np.float64),
        "mu": np.array(out["mu"], dtype=np.float64),
        "g": np.array(out["g"], dtype=np.float64),
        "residual": np.array(out["residual"], dtype=np.float64),
        "expr_str": np.array(out["expr_str"], dtype=object),
        "feature_desc": np.array(["coeffs shape=(N,3,degree+1): eq1 wrt Q1, eq2 wrt x1, eq3 wrt x2; center=(Q1_center,x1_center,x2_center); target=(Q1*,x1*,x2*)"], dtype=object),
    }


def save_npz(path: Path, data: Dict[str, np.ndarray]):
    np.savez_compressed(path, **data)
    print(f"[saved] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=4000)
    ap.add_argument("--degree", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="./multi_colebrook_data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating train split...")
    train = build_split(args.n_train, args.degree, seed=args.seed)
    save_npz(out_dir / f"parallel2_colebrook_deg{args.degree}_train.npz", train)

    print("Generating val split...")
    val = build_split(args.n_val, args.degree, seed=args.seed + 1)
    save_npz(out_dir / f"parallel2_colebrook_deg{args.degree}_val.npz", val)

    print("Generating test split...")
    test = build_split(args.n_test, args.degree, seed=args.seed + 2)
    save_npz(out_dir / f"parallel2_colebrook_deg{args.degree}_test.npz", test)

    print("[DONE]")
    print(f"Generated: parallel2_colebrook_deg{args.degree}_train/val/test.npz")


if __name__ == "__main__":
    main()
